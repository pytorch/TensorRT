import logging
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._ConverterRegistry import ConverterRegistry
from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
    _PASS_THROUGH_OPS,
    _inputs_quantized_to_fp8,
    attention_capability_validator,
    scaled_dot_product_attention_validator,
    scaled_dot_product_efficient_attention_validator,
)
from torch_tensorrt.dynamo.lowering._decompositions import (
    get_decompositions,
    scaled_dot_product_attention_decomposition,
    scaled_dot_product_efficient_attention_decomposition,
)
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)
from torch_tensorrt.dynamo.partitioning.common import node_in_torch_executed_module

logger = logging.getLogger(__name__)

# Kept out of the default decomp table so IAttention can consume it when valid.
_ATTENTION_FALLBACKS: Dict[
    Any,
    Tuple[
        Callable[[torch.fx.Node, Optional[CompilationSettings]], bool],
        Callable[..., Any],
    ],
] = {
    torch.ops.aten.scaled_dot_product_attention.default: (
        scaled_dot_product_attention_validator,
        scaled_dot_product_attention_decomposition,
    ),
}

_EFFICIENT = torch.ops.aten._scaled_dot_product_efficient_attention.default
# The attention ops that take FP32 inputs (flash and cuDNN do not), with their decomposition.
_FP32_ATTENTION = {
    torch.ops.aten.scaled_dot_product_attention.default: scaled_dot_product_attention_decomposition,
    _EFFICIENT: scaled_dot_product_efficient_attention_decomposition,
}
# TensorRT folds a cast through these, but not through expand.
_VIEWS = _PASS_THROUGH_OPS - {
    torch.ops.aten._to_copy.default,
    torch.ops.aten.expand.default,
}
_HALF = (torch.float16, torch.bfloat16)


def _dtype(arg: Any) -> Optional[torch.dtype]:
    val = arg.meta.get("val") if isinstance(arg, torch.fx.Node) else None
    return getattr(val, "dtype", None)


def _upcast_source(arg: Any) -> Optional[torch.dtype]:
    """The 16-bit dtype that arg was cast to FP32 from, through views, or None."""
    while isinstance(arg, torch.fx.Node) and arg.target in _VIEWS:
        arg = arg.args[0]
    if (
        isinstance(arg, torch.fx.Node)
        and arg.target == torch.ops.aten._to_copy.default
        and _dtype(arg) == torch.float32
        and _dtype(arg.args[0]) in _HALF
    ):
        return _dtype(arg.args[0])
    return None


def _cast_back_to(node: torch.fx.Node, dtype: torch.dtype) -> bool:
    users = list(node.users)
    if node.target == _EFFICIENT:
        users = [user for item in users for user in item.users]
    while users:
        user = users.pop()
        if user.target in _VIEWS:
            users.extend(user.users)
        elif user.target != torch.ops.aten._to_copy.default or _dtype(user) != dtype:
            return False
    return True


def _fp32_attention_layer_fails(node: torch.fx.Node) -> bool:
    # TensorRT 11.4 folds a cast from FP16 or BF16 to FP32 into its attention layer when the cast reaches the key
    # or value, or the query through a given scale (a scale computed from the query's shape stops it). It then
    # builds only if it can run the whole layer in one 16-bit type: all three inputs folded from it, no float
    # mask, and the result cast back to it. Otherwise the build crashes, rejects the mask or finds no
    # implementation. Seen on an A100 and an H100. The decomposition keeps the attention in FP32, as asked.
    query, key, value = node.args[:3]
    folded = [
        _upcast_source(query) if node.kwargs.get("scale") is not None else None,
        _upcast_source(key),
        _upcast_source(value),
    ]
    mask = node.args[3] if len(node.args) > 3 else None
    if not any(folded):
        # An FP32 attention layer gives NaN for a query that the mask hides all keys from, where PyTorch and the
        # decomposition give zeros. TensorRT splits an FP32 attention layer into the same matmuls and softmax.
        # The fused FP8 attention is kept.
        # With no keys no query is hidden, and the empty matmuls of the decomposition do not build.
        if (
            mask is None
            or _dtype(query) != torch.float32
            or _inputs_quantized_to_fp8(node)
            or (
                key.meta.get("val") is not None
                and statically_known_true(key.meta["val"].size(-2) == 0)
            )
        ):
            return False
    elif (
        folded[0] is not None
        and folded.count(folded[0]) == 3
        and _dtype(mask) in (None, torch.bool)
        and _cast_back_to(node, folded[0])
    ):
        return False
    if node.target == _EFFICIENT:
        # The decomposition computes only the attention output, and it refuses a bias with is_causal.
        is_causal = len(node.args) > 6 and node.args[6]
        return not (is_causal and mask is not None) and all(
            item.args[1] == 0 for item in node.users if item.users
        )
    return True


def _torch_executed(node: torch.fx.Node, settings: CompilationSettings) -> bool:
    # The decomposition removes the op and its module metadata, so the partitioner could no longer keep it in PyTorch.
    excluded = settings.torch_executed_ops
    return (
        node.target in excluded
        or ConverterRegistry.qualified_name_or_str(node.target) in excluded
        or node_in_torch_executed_module(node, settings.torch_executed_modules)
    )


def _example_val(x: Any) -> Any:
    if isinstance(x, torch.fx.Node):
        val = x.meta.get("val")
        if val is None:
            raise RuntimeError(
                f"Cannot decompose attention node: {x} has no meta['val']"
            )
        return val
    return x


def _inline_traced_decomp(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    decomp_fn: Callable[..., Any],
    settings: CompilationSettings,
) -> None:
    """Replace ``node`` with the aten subgraph from tracing ``decomp_fn``."""

    example_args = tuple(_example_val(a) for a in node.args)
    example_kwargs = {k: _example_val(v) for k, v in node.kwargs.items()}
    example_kwargs["use_fp32_acc"] = settings.use_fp32_acc

    def wrapped(*args: Any) -> Any:
        return decomp_fn(*args, **example_kwargs)

    # Lowered with the same table as the rest of the graph: this pass runs after decomposition, and ops such
    # as aten.transpose.int have no converter of their own, so they would be left to PyTorch.
    traced = make_fx(
        wrapped,
        decomposition_table=get_decompositions(
            settings.enable_experimental_decompositions,
            settings.decompose_attention,
            settings.use_distributed_mode_trace,
            settings.use_fp32_acc,
        ),
        tracing_mode="fake",
    )(*example_args)

    arg_nodes = list(node.args)
    env: Dict[torch.fx.Node, Any] = {}
    ph_idx = 0
    output: Any = None

    with gm.graph.inserting_before(node):
        for n in traced.graph.nodes:
            if n.op == "placeholder":
                env[n] = arg_nodes[ph_idx]
                ph_idx += 1
            elif n.op == "get_attr":
                const = getattr(traced, n.target)
                const_name = f"_attn_fb_{node.name}_{n.target.replace('.', '_')}"
                gm.register_buffer(const_name, const.detach().clone())
                env[n] = gm.graph.get_attr(const_name)
            elif n.op == "call_function":
                new_args = torch.fx.node.map_arg(n.args, lambda x: env[x])
                new_kwargs = torch.fx.node.map_arg(n.kwargs, lambda x: env[x])
                new_n = gm.graph.call_function(n.target, new_args, new_kwargs)
                if "val" in n.meta:
                    new_n.meta["val"] = n.meta["val"]
                env[n] = new_n
            elif n.op == "output":
                output = torch.fx.node.map_arg(n.args[0], lambda x: env[x])
            else:
                raise RuntimeError(
                    f"Unexpected op {n.op} while inlining attention decomposition"
                )

    assert output is not None
    if isinstance(output, tuple):
        # The efficient attention op returns a tuple that its users read through getitem.
        for item in list(node.users):
            if item.users:
                item.replace_all_uses_with(output[item.args[1]])
            gm.graph.erase_node(item)
    else:
        node.replace_all_uses_with(output)
    gm.graph.erase_node(node)


def decompose_unsupported_attention(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Per-node fallback when the native attention converter declines a node.

    Attention ops are deliberately left out of the default decomposition table so
    ``IAttentionLayer`` can consume them whole. When the capability validator
    rejects a node (e.g. MLA where K and V head dims differ), leave the rest of
    the graph on the native path and decompose only the declined nodes. Capability
    rejections (TensorRT-RTX on Turing, a constant key) keep the fused op intact for
    PyTorch fallback. FP32 attention that TensorRT's attention layer fails to build, or
    that has a mask, for which the layer gives NaN where a query sees no key, is
    decomposed as well when the decomposition can compute it.
    """
    if settings.decompose_attention:
        return gm

    # Decide for every node before changing the graph: whether a node is supported can depend on the other
    # attention nodes, such as two that read the same query.
    declined = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or (
            node.target not in _ATTENTION_FALLBACKS
            and node.target not in _FP32_ATTENTION
        ):
            continue

        # Decomposition cannot make FP32 GEMMs supported on Turing, and for a constant key
        # it is less exact than PyTorch (NaN for a fully masked row, FP16 overflow). Preserve
        # the fused op so partitioning falls back to PyTorch for the whole attention.
        if not attention_capability_validator(node, settings) or _torch_executed(
            node, settings
        ):
            continue

        if node.target in _ATTENTION_FALLBACKS:
            validator, decomp = _ATTENTION_FALLBACKS[node.target]
            if not validator(node, settings):
                declined.append((node, decomp))
                continue
        # The converter leaves the efficient attention it declines to PyTorch.
        elif not scaled_dot_product_efficient_attention_validator(node, settings):
            continue
        if node.target in _FP32_ATTENTION and _fp32_attention_layer_fails(node):
            declined.append((node, _FP32_ATTENTION[node.target]))

    for node, decomp in declined:
        logger.info("Decomposing attention %s in place", node.name)
        _inline_traced_decomp(gm, node, decomp, settings)
    changed = bool(declined)

    if changed:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug("After decompose_unsupported_attention:\n%s", gm.graph)

    return gm
