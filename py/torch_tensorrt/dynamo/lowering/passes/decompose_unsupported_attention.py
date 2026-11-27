import logging
import operator
from typing import Any, Callable, Dict, Optional, Tuple

import tensorrt as trt
import torch
from torch._prims_common import compute_elementwise_output_logical_to_physical_perm
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._ConverterRegistry import ConverterRegistry
from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
    _PASS_THROUGH_OPS,
    _inputs_quantized_to_fp8,
    attention_capability_validator,
    scaled_dot_product_attention_validator,
    scaled_dot_product_cudnn_attention_validator,
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
_CUDNN = torch.ops.aten._scaled_dot_product_cudnn_attention.default
# The attention ops that take a bias and is_causal at the same positions and return a tuple.
_BIAS_ATTENTION = (_EFFICIENT, _CUDNN)
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


def _folded_half(node: torch.fx.Node) -> Optional[torch.dtype]:
    """The 16-bit type TensorRT runs this FP32 attention layer in, if it folds the casts of all its inputs."""
    query, key, value = node.args[:3]
    mask = node.args[3] if len(node.args) > 3 else None
    half = _upcast_source(query) if node.kwargs.get("scale") is not None else None
    if (
        half is not None
        and _upcast_source(key) == half == _upcast_source(value)
        and _dtype(mask) in (None, torch.bool)
        and _cast_back_to(node, half)
    ):
        return half
    return None


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
    elif _folded_half(node) is not None:
        return False
    return _decomposable(node)


def _decomposable(node: torch.fx.Node) -> bool:
    if node.target == _EFFICIENT:
        # The decomposition computes only the attention output, and it refuses a bias with is_causal.
        mask = node.args[3] if len(node.args) > 3 else None
        is_causal = len(node.args) > 6 and node.args[6]
        return not (is_causal and mask is not None) and all(
            item.args[1] == 0 for item in node.users if item.users
        )
    return True


# The longest key the rewrite leaves alone, by query dtype and hardware_compatible. Measured with TensorRT 11.4
# on an A100, an H100, a Jetson Thor and a Jetson Orin: FP16 is right up to 512 keys and wrong from 513, and BF16 is
# right up to 55296 and wrong from 57344. With hardware_compatible, measured on the A100 and the H100, FP16 keeps its
# edge and BF16 is right up to 7680 and wrong from 8192. Both BF16 limits keep a margin. TensorRT 11.2 and 11.3 have
# the same FP16 edge and a higher BF16 one. A head size that is not a multiple of 8 is masked correctly, and with a
# mask TensorRT cannot build it with more query rows. On the A100, giving shorter keys more query rows too was up to
# 1.2x slower in FP16, and in BF16 up to 1.4x slower at 4096 keys.
_SINGLE_QUERY_MASKED_KEYS = {
    (torch.float16, False): 512,
    (torch.float16, True): 512,
    (torch.bfloat16, False): 32768,
    (torch.bfloat16, True): 4096,
}


def _ignores_single_query_mask() -> bool:
    """Whether the installed TensorRT ignores the mask of a one-row query, as _single_query_mask_ignored describes.

    TensorRT 11.2 to 11.4 do, and TensorRT-RTX 1.7 does not. Other versions were not tested; the engine tests in
    test_attention_aten.py compare with PyTorch, so they fail on a version that has the bug but is not listed.
    """
    return bool(
        trt._package_name != "tensorrt_rtx"
        and trt.__version__.startswith(("11.2.", "11.3.", "11.4."))
    )


def _single_query_mask_ignored(
    node: torch.fx.Node,
    settings: CompilationSettings,
    dtype: Optional[torch.dtype] = None,
) -> bool:
    """Whether TensorRT would ignore the mask, or the causal mask, of this attention.

    Affected TensorRT versions do that for a 16-bit attention layer with a query length fixed at 1, a head size that
    is a multiple of 8 and a key that is not known to be short; see ``_SINGLE_QUERY_MASKED_KEYS``. Fused FP8
    attention, which was not measured, is kept: it has to read its quantize ops directly.
    """
    query, key = node.args[:2]
    dtype = dtype or _dtype(query)
    mask = node.args[3] if len(node.args) > 3 else None
    causal = 6 if node.target in _BIAS_ATTENTION else 5
    is_causal = len(node.args) > causal and node.args[causal]
    query_val = query.meta.get("val") if isinstance(query, torch.fx.Node) else None
    key_val = key.meta.get("val") if isinstance(key, torch.fx.Node) else None
    masked_keys = _SINGLE_QUERY_MASKED_KEYS.get((dtype, settings.hardware_compatible))
    if not (
        (mask is not None or bool(is_causal))
        and masked_keys is not None
        and not _inputs_quantized_to_fp8(node)
        and query_val is not None
        and key_val is not None
        and query_val.dim() == 4
        and statically_known_true(query_val.size(-2) == 1)
        and statically_known_true(query_val.size(-1) % 8 == 0)
        and not statically_known_true(key_val.size(-2) <= masked_keys)
        # An empty query has no element a mask could change.
        and not statically_known_true(query_val.numel() == 0)
        # The rewrite rebuilds only the attention output, so readers of the other outputs would get the wrong tensor.
        and (
            node.target not in _BIAS_ATTENTION
            or all(item.args[1] == 0 for item in node.users if item.users)
        )
    ):
        return False
    if _ignores_single_query_mask():
        return True
    logger.debug(
        "Keeping attention %s as it is: %s %s is not known to ignore the mask of a one-row query",
        node.name,
        trt._package_name,
        trt.__version__,
    )
    return False


def _foldable_query_heads(node: torch.fx.Node) -> bool:
    if (
        node.target != torch.ops.aten.scaled_dot_product_attention.default
        or not node.kwargs.get("enable_gqa", False)
        or (len(node.args) > 5 and node.args[5])
    ):
        return False
    query, key = (node.args[i].meta["val"] for i in (0, 1))
    heads, kv_heads, dim = query.size(-3), key.size(-3), query.size(-1)
    if not all(isinstance(size, int) for size in (heads, kv_heads, dim)):
        return False
    mask = node.args[3] if len(node.args) > 3 else None
    mask_val = mask.meta.get("val") if isinstance(mask, torch.fx.Node) else None
    # The heads of a mask with one entry per query head are regrouped, which needs its leading dimensions static.
    return heads > kv_heads and (
        mask_val is None
        or mask_val.dim() < 3
        or mask_val.size(-3) == 1
        or all(isinstance(size, int) for size in mask_val.shape[:-3])
    )


def _give_query_rows(gm: torch.fx.GraphModule, node: torch.fx.Node) -> None:
    """Gives a one-row attention more query rows, so TensorRT applies its mask.

    With grouped-query attention that ``_foldable_query_heads`` accepts (no causal mask, static head counts and head
    size, and a static batch for a mask per query head), the query heads of each key head become its query rows, so
    each key and value head is still read once. Otherwise the row is repeated and the first output row is kept.
    """
    query, key, value = node.args[:3]
    mask = node.args[3] if len(node.args) > 3 else None
    output_val = (
        node.meta["val"][0] if node.target in _BIAS_ATTENTION else node.meta["val"]
    )
    fold = _foldable_query_heads(node)

    def call(target: Any, *args: Any, val: Any = None, **kw: Any) -> torch.fx.Node:
        new = gm.graph.call_function(target, args, kw)
        new.meta["val"] = (
            target(*torch.fx.node.map_arg(args, lambda n: n.meta["val"]), **kw)
            if val is None
            else val
        )
        for name in ("nn_module_stack", "stack_trace", "source_fn_stack"):
            if name in node.meta:
                new.meta[name] = node.meta[name]
        return new

    kwargs = {k: v for k, v in node.kwargs.items() if not (fold and k == "enable_gqa")}
    with gm.graph.inserting_before(node):
        if fold:
            heads, dim = output_val.size(-3), output_val.size(-1)
            kv_heads = key.meta["val"].size(-3)
            query = call(
                torch.ops.aten.reshape.default,
                query,
                [-1, kv_heads, heads // kv_heads, query.meta["val"].size(-1)],
            )
            mask_val = mask.meta["val"] if isinstance(mask, torch.fx.Node) else None
            if mask_val is not None and mask_val.dim() >= 3 and mask_val.size(-3) > 1:
                mask = call(
                    torch.ops.aten.reshape.default,
                    mask,
                    [*mask_val.shape[:-3], kv_heads, heads // kv_heads, -1],
                )
            rows_val = output_val.reshape(*query.meta["val"].shape[:-1], dim)
        else:
            query = call(torch.ops.aten.cat.default, [query, query], 2)
            if (
                node.target in _BIAS_ATTENTION
                and isinstance(mask, torch.fx.Node)
                and mask.meta["val"].dim() >= 2
            ):
                # Efficient attention needs a bias row for every query row, and cuDNN attention accepts one too.
                # Unlike cat, expanding keeps the bias strides aligned, as the efficient PyTorch kernel requires.
                mask = call(
                    torch.ops.aten.expand.default,
                    mask,
                    [-1] * (mask.meta["val"].dim() - 2) + [2, -1],
                )
            rows_val = torch.cat([output_val, output_val], 2)
        # Running the attention on fake tensors can need a static key length, so its value is derived instead.
        if node.target in _BIAS_ATTENTION:
            rows_val = (rows_val, *node.meta["val"][1:])
        output = call(
            node.target,
            query,
            key,
            value,
            mask,
            *node.args[4:],
            val=rows_val,
            **kwargs,
        )
        if node.target in _BIAS_ATTENTION:
            output = call(operator.getitem, output, 0)
        if fold:
            output = call(torch.ops.aten.reshape.default, output, [-1, heads, 1, dim])
        else:
            output = call(torch.ops.aten.slice.Tensor, output, 2, 0, 1)
        # Where PyTorch runs it, attention returns a dense tensor of its own, in a memory order that a reader of its
        # strides relies on, such as the order of a transposed query. The result is a view, so it is copied to that
        # order.
        order, _ = compute_elementwise_output_logical_to_physical_perm(output_val)
        if order != sorted(order):
            output = call(torch.ops.aten.permute.default, output, order)
            output = call(
                torch.ops.aten.clone.default,
                output,
                memory_format=torch.contiguous_format,
            )
            output = call(
                torch.ops.aten.permute.default,
                output,
                [order.index(i) for i in range(len(order))],
            )
        output = call(torch.ops.aten.clone.default, output)
    if node.target in _BIAS_ATTENTION:
        for item in list(node.users):
            if item.users:
                item.replace_all_uses_with(output)
            gm.graph.erase_node(item)
    else:
        node.replace_all_uses_with(output)
    gm.graph.erase_node(node)


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
                env[n] = new_n
            elif n.op == "output":
                output = torch.fx.node.map_arg(n.args[0], lambda x: env[x])
            else:
                raise RuntimeError(
                    f"Unexpected op {n.op} while inlining attention decomposition"
                )

            if n.op in ("get_attr", "call_function"):
                # Copy only val and the module keys. The partitioner keeps connected nodes
                # without tensor_meta in one engine, which keeps the bool mask with the
                # attention. Module fallback needs the module keys, which the standalone
                # trace lacks.
                if "val" in n.meta:
                    env[n].meta["val"] = n.meta["val"]
                for key in ("nn_module_stack", "stack_trace", "source_fn_stack"):
                    if key in node.meta:
                        env[n].meta[key] = node.meta[key]

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
    decomposed as well when the decomposition can compute it. A 16-bit attention whose
    mask TensorRT ignores, with a query length fixed at 1 and a long key, gets more
    query rows instead.
    """
    if settings.decompose_attention:
        return gm

    # Decide for every node before changing the graph: whether a node is supported can depend on the other
    # attention nodes, such as two that read the same query.
    declined = []
    rewritten = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or (
            node.target not in _ATTENTION_FALLBACKS
            and node.target not in _BIAS_ATTENTION
        ):
            continue

        # Decomposition cannot make FP32 GEMMs supported on Turing, and for a constant key
        # it is less exact than PyTorch (NaN for a fully masked row, FP16 overflow). Preserve
        # the fused op so partitioning falls back to PyTorch for the whole attention.
        if not attention_capability_validator(node, settings) or _torch_executed(
            node, settings
        ):
            continue
        # An attention in a torch_executed_modules module runs in PyTorch as well.
        if node_in_torch_executed_module(node, settings.torch_executed_modules):
            continue

        if node.target in _ATTENTION_FALLBACKS:
            validator, decomp = _ATTENTION_FALLBACKS[node.target]
            if not validator(node, settings):
                declined.append((node, decomp))
                continue
        # The converter leaves the efficient and cuDNN attention it declines to PyTorch.
        elif not (
            scaled_dot_product_cudnn_attention_validator
            if node.target == _CUDNN
            else scaled_dot_product_efficient_attention_validator
        )(node, settings):
            continue
        if node.target in _FP32_ATTENTION and (
            _fp32_attention_layer_fails(node)
            # TensorRT runs this FP32 attention as a 16-bit layer, so it ignores the mask the same way. More query
            # rows would sit between the casts and the layer, so it is decomposed in FP32 instead.
            or (
                _decomposable(node)
                and _single_query_mask_ignored(
                    node, settings, _folded_half(node) or torch.float32
                )
            )
        ):
            declined.append((node, _FP32_ATTENTION[node.target]))
        elif _single_query_mask_ignored(node, settings):
            rewritten.append(node)

    for node in rewritten:
        logger.info("Giving the one-row query of attention %s more rows", node.name)
        _give_query_rows(gm, node)

    for node, decomp in declined:
        logger.info("Decomposing attention %s in place", node.name)
        _inline_traced_decomp(gm, node, decomp, settings)
    changed = bool(declined or rewritten)

    if changed:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug("After decompose_unsupported_attention:\n%s", gm.graph)

    return gm
