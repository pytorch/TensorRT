import logging
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
    scaled_dot_product_attention_validator,
)
from torch_tensorrt.dynamo.lowering._decompositions import (
    scaled_dot_product_attention_decomposition,
)
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

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
    use_fp32_acc: bool,
) -> None:
    """Replace ``node`` with the aten subgraph from tracing ``decomp_fn``."""

    example_args = tuple(_example_val(a) for a in node.args)
    example_kwargs = {k: _example_val(v) for k, v in node.kwargs.items()}
    example_kwargs["use_fp32_acc"] = use_fp32_acc

    def wrapped(*args: Any) -> Any:
        return decomp_fn(*args, **example_kwargs)

    traced = make_fx(wrapped, tracing_mode="fake")(*example_args)

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
    node.replace_all_uses_with(output)
    gm.graph.erase_node(node)


def decompose_unsupported_attention(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Per-node fallback when the native attention converter declines a node.

    Attention ops are deliberately left out of the default decomposition table so
    ``IAttentionLayer`` can consume them whole. When the capability validator
    rejects a node (e.g. MLA where K and V head dims differ), leave the rest of
    the graph on the native path and decompose only the declined nodes.
    """
    if settings.decompose_attention:
        return gm

    changed = False
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target not in _ATTENTION_FALLBACKS:
            continue

        validator, decomp = _ATTENTION_FALLBACKS[node.target]
        if validator(node, settings):
            continue

        logger.info(
            "Attention converter declined %s; decomposing that node in place "
            "(native IAttention kept for other attention nodes)",
            node.name,
        )
        _inline_traced_decomp(gm, node, decomp, settings.use_fp32_acc)
        changed = True

    if changed:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug("After decompose_unsupported_attention:\n%s", gm.graph)

    return gm
