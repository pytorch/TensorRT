import logging

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def _fused_target(dequantize: torch.fx.Node) -> object:
    from torch_tensorrt.dynamo.conversion import experts_custom_op  # noqa: F401

    return {
        torch.ops.tensorrt.dequantize_nvfp4_experts.default: torch.ops.tensorrt.grouped_mm_nvfp4.default,
        torch.ops.tensorrt.dequantize_mxfp8_experts.default: torch.ops.tensorrt.grouped_mm_mxfp8.default,
    }.get(dequantize.target)


def _dequantized_weight(mat2: object) -> object:
    """The dequantize op whose [groups, N, K] result mat2 transposes to [groups, K, N], or None."""
    if not isinstance(mat2, torch.fx.Node) or mat2.op != "call_function":
        return None
    if mat2.target == torch.ops.aten.permute.default:
        if list(mat2.args[1]) not in ([0, 2, 1], [-3, -1, -2]):
            return None
    elif mat2.target == torch.ops.aten.transpose.int:
        if sorted(d % 3 for d in mat2.args[1:3]) != [1, 2]:
            return None
    else:
        return None
    source = mat2.args[0]
    if (
        isinstance(source, torch.fx.Node)
        and source.op == "call_function"
        and _fused_target(source) is not None
    ):
        return source
    return None


def fuse_dequantize_into_grouped_mm(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Turn dequantize_{nvfp4,mxfp8}_experts followed by aten._grouped_mm into one op.

    The fused op's kernel reads the packed weights, so the engine keeps them packed
    instead of holding a 16-bit copy. Runs only with grouped_mm_kernel set, before
    constant folding would dequantize the weights at compile time.
    """
    if not settings.grouped_mm_kernel:
        return gm
    from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
        _grouped_mm_kernel_blocked,
        _grouped_mm_kernel_unavailable,
    )

    if _grouped_mm_kernel_blocked(settings) or _grouped_mm_kernel_unavailable():
        return gm
    fused = 0
    for node in list(gm.graph.nodes):
        if (
            node.op != "call_function"
            or node.target != torch.ops.aten._grouped_mm.default
        ):
            continue
        args = list(node.args) + [None] * (6 - len(node.args))
        mat1, mat2 = args[0], args[1]
        offs = node.kwargs.get("offs", args[2])
        bias = node.kwargs.get("bias", args[3])
        out_dtype = node.kwargs.get("out_dtype", args[4])
        dequantize = _dequantized_weight(mat2)
        if (
            dequantize is None
            or bias is not None
            or not isinstance(offs, torch.fx.Node)
        ):
            continue
        mat1_val = mat1.meta.get("val") if isinstance(mat1, torch.fx.Node) else None
        if (
            mat1_val is None
            or mat1_val.dim() != 2
            or mat1_val.dtype not in (torch.bfloat16, torch.float16)
        ):
            continue
        if (
            out_dtype not in (None, mat1_val.dtype)
            or dequantize.args[-1] != mat1_val.dtype
        ):
            continue
        with gm.graph.inserting_before(node):
            replacement = gm.graph.call_function(
                _fused_target(dequantize), (mat1, *dequantize.args[:-1], offs)
            )
        replacement.meta.update(node.meta)
        node.replace_all_uses_with(replacement)
        gm.graph.erase_node(node)
        for dead in (mat2, dequantize):
            if not dead.users:
                gm.graph.erase_node(dead)
        fused += 1
    if fused:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(
            f"Fused {fused} dequantize + grouped_mm pairs into quantized grouped GEMMs"
        )
    return gm
