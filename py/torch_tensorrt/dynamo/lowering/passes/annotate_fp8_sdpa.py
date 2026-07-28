import logging

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings

logger = logging.getLogger(__name__)

# FP8 E4M3 max. Softmax output is bounded to [0, 1], so 1/448 saturates at 1.0 exactly
# and is data-independent (no calibration required for the softmax output scale).
_FP8_E4M3_SOFTMAX_SCALE = 1.0 / 448.0

_SDPA_TARGETS = {
    torch.ops.aten.scaled_dot_product_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
}


def _is_fp8_quantize_op(node: torch.fx.Node) -> bool:
    """Return True when node is a tensorrt.quantize_op with FP8 dtype (exponent_bits=4)."""
    if node.op != "call_function":
        return False
    try:
        if node.target != torch.ops.tensorrt.quantize_op.default:
            return False
    except AttributeError:
        return False
    # args: (input, amax, num_bits, exponent_bits, ...)
    args = node.args
    return len(args) >= 4 and args[2] == 8 and args[3] == 4


def annotate_fp8_sdpa(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Annotate SDPA nodes whose Q, K, V inputs are all FP8-quantized.

    Detects the pattern emitted by modelopt when an attention module is
    registered via ``register_attention_for_kv_quant``, which wraps the
    Q, K, V arguments to ``F.scaled_dot_product_attention`` with
    ``q_bmm_quantizer``, ``k_bmm_quantizer``, ``v_bmm_quantizer``:

        q_fp8 = quantize_op(q, amax_q, num_bits=8, exponent_bits=4, ...)
        k_fp8 = quantize_op(k, amax_k, num_bits=8, exponent_bits=4, ...)
        v_fp8 = quantize_op(v, amax_v, num_bits=8, exponent_bits=4, ...)
        out   = scaled_dot_product_attention(q_fp8, k_fp8, v_fp8, ...)

    When all three inputs match this pattern the pass sets
    ``node.meta["_fp8_softmax_scale"] = 1/448`` on the SDPA node so the
    attention converter can set ``IAttention.normalization_quantize_to_type
    = FP8`` and ``IAttention.normalization_quantize_scale``, which TRT
    requires to fuse into the ``_gemm_mha_v2`` FP8 MHA kernel.
    """
    changed = False
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in _SDPA_TARGETS:
            continue
        if len(node.args) < 3:
            continue
        q_node, k_node, v_node = node.args[0], node.args[1], node.args[2]
        if not all(
            isinstance(n, torch.fx.Node) and _is_fp8_quantize_op(n)
            for n in (q_node, k_node, v_node)
        ):
            continue
        node.meta["_fp8_softmax_scale"] = _FP8_E4M3_SOFTMAX_SCALE
        changed = True
        logger.debug(
            f"Annotated SDPA node {node.name} with FP8 softmax scale "
            f"{_FP8_E4M3_SOFTMAX_SCALE} (Q/K/V inputs are FP8-quantized)"
        )

    if changed:
        logger.debug("FP8 SDPA softmax annotation complete")
    return gm
