import logging
from typing import Optional

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings

from .annotate_fp8_sdpa import _is_fp8_quantize_op

logger = logging.getLogger(__name__)

_FP8_E4M3_SOFTMAX_AMAX = 1.0
_SOFTMAX_TARGETS = {
    torch.ops.aten._softmax.default,
    torch.ops.aten.softmax.int,
}
_MATMUL_TARGETS = {
    torch.ops.aten.matmul.default,
    torch.ops.aten.bmm.default,
}
# Shape-only ops that may sit between a quantize_op output and a matmul input.
_TRANSPARENT_TARGETS = {
    torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.reshape.default,
    torch.ops.aten._reshape_copy.default,
    torch.ops.aten.view.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.contiguous.default,
}


def _source_is_fp8_quantize(node: Optional[torch.fx.Node]) -> bool:
    """Walk through shape-transparent ops to find the producer; True if FP8 quantize_op."""
    seen: set[int] = set()
    cur = node
    while isinstance(cur, torch.fx.Node) and id(cur) not in seen:
        seen.add(id(cur))
        if _is_fp8_quantize_op(cur):
            return True
        if cur.op == "call_function" and cur.target in _TRANSPARENT_TARGETS:
            cur = cur.args[0] if cur.args else None
            continue
        return False
    return False


def _single_matmul_user(node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """Return the matmul user of ``node`` if it has exactly one and it is a matmul."""
    users = list(node.users)
    if len(users) != 1:
        return None
    user = users[0]
    if user.op != "call_function" or user.target not in _MATMUL_TARGETS:
        return None
    return user


def insert_fp8_softmax_qdq(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Insert an FP8 Q/DQ on softmax output in the decomposed FP8 MHA pattern.

    TRT's Method 2 FP8 MHA fusion requires FP8 Q/DQ on Q, K, V **and** on the
    softmax output.  modelopt's ``NVFP4_FP8_MHA_CONFIG`` specifies a
    ``*softmax_quantizer`` but the HF ``_QuantAttention.softmax_quantizer`` is
    only applied in the Triton FA path — not in the standard
    ``F.scaled_dot_product_attention`` path used by ``torch.export``.
    Consequently the exported FX graph has::

        matmul(q_fp8, k_fp8.T)  →  mul(1/sqrt(D))  →  softmax  →  matmul(·, v_fp8)

    with no FP8 Q/DQ between ``softmax`` and the second ``matmul``, so TRT
    keeps the two matmuls and the softmax as separate kernels instead of
    producing ``_gemm_mha_v2``.

    This pass recovers the fusion by inserting a ``tensorrt.quantize_op`` with
    ``num_bits=8, exponent_bits=4, amax=1.0`` (→ scale = 1/448) on the softmax
    output when the surrounding matmul inputs are FP8-quantized.  1/448 is
    data-independent because softmax output ∈ [0, 1].

    The pass is conservative: it fires only when *all three* of Q, K, V on the
    two matmuls trace back to FP8 ``tensorrt.quantize_op`` nodes.  If the
    graph is not a quantized MHA, nothing changes.
    """
    changed = False
    amax_buffer_idx = 0
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target not in _SOFTMAX_TARGETS:
            continue
        # The softmax must feed a single matmul (BMM2 = softmax_out @ V).
        bmm2 = _single_matmul_user(node)
        if bmm2 is None or len(bmm2.args) < 2:
            continue
        v_source = bmm2.args[1]
        if not _source_is_fp8_quantize(v_source):
            continue

        # Trace back from softmax to BMM1 through a possible scale/mul/div.
        attn_src = node.args[0] if node.args else None
        while (
            isinstance(attn_src, torch.fx.Node)
            and attn_src.op == "call_function"
            and attn_src.target
            in {
                torch.ops.aten.mul.Tensor,
                torch.ops.aten.div.Tensor,
                torch.ops.aten.add.Tensor,
                torch.ops.aten.sub.Tensor,
            }
        ):
            attn_src = attn_src.args[0]
        if not isinstance(attn_src, torch.fx.Node):
            continue
        if attn_src.op != "call_function" or attn_src.target not in _MATMUL_TARGETS:
            continue
        if len(attn_src.args) < 2:
            continue
        q_source, k_source = attn_src.args[0], attn_src.args[1]
        if not (
            _source_is_fp8_quantize(q_source) and _source_is_fp8_quantize(k_source)
        ):
            continue

        # Register a per-insertion amax buffer (1.0).
        amax_name = f"_fp8_softmax_qdq_amax_{amax_buffer_idx}"
        amax_buffer_idx += 1
        gm.register_buffer(
            amax_name,
            torch.tensor(_FP8_E4M3_SOFTMAX_AMAX, dtype=torch.float32),
            persistent=False,
        )

        with gm.graph.inserting_after(node):
            amax_node = gm.graph.create_node(
                "get_attr", amax_name, (), {}, name=amax_name
            )
        with gm.graph.inserting_after(amax_node):
            q_op = gm.graph.create_node(
                "call_function",
                torch.ops.tensorrt.quantize_op.default,
                (node, amax_node, 8, 4, False, False),
                {},
                name=f"fp8_softmax_quantize_{amax_buffer_idx - 1}",
            )

        # Re-route downstream matmul to read from the new quantize_op output.
        bmm2.replace_input_with(node, q_op)
        changed = True
        logger.debug(
            f"Inserted FP8 softmax Q/DQ after {node.name} "
            f"(scale=1/448, pattern=matmul→...→softmax→matmul with FP8 Q/K/V)"
        )

    if changed:
        gm.graph.lint()
        gm.recompile()
        logger.debug("FP8 decomposed-MHA softmax Q/DQ insertion complete")
    return gm
