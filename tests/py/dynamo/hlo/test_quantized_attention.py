"""Quantized attention tests using modelopt PTQ.

Two quantization schemes, each covering bmm quantization (q/k/v inputs to
the attention dot products) in addition to the linear projection layers:

  FP8_MHA_CONFIG  — FP8 E4M3 W8A8 linear + FP8 bmm quantizers
  INT8_MHA_CONFIG — INT8 W8A8 linear (per-channel weight, per-tensor
                    activation) + INT8 bmm quantizers

register_attention_for_kv_quant is called once at module level for both
VanillaAttention and GQAAttention to inject q_bmm / k_bmm / v_bmm
TensorQuantizer attributes.  mtq.quantize then matches the bmm-quantizer
patterns from each config and wires them into the SDPA forward pass.

VanillaAttention
----------------
  Q/K/V/out linear projections + F.scaled_dot_product_attention.
  Input shape: [batch, seq, embed_dim].  No mask; optional is_causal.

GQAAttention
------------
  MHA with separate Q and KV head counts (Hq != Hkv, Hq % Hkv == 0).
  Uses enable_gqa=True in SDPA.

Test classes
------------
  TestFP8MHAAttention    FP8_MHA_CONFIG — SDPA-based, IAttentionLayer fused kernel
  TestINT8MHAAttention   INT8_MHA_CONFIG — SDPA-based, IAttentionLayer fused kernel
  TestFP8EagerAttention  FP8_MHA_CONFIG — hand-rolled (HF ViT style), no IAttentionLayer
  TestINT8EagerAttention INT8_MHA_CONFIG — hand-rolled (HF ViT style), no IAttentionLayer

SDPA classes exercise (via _QuantAttentionMixin):
  test_static              fixed [batch, seq, embed_dim] inputs; causal + non-causal;
                           LLM-realistic shapes (Qwen2.5, Llama-3.2 style)
  test_dynamic_batch       dynamic batch dim; calibrated at opt_batch; causal variant
  test_dynamic_seq         dynamic seq dim; min=1 covers decode phase; causal variant
  test_edge_cases          seq=1 decode, single head, non-pow2 head_dim, causal prefill
  test_gqa                 GQA/MQA (Hq != Hkv) with bmm quantization
  test_mha_kernel_precision verify MHA kernel input dtype matches quantization dtype;
                           requires bmm quantizers to be enabled (via "*" wildcard
                           config, not "default" key) so quantize_op nodes appear in
                           the FX graph and become Q/DQ layers in TRT

Eager classes exercise (via _EagerAttentionMixin):
  test_static              ViT-realistic shapes plus generic configs
  test_dynamic_seq         dynamic seq dim covering ViT patch-count range

Precision checks
----------------
  _assert_mha_fused() (Check 1): asserts TRT selected a fused MHA kernel
    (called directly from each test method).  Checks for the _gemm_mha
    prefix.

  test_mha_kernel_precision (Check 2): asserts MHA kernel inputs carry the
    quantization dtype (FP8 / Int8).  For SDPA classes, this works via
    IAttentionLayer::normalization_quantize_scale.  For eager classes,
    _maybe_apply_matmul_quant detects upstream quantized GEMMs and inserts
    Q/DQ before the QK and AV matmul inputs with default scales.

Known TRT limitations (xfail)
------------------------------
  FP8 + dynamic min_seq=1 (TestFP8MHAAttention::test_dynamic_seq with min_s=1):
    Setting IAttentionLayer.normalization_quantize_to_type=FP8 prevents TRT
    10.x from selecting any fused MHA kernel when the dynamic sequence range
    includes seq=1. TRT needs a single kernel covering both _gemm_mha_v2
    (prefill, seq>1) and _gemv_mha_v1 (decode, seq=1), but no such kernel
    exists with FP8 normalization quantization in TRT 10.x. Static seq=1
    (decode) and dynamic min_seq>=2 are unaffected. INT8 is also unaffected.
"""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from ..conversion.harness import DispatchTestCase


def _cuda_sm_major() -> int:
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.get_device_capability()[0]


_SM_MAJOR = _cuda_sm_major()

# ---------------------------------------------------------------------------
# Quantization configs
# ---------------------------------------------------------------------------

FP8_MHA_CONFIG = {
    "quant_cfg": {
        "*": {"enable": False},
        "*weight_quantizer": {"num_bits": [4, 3], "axis": None},
        "*input_quantizer": {"num_bits": [4, 3], "axis": None},
        "*q_bmm_quantizer": {"num_bits": [4, 3], "axis": None},
        "*k_bmm_quantizer": {"num_bits": [4, 3], "axis": None},
        "*v_bmm_quantizer": {"num_bits": [4, 3], "axis": None},
        "*softmax_quantizer": {"num_bits": [4, 3], "axis": None},
        "*bmm2_output_quantizer": {"num_bits": [4, 3], "axis": None},
    },
    "algorithm": "max",
}

INT8_MHA_CONFIG = {
    "quant_cfg": {
        "*": {"enable": False},
        "*weight_quantizer": {"num_bits": 8, "axis": 0},
        "*input_quantizer": {"num_bits": 8, "axis": None},
        "*q_bmm_quantizer": {"num_bits": 8, "axis": None},
        "*k_bmm_quantizer": {"num_bits": 8, "axis": None},
        "*v_bmm_quantizer": {"num_bits": 8, "axis": None},
        "*softmax_quantizer": {"num_bits": 8, "axis": None},
        "*bmm2_output_quantizer": {"num_bits": 8, "axis": None},
    },
    "algorithm": "max",
}

# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------


class VanillaAttention(nn.Module):
    """MHA with Q/K/V/out projections and SDPA. Input: [batch, seq, embed_dim]."""

    def __init__(self, embed_dim: int, num_heads: int, is_causal: bool = False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.is_causal = is_causal
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, E = x.shape
        q = self.q_proj(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        return self.out_proj(out.transpose(1, 2).reshape(B, S, E))


class GQAAttention(nn.Module):
    """GQA/MQA with separate Q and KV head counts. Input: [batch, seq, embed_dim]."""

    def __init__(
        self,
        embed_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        is_causal: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_q_heads == 0
        assert num_q_heads % num_kv_heads == 0
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_q_heads
        self.is_causal = is_causal
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, E = x.shape
        q = (
            self.q_proj(x)
            .reshape(B, S, self.num_q_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .reshape(B, S, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .reshape(B, S, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=self.is_causal, enable_gqa=True
        )
        return self.out_proj(out.transpose(1, 2).reshape(B, S, E))


class EagerAttention(nn.Module):
    """Attention with explicit QK matmul, softmax, and AV matmul.

    Mirrors the HF ViT 'eager' attention pattern: decomposes attention into
    three separate ops (torch.matmul + F.softmax + torch.matmul) rather than
    delegating to F.scaled_dot_product_attention.  register_attention_for_kv_quant
    injects bmm quantizers on Q, K, and V (converted to Q/DQ pairs by torch-trt).
    The matmul converter detects the DQ producers on Q/K/V and inserts Q/DQ for
    the softmax output (attn), enabling TRT to fuse into the _gemm_mha kernel.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, E = x.shape
        q = self.q_proj(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self.out_proj(out.transpose(1, 2).reshape(B, S, E))


from modelopt.torch.quantization.plugins.attention import (
    register_attention_for_kv_quant,
)

register_attention_for_kv_quant(VanillaAttention)
register_attention_for_kv_quant(GQAAttention)
register_attention_for_kv_quant(EagerAttention)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_COMPILE_KWARGS = dict(
    use_explicit_typing=True,
    min_block_size=1,
    cache_built_engines=False,
    reuse_cached_engines=False,
)


def _export(model, example, dynamic_shapes=None):
    """Export a quantized model."""
    kwargs = dict(strict=False)
    if dynamic_shapes is not None:
        kwargs["dynamic_shapes"] = (dynamic_shapes,)
    try:
        return torch.export.export(model, (example,), **kwargs)
    except RuntimeError:
        return torch.export._trace._export(
            model,
            (example,),
            prefer_deferred_runtime_asserts_over_guards=True,
            **kwargs,
        )


def _get_trt_engine(trt_model):
    """Return the first TRT ICudaEngine found inside trt_model, or None.

    Handles both PythonTorchTensorRTModule (Python runtime, engine already
    deserialized) and TorchTensorRTModule (C++ runtime, holds serialized bytes).
    """
    import tensorrt as trt
    from torch_tensorrt.dynamo.runtime._PythonTorchTensorRTModule import (
        PythonTorchTensorRTModule,
    )
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule

    _LOGGER = trt.Logger(trt.Logger.WARNING)
    for _, mod in trt_model.named_modules():
        try:
            if isinstance(mod, PythonTorchTensorRTModule):
                return mod.engine
            if isinstance(mod, TorchTensorRTModule) and mod.serialized_engine:
                return trt.Runtime(_LOGGER).deserialize_cuda_engine(
                    mod.serialized_engine
                )
        except Exception:
            pass
    return None


def _collect_trt_layer_names(trt_model):
    """Return layer name strings from the engine using ONELINE format.

    Name strings like '__myl_TranCastMulCastReshMove_myl0_2' encode the fused
    op sequence TRT chose.  Returns [] when no engine is found or inspection
    fails.
    """
    import tensorrt as trt

    eng = _get_trt_engine(trt_model)
    if eng is None:
        return []
    try:
        insp = eng.create_engine_inspector()
        return [
            insp.get_layer_information(i, trt.LayerInformationFormat.ONELINE).strip()
            for i in range(eng.num_layers)
        ]
    except Exception:
        return []


def _collect_trt_layer_info_json(trt_model):
    """Return JSON-parsed layer info dicts from a DETAILED-verbosity engine.

    Requires the engine to have been compiled under a Debugger() context so
    that profilingVerbosity=DETAILED was set at build time.  When verbosity
    is LAYER_NAMES_ONLY, get_layer_information returns a bare name string
    rather than a JSON object; this function detects that and returns [].

    Each entry is a dict with keys 'Name', 'LayerType', 'TacticName',
    'Inputs', 'Outputs'.  'Inputs'/'Outputs' are lists of dicts with a
    'Format/Datatype' key (e.g. 'Half', 'FP8 linear', 'Int8 linear').
    """
    import json

    import tensorrt as trt

    eng = _get_trt_engine(trt_model)
    if eng is None:
        return []
    try:
        insp = eng.create_engine_inspector()
        layers = []
        for i in range(eng.num_layers):
            raw = insp.get_layer_information(i, trt.LayerInformationFormat.JSON)
            try:
                info = json.loads(raw)
                if isinstance(info, str):
                    return []  # LAYER_NAMES_ONLY verbosity — no IO dtype info
                layers.append(info)
            except (json.JSONDecodeError, ValueError):
                return []
        return layers
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Mixins
# ---------------------------------------------------------------------------


class _AttentionMixin:
    """Base mixin with shared state and precision checks for all attention tests."""

    QUANT_CFG = None
    # TRT fused MHA kernel prefixes: _gemm_mha (prefill, seq>1), _gemv_mha (decode, seq=1).
    _MHA_KERNEL_PREFIXES = ("_gemm_mha", "_gemv_mha")
    # Expected TRT Format/Datatype substring for quantized MHA inputs.
    # Subclasses must override (e.g. "FP8", "Int8").
    _QUANT_DTYPE = None

    def _assert_mha_fused(self, trt_model):
        """Check 1: Assert TRT produced a fused MHA kernel.

        Matches _gemm_mha (prefill, seq>1) and _gemv_mha (decode, seq=1).
        TRT fuses both SDPA-based (IAttentionLayer) and hand-rolled
        (matmul+softmax+matmul) attention into one of these kernels.
        """
        names = _collect_trt_layer_names(trt_model)
        if not names:
            return
        has_mha = any(
            any(prefix in n for prefix in self._MHA_KERNEL_PREFIXES) for n in names
        )
        self.assertTrue(
            has_mha,
            f"No fused MHA kernel {self._MHA_KERNEL_PREFIXES} found in TRT engine. "
            f"Layer names:\n" + "\n".join(f"  {n}" for n in names),
        )

    def test_mha_kernel_precision(self):
        """Check 2: MHA kernel inputs should match the quantization dtype.

        Asserts that the fused MHA kernel (_gemm_mha / _gemv_mha) receives
        inputs in the expected quantized precision (FP8 / Int8).
        """
        from modelopt.torch.quantization.utils import export_torch_mode
        from torch_tensorrt.dynamo.debug._Debugger import Debugger

        self.assertIsNotNone(
            self._QUANT_DTYPE,
            f"{type(self).__name__} must define _QUANT_DTYPE (e.g. 'FP8', 'Int8')",
        )

        embed_dim, num_heads, seq = 512, 8, 32
        dtype = torch.float16
        x = torch.randn(1, seq, embed_dim, dtype=dtype, device="cuda")
        model = self._quantize(embed_dim, num_heads, dtype, x)

        with torch.no_grad(), export_torch_mode():
            ep = _export(model, x)
            with Debugger():
                trt_model = torch_tensorrt.dynamo.compile(
                    ep, inputs=[x], **_COMPILE_KWARGS
                )

        layers = _collect_trt_layer_info_json(trt_model)
        self.assertTrue(
            layers,
            "Engine was not built with DETAILED verbosity — cannot inspect "
            "layer IO dtypes.  Ensure Debugger() is active during compile.",
        )

        mha_layer = next(
            (
                l
                for l in layers
                if any(p in l.get("Name", "") for p in self._MHA_KERNEL_PREFIXES)
            ),
            None,
        )
        self.assertIsNotNone(
            mha_layer,
            f"No fused MHA kernel {self._MHA_KERNEL_PREFIXES} found in DETAILED engine.",
        )

        bad_inputs = [
            inp
            for inp in mha_layer.get("Inputs", [])
            if self._QUANT_DTYPE not in inp.get("Format/Datatype", "")
        ]
        self.assertFalse(
            bad_inputs,
            f"MHA kernel '{mha_layer.get('Name')}' (tactic: {mha_layer.get('TacticName', '?')}) "
            f"inputs are not in quantized precision '{self._QUANT_DTYPE}':\n"
            + "\n".join(
                f"  {inp.get('Name', '?')}: {inp.get('Format/Datatype', '?')}"
                for inp in bad_inputs
            )
            + "\nAll input dtypes: "
            + str([inp.get("Format/Datatype") for inp in mha_layer.get("Inputs", [])]),
        )


class _QuantAttentionMixin(_AttentionMixin):
    """SDPA-based attention tests (VanillaAttention / GQAAttention)."""

    def _quantize(self, embed_dim, num_heads, dtype, calib_x, is_causal=False):
        import modelopt.torch.quantization as mtq

        model = (
            VanillaAttention(embed_dim, num_heads, is_causal).eval().cuda().to(dtype)
        )
        mtq.quantize(model, self.QUANT_CFG, forward_loop=lambda m: m(calib_x))
        return model

    def _quantize_gqa(
        self, embed_dim, num_q_heads, num_kv_heads, dtype, calib_x, is_causal=False
    ):
        import modelopt.torch.quantization as mtq

        model = (
            GQAAttention(embed_dim, num_q_heads, num_kv_heads, is_causal)
            .eval()
            .cuda()
            .to(dtype)
        )
        mtq.quantize(model, self.QUANT_CFG, forward_loop=lambda m: m(calib_x))
        return model

    # ------------------------------------------------------------------ static

    # fmt: off
    @parameterized.expand(
        [
            # (name,                    batch, heads,  seq, head_dim, is_causal, dtype)
            # --- non-causal ---
            ("b1_h8_s32_d32_nc_fp16",   1,  8,  32,   32, False, torch.float16),
            ("b1_h8_s32_d64_nc_fp16",   1,  8,  32,   64, False, torch.float16),
            ("b2_h8_s32_d64_nc_fp16",   2,  8,  32,   64, False, torch.float16),
            ("b1_h16_s64_d64_nc_fp16",  1, 16,  64,   64, False, torch.float16),
            ("b1_h8_s32_d64_nc_bf16",   1,  8,  32,   64, False, torch.bfloat16),
            # --- causal ---
            ("b1_h8_s32_d64_ca_fp16",   1,  8,  32,   64, True,  torch.float16),
            ("b2_h8_s128_d64_ca_fp16",  2,  8, 128,   64, True,  torch.float16),
            ("b1_h8_s32_d64_ca_bf16",   1,  8,  32,   64, True,  torch.bfloat16),
            # --- LLM-realistic ---
            ("qwen25_ca_fp16",          1, 14, 128,   64, True,  torch.float16),  # Qwen2.5-0.5B
        ]
    )
    # fmt: on
    def test_static(self, name, batch, heads, seq, head_dim, is_causal, dtype):
        from modelopt.torch.quantization.utils import export_torch_mode

        embed_dim = heads * head_dim
        x = torch.randn(batch, seq, embed_dim, dtype=dtype, device="cuda")
        model = self._quantize(embed_dim, heads, dtype, x, is_causal)
        with torch.no_grad():
            with export_torch_mode():
                ep = _export(model, x)
                trt_model = torch_tensorrt.dynamo.compile(
                    ep, inputs=[x], **_COMPILE_KWARGS
                )
        self._assert_mha_fused(trt_model)

    # --------------------------------------------------------------- dynamic batch

    # fmt: off
    @parameterized.expand(
        [
            # (name,                       min_b, opt_b, max_b, heads,  seq, head_dim, is_causal, dtype)
            ("b1to4_h8_s32_d64_nc_fp16",   1, 2, 4,  8,  32,  64, False, torch.float16),
            ("b1to8_h16_s32_d64_nc_fp16",  1, 4, 8, 16,  32,  64, False, torch.float16),
            ("b1to4_h8_s32_d64_ca_fp16",   1, 2, 4,  8,  32,  64, True,  torch.float16),
        ]
    )
    # fmt: on
    def test_dynamic_batch(
        self, name, min_b, opt_b, max_b, heads, seq, head_dim, is_causal, dtype
    ):
        from modelopt.torch.quantization.utils import export_torch_mode

        embed_dim = heads * head_dim
        x_calib = torch.randn(opt_b, seq, embed_dim, dtype=dtype, device="cuda")
        model = self._quantize(embed_dim, heads, dtype, x_calib, is_causal)
        batch_dim = torch.export.Dim("batch", min=min_b, max=max_b)
        with torch.no_grad():
            with export_torch_mode():
                ep = _export(model, x_calib, dynamic_shapes={0: batch_dim})
                trt_model = torch_tensorrt.dynamo.compile(
                    ep,
                    inputs=[
                        torch_tensorrt.Input(
                            min_shape=[min_b, seq, embed_dim],
                            opt_shape=[opt_b, seq, embed_dim],
                            max_shape=[max_b, seq, embed_dim],
                            dtype=dtype,
                        )
                    ],
                    **_COMPILE_KWARGS,
                )
                x_test = torch.randn(max_b, seq, embed_dim, dtype=dtype, device="cuda")
                trt_model(x_test)
        self._assert_mha_fused(trt_model)

    # --------------------------------------------------------------- dynamic seq

    # fmt: off
    @parameterized.expand(
        [
            # (name,                       batch, heads, min_s, opt_s, max_s, head_dim, is_causal, dtype)
            # min_s=1 cases are xfail for FP8: see module docstring "Known TRT limitations".
            ("b1_h8_s1to64_d64_nc_fp16",   1,  8,   1,  32,   64,  64, False, torch.float16),
            ("b1_h8_s1to128_d64_nc_fp16",  1,  8,   1,  64,  128,  64, False, torch.float16),
            ("b1_h8_s32to256_d64_nc_fp16", 1,  8,  32, 128,  256,  64, False, torch.float16),
            ("b1_h8_s1to128_d64_ca_fp16",  1,  8,   1,  64,  128,  64, True,  torch.float16),
        ]
    )
    # fmt: on
    def test_dynamic_seq(
        self, name, batch, heads, min_s, opt_s, max_s, head_dim, is_causal, dtype
    ):
        import pytest

        from modelopt.torch.quantization.utils import export_torch_mode

        if min_s == 1 and getattr(self, "_QUANT_DTYPE", None) == "FP8":
            pytest.xfail(
                "TRT 10.x limitation: normalization_quantize_to_type=FP8 on "
                "IAttentionLayer prevents fused MHA kernel selection for dynamic "
                "sequence ranges that include seq=1 (no kernel spans both "
                "_gemm_mha_v2 prefill and _gemv_mha_v1 decode with FP8 "
                "normalization quantization)."
            )

        embed_dim = heads * head_dim
        x_calib = torch.randn(batch, opt_s, embed_dim, dtype=dtype, device="cuda")
        model = self._quantize(embed_dim, heads, dtype, x_calib, is_causal)
        seq_dim = torch.export.Dim("seq", min=min_s, max=max_s)
        with torch.no_grad():
            with export_torch_mode():
                ep = _export(model, x_calib, dynamic_shapes={1: seq_dim})
                trt_model = torch_tensorrt.dynamo.compile(
                    ep,
                    inputs=[
                        torch_tensorrt.Input(
                            min_shape=[batch, min_s, embed_dim],
                            opt_shape=[batch, opt_s, embed_dim],
                            max_shape=[batch, max_s, embed_dim],
                            dtype=dtype,
                        )
                    ],
                    **_COMPILE_KWARGS,
                )
                x_test = torch.randn(
                    batch, max_s, embed_dim, dtype=dtype, device="cuda"
                )
                trt_model(x_test)
        self._assert_mha_fused(trt_model)

    # --------------------------------------------------------------- edge cases

    # fmt: off
    @parameterized.expand(
        [
            # (name,                       batch, heads,  seq, head_dim, is_causal, dtype)
            ("decode_b1_h8_s1_d64_fp16",   1,  8,    1,   64, False, torch.float16),  # decode
            ("b1_h1_s32_d64_fp16",         1,  1,   32,   64, False, torch.float16),  # single head
            ("b1_h8_s512_d64_fp16",        1,  8,  512,   64, False, torch.float16),  # large seq
            ("b1_h8_s128_d64_ca_fp16",     1,  8,  128,   64, True,  torch.float16),  # causal prefill
        ]
    )
    # fmt: on
    def test_edge_cases(self, name, batch, heads, seq, head_dim, is_causal, dtype):
        from modelopt.torch.quantization.utils import export_torch_mode

        embed_dim = heads * head_dim
        x = torch.randn(batch, seq, embed_dim, dtype=dtype, device="cuda")
        model = self._quantize(embed_dim, heads, dtype, x, is_causal)
        with torch.no_grad():
            with export_torch_mode():
                ep = _export(model, x)
                trt_model = torch_tensorrt.dynamo.compile(
                    ep, inputs=[x], **_COMPILE_KWARGS
                )
                trt_model(x)
        self._assert_mha_fused(trt_model)

    # --------------------------------------------------------------- GQA / MQA

    # fmt: off
    @parameterized.expand(
        [
            # (name,                       q_h, kv_h,  seq, head_dim, is_causal, dtype)
            ("gqa_8q_2kv_s64_nc_fp16",      8,   2,   64,   64, False, torch.float16),
            ("gqa_8q_2kv_s64_ca_fp16",      8,   2,   64,   64, True,  torch.float16),
            ("gqa_16q_4kv_s128_ca_fp16",   16,   4,  128,   64, True,  torch.float16),
            ("mqa_8q_1kv_s64_ca_fp16",      8,   1,   64,   64, True,  torch.float16),
            ("gqa_14q_2kv_s128_ca_fp16",   14,   2,  128,   64, True,  torch.float16),  # Qwen2.5
            ("gqa_decode_8q_2kv_fp16",      8,   2,    1,   64, False, torch.float16),  # decode
        ]
    )
    # fmt: on
    def test_gqa(
        self, name, num_q_heads, num_kv_heads, seq, head_dim, is_causal, dtype
    ):
        from modelopt.torch.quantization.utils import export_torch_mode

        embed_dim = num_q_heads * head_dim
        x = torch.randn(1, seq, embed_dim, dtype=dtype, device="cuda")
        model = self._quantize_gqa(
            embed_dim, num_q_heads, num_kv_heads, dtype, x, is_causal
        )
        with torch.no_grad():
            with export_torch_mode():
                ep = _export(model, x)
                trt_model = torch_tensorrt.dynamo.compile(
                    ep, inputs=[x], **_COMPILE_KWARGS
                )
                trt_model(x)
        self._assert_mha_fused(trt_model)


# ---------------------------------------------------------------------------
# Concrete test classes — one per quantization config
# ---------------------------------------------------------------------------


class TestFP8MHAAttention(_QuantAttentionMixin, DispatchTestCase):
    QUANT_CFG = FP8_MHA_CONFIG
    _QUANT_DTYPE = "FP8"


@unittest.skipIf(_SM_MAJOR >= 10, "INT8 quantized attention is not supported on SM100+")
class TestINT8MHAAttention(_QuantAttentionMixin, DispatchTestCase):
    QUANT_CFG = INT8_MHA_CONFIG
    _QUANT_DTYPE = "Int8"


# ---------------------------------------------------------------------------
# Eager attention (HF ViT style) — hand-rolled matmul+softmax+matmul
# ---------------------------------------------------------------------------


class _EagerAttentionMixin(_AttentionMixin):
    """Hand-rolled attention tests (HF ViT eager style, EagerAttention module).

    Only the linear projections are quantized by modelopt; bmm quantizer
    keys match nothing since SDPA is not used.  The matmul converter inserts
    Q/DQ pairs before the QK and AV matmuls (detected via upstream quantized
    GEMMs), causing TRT to fuse into _gemm_mha with an FP8/INT8 tactic.
    """

    def test_mha_kernel_precision(self):
        """MHA kernel inputs should be in the quantized dtype for eager attention.

        _maybe_apply_matmul_quant detects quantized linear-projection outputs
        via SHUFFLE→MATRIX_MULTIPLY traversal and inserts Q/DQ pairs before
        the QK and AV matmul inputs with default scales, causing TRT to select
        the FP8/INT8 tactic for the fused _gemm_mha_v2 kernel.
        """
        super().test_mha_kernel_precision()

    def _quantize(self, embed_dim, num_heads, dtype, calib_x, is_causal=False):
        import modelopt.torch.quantization as mtq

        model = EagerAttention(embed_dim, num_heads).eval().cuda().to(dtype)
        mtq.quantize(model, self.QUANT_CFG, forward_loop=lambda m: m(calib_x))
        return model

    # fmt: off
    @parameterized.expand(
        [
            # (name,                        batch, heads,  seq, head_dim, dtype)
            ("b1_h8_s32_d32_fp16",          1,  8,  32,  32, torch.float16),
            ("vit_base_b1_h12_s197_d64",    1, 12, 197,  64, torch.float16),  # ViT-Base
            ("vit_small_b1_h6_s197_d64",    1,  6, 197,  64, torch.float16),  # ViT-Small
            ("b1_h8_s64_d64_fp16",          1,  8,  64,  64, torch.float16),
            ("b2_h8_s32_d64_fp16",          2,  8,  32,  64, torch.float16),
            ("b1_h8_s32_d64_bf16",          1,  8,  32,  64, torch.bfloat16),
        ]
    )
    # fmt: on
    def test_static(self, name, batch, heads, seq, head_dim, dtype):
        from modelopt.torch.quantization.utils import export_torch_mode

        embed_dim = heads * head_dim
        x = torch.randn(batch, seq, embed_dim, dtype=dtype, device="cuda")
        model = self._quantize(embed_dim, heads, dtype, x)
        with torch.no_grad():
            with export_torch_mode():
                ep = _export(model, x)
                trt_model = torch_tensorrt.dynamo.compile(
                    ep, inputs=[x], **_COMPILE_KWARGS
                )
                trt_model(x)
        self._assert_mha_fused(trt_model)

    # fmt: off
    @parameterized.expand(
        [
            # (name,                        batch, heads, min_s, opt_s, max_s, head_dim, dtype)
            ("b1_h12_s50to197_d64_fp16",    1, 12,  50, 197, 197,  64, torch.float16),
            # min_s=1 is xfail for FP8: see module docstring "Known TRT limitations".
            ("b1_h8_s1to128_d64_fp16",      1,  8,   1,  64, 128,  64, torch.float16),
        ]
    )
    # fmt: on
    def test_dynamic_seq(
        self, name, batch, heads, min_s, opt_s, max_s, head_dim, dtype
    ):
        import pytest

        from modelopt.torch.quantization.utils import export_torch_mode

        if min_s == 1 and getattr(self, "_QUANT_DTYPE", None) == "FP8":
            pytest.xfail(
                "TRT 10.x limitation: FP8-quantized matmul+softmax+matmul cannot "
                "be fused into a single MHA kernel for dynamic sequence ranges "
                "that include seq=1. Same root cause as _QuantAttentionMixin: no "
                "kernel spans both _gemm_mha_v2 (prefill) and _gemv_mha_v1 "
                "(decode) with FP8 quantization."
            )

        embed_dim = heads * head_dim
        x_calib = torch.randn(batch, opt_s, embed_dim, dtype=dtype, device="cuda")
        model = self._quantize(embed_dim, heads, dtype, x_calib)
        seq_dim = torch.export.Dim("seq", min=min_s, max=max_s)
        with torch.no_grad():
            with export_torch_mode():
                ep = _export(model, x_calib, dynamic_shapes={1: seq_dim})
                trt_model = torch_tensorrt.dynamo.compile(
                    ep,
                    inputs=[
                        torch_tensorrt.Input(
                            min_shape=[batch, min_s, embed_dim],
                            opt_shape=[batch, opt_s, embed_dim],
                            max_shape=[batch, max_s, embed_dim],
                            dtype=dtype,
                        )
                    ],
                    **_COMPILE_KWARGS,
                )
                x_test = torch.randn(
                    batch, max_s, embed_dim, dtype=dtype, device="cuda"
                )
                trt_model(x_test)
        self._assert_mha_fused(trt_model)


class TestFP8EagerAttention(_EagerAttentionMixin, DispatchTestCase):
    QUANT_CFG = FP8_MHA_CONFIG
    _QUANT_DTYPE = "FP8"


@unittest.skipIf(_SM_MAJOR >= 10, "INT8 quantized attention is not supported on SM100+")
class TestINT8EagerAttention(_EagerAttentionMixin, DispatchTestCase):
    QUANT_CFG = INT8_MHA_CONFIG
    _QUANT_DTYPE = "Int8"


if __name__ == "__main__":
    run_tests()
