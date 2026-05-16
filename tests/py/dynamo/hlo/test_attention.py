"""Comprehensive attention subgraph tests for TRT converter bug discovery.

Covers all SDPA kernel variants, MHA/GQA/MQA attention patterns, causal vs
non-causal masking, bool/float/broadcast mask shapes, decode-phase attention
(seq_q=1), non-power-of-2 head dims, LLM-realistic configs, and multiple dtypes.

Known limitations
-----------------------------------------------------
  TensorRT 10.x (resolved in TRT 11.0) and TensorRT-RTX-1.4:
  For TensorRT 10.x, large causal sequences of k/v (seq >= 512, is_causal=True) in FP16/BF16
    IAttentionLayer produces ~80% element mismatch at long sequences. Thus, we use FP32 for
    the scale factor. If you want to use the accurate dtype, please set `decompose_attention=True`
    or upgrade to TRT 11.0 or later. TODO: @Evan to verify the version of TensorRT-RTX that
    resolves this bug.

Notes on attn_bias_is_causal
-----------------------------
  Default True: the force_causal_efficient_attention lowering pass strips
    attn_bias and sets is_causal=True before reaching the converter.
    This is an HF-model optimization; most production uses keep the default.
  Set False: attn_bias is forwarded to IAttentionLayer.mask.  Required for
    any test that validates actual bias tensor values.

Test classes
------------
  TestSDPA                - aten.scaled_dot_product_attention — all configurations:
      test_no_mask             no mask; IAttentionLayer native (decompose=True for large causal)
      test_decode              decode-phase (seq_q=1, seq_k>1); IAttentionLayer native
      test_bool_mask           bool attention masks (full, broadcast, 2-D, decode)
      test_float_mask          additive float attention masks (incl. decode-phase)
      test_gqa                 GQA/MQA (Hq != Hkv); IAttentionLayer native
  TestFlashAttention      - _scaled_dot_product_flash_attention kernel:
      test_no_mask             no mask; IAttentionLayer native (decompose=True for large causal)
      test_decode              decode-phase (seq_q=1, seq_k>1); IAttentionLayer native
      test_gqa                 GQA/MQA (Hq != Hkv); IAttentionLayer native
  TestEfficientAttention  - _scaled_dot_product_efficient_attention:
      test_no_bias             attn_bias=None; decompose=True
      test_with_bias           native IAttentionLayer.mask, incl. h=1/b=1 shapes
      test_with_bias_decode    decode-phase (seq_q=1, seq_k>1) + 4D bias
      test_with_bias_causal    is_causal=True + attn_bias combined converter path
      test_attn_bias_is_causal_opt  force_causal_efficient_attention pass
"""

import sys
import unittest

import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from ..conversion.harness import DispatchTestCase

_BF16_SKIP = unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8,
    "BF16 requires Ampere (SM80) or higher",
)

_FLASH_ATTN_SKIP = unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8,
    "Flash attention requires Ampere (SM80) or higher",
)

# skip on Windows
_WINDOWS_SKIP = unittest.skipIf(
    sys.platform == "win32",
    "This test is skipped on Windows because USE_FLASH_ATTENTION was not enabled for build",
)


def _skip_bf16_on_rtx(test_self, dtype):
    """Call at the top of a test to skip BF16 on TensorRT-RTX builds."""
    if dtype == torch.bfloat16 and getattr(
        torch_tensorrt.ENABLED_FEATURES, "tensorrt_rtx", False
    ):
        test_self.skipTest("TensorRT-RTX does not support bfloat16")


# ---------------------------------------------------------------------------
# Standard SDPA — all configurations
# ---------------------------------------------------------------------------


class TestSDPA(DispatchTestCase):
    """aten.scaled_dot_product_attention — all configurations.

    test_no_mask
        Standard MHA, no mask.

    test_decode
        Decode-step (seq_q=1, K/V span full context), no mask.
        IAttentionLayer handles non-square Q/K natively.

    test_bool_mask
        Bool attention masks (full, broadcast, 2D, decode-phase).
        Exercises the bool→float (-inf fill) mask conversion path.

    test_float_mask
        Additive float attention masks (added to QK^T before softmax).
        Exercises the add-bias path, distinct from the bool mask path.

    test_gqa
        GQA/MQA (Hq != Hkv).  IAttentionLayer native; no K/V expansion.
        FP32 and large causal (seq >= 512) are tested.
    """

    # fmt: off
    @parameterized.expand(
        [
            # (name, batch, heads, seq_q, seq_k, head_dim, is_causal, scale, dtype, use_decompose, test_atol)
            # --- FP16, varying batch ---
            ("b1_h8_s32_d64_nc_fp16",   1,  8,   32,   32,  64, False, None,  torch.float16, False, 1e-2),
            ("b1_h8_s32_d64_ca_fp16",   1,  8,   32,   32,  64, True,  None,  torch.float16, False, 1e-2),
            ("b2_h8_s128_d64_nc_fp16",  2,  8,  128,  128,  64, False, None,  torch.float16, False, 1e-2),
            ("b2_h8_s128_d64_ca_fp16",  2,  8,  128,  128,  64, True,  None,  torch.float16, False, 1e-2),
            ("b4_h8_s128_d64_fp16",     4,  8,  128,  128,  64, True,  None,  torch.float16, False, 1e-2),
            # --- FP16, varying num_heads ---
            ("h1_fp16",                 1,  1,   64,   64,  64, False, None,  torch.float16, False, 1e-2),
            ("h16_fp16",                2, 16,   64,   64,  64, False, None,  torch.float16, False, 1e-2),
            ("h32_fp16",                2, 32,  128,  128,  64, True,  None,  torch.float16, False, 1e-2),
            # --- FP16, varying head_dim ---
            ("d16_fp16",                2,  8,   64,   64,  16, False, None,  torch.float16, False, 1e-2),
            ("d32_fp16",                2,  8,   64,   64,  32, False, None,  torch.float16, False, 1e-2),
            ("d128_fp16",               1,  4,   64,   64, 128, False, None,  torch.float16, False, 1e-2),
            # Non-power-of-2 head dims
            ("d48_fp16",                1,  4,   32,   32,  48, False, None,  torch.float16, False, 1e-2),
            ("d96_fp16",                1,  4,   32,   32,  96, False, None,  torch.float16, False, 1e-2),
            # Large causal in fp16
            ("s512_ca_fp16",            1,  8,  512,  512,  64, True,  None,  torch.float16, True, 0.1),
            ("s2048_ca_fp16",           1,  8, 2048, 2048,  64, True,  None,  torch.float16, True, 0.1),
            # Large causal in bf16
            ("s512_ca_bf16",            1,  8,  512,  512,  64, True,  None,  torch.bfloat16, False, 1e-2),
            ("s2048_ca_bf16",           1,  8, 2048, 2048,  64, True,  None,  torch.bfloat16, False, 1e-2),
            # --- FP16, custom scale ---
            ("scale_0125_fp16",         2,  8,   64,   64,  64, False, 0.125, torch.float16, False, 1e-2),
            ("scale_05_ca_fp16",        2,  8,   64,   64,  64, True,  0.5,   torch.float16, False, 1e-2),
            # scale=2.0 in FP16 causes ~0.5% mismatch due to fp16 overflow; loosen atol
            ("scale_2_fp16",            2,  8,   64,   64,  64, False, 2.0,   torch.float16, False, 0.1),
            # --- FP32 ---
            ("b1_h8_s32_d64_nc_fp32",   1,  8,   32,   32,  64, False, None,  torch.float32, False, 1e-2),
            ("b1_h8_s32_d64_ca_fp32",   1,  8,   32,   32,  64, True,  None,  torch.float32, False, 1e-2),
            ("b2_h8_s128_d64_fp32",     2,  8,  128,  128,  64, False, None,  torch.float32, False, 1e-2),
            ("scale_05_ca_fp32",        2,  8,   64,   64,  64, True,  0.5,   torch.float32, False, 1e-2),
            # --- BF16 (Ampere+ only, guarded per-test) ---
            ("b1_h8_s32_d64_nc_bf16",   1,  8,   32,   32,  64, False, None,  torch.bfloat16, False, 1e-2),
            ("b2_h8_s128_d64_ca_bf16",  2,  8,  128,  128,  64, True,  None,  torch.bfloat16, False, 1e-2),
            # LLM-realistic configs
            ("llama32_1b_prefill_fp16", 1, 32, 2048, 2048,  64, True,  None,  torch.float16, True, 0.1),  # Llama-3.2-1B, large causal
            ("llama32_3b_prefill_fp16", 1, 24, 2048, 2048, 128, True,  None,  torch.float16, True, 1e-2),  # Llama-3.2-3B
            ("qwen25_05b_fp16",         1, 14,  128,  128,  64, True,  None,  torch.float16, False, 1e-2),  # Qwen2.5-0.5B
            ("mistral_7b_fp16",         1, 32,  512,  512, 128, True,  None,  torch.float16, True, 1e-2),  # Mistral-7B, flash dispatch
        ]
    )
    # fmt: on
    def test_no_mask(
        self,
        name,
        batch,
        num_heads,
        seq_q,
        seq_k,
        head_dim,
        is_causal,
        scale,
        dtype,
        use_decompose,
        test_atol,
    ):
        _skip_bf16_on_rtx(self, dtype)

        class SDPA(nn.Module):
            def forward(self, q, k, v):
                return torch.ops.aten.scaled_dot_product_attention.default(
                    q, k, v, None, 0.0, is_causal, scale=scale
                )

        q = torch.randn(batch, num_heads, seq_q, head_dim, dtype=dtype)
        k = torch.randn(batch, num_heads, seq_k, head_dim, dtype=dtype)
        v = torch.randn(batch, num_heads, seq_k, head_dim, dtype=dtype)
        self.run_test(
            SDPA(),
            [q, k, v],
            rtol=1e-2,
            atol=test_atol,
            use_explicit_typing=True,
            enable_passes=True,
            decompose_attention=use_decompose,
        )

    # fmt: off
    @parameterized.expand(
        [
            # (name, batch, num_heads, context_len, head_dim, dtype)
            ("b1_h8_ctx128_d64_fp16",   1,  8,  128,  64, torch.float16),
            ("b1_h8_ctx2048_d64_fp16",  1,  8, 2048,  64, torch.float16),
            ("b2_h8_ctx128_d64_fp16",   2,  8,  128,  64, torch.float16),
            ("b1_h8_ctx128_d64_fp32",   1,  8,  128,  64, torch.float32),
            ("b1_h8_ctx128_d64_bf16",   1,  8,  128,  64, torch.bfloat16),
            # LLM-realistic decode configs
            ("llama32_1b_dec_fp16",     1, 32, 2048, 128, torch.float16),
            ("qwen25_dec_fp16",         1, 14,  128,  64, torch.float16),
            ("mistral_dec_fp16",        1, 32,  512, 128, torch.float16),
            # Non-power-of-2 head dim
            ("d48_dec_fp16",            1,  8,  128,  48, torch.float16),
            ("d96_dec_fp16",            1,  8,  128,  96, torch.float16),
            # Long context
            ("b1_h32_ctx4096_d128_fp16", 1, 32, 4096, 128, torch.float16),
        ]
    )
    # fmt: on
    def test_decode(self, name, batch, num_heads, context_len, head_dim, dtype):
        """Single-token decode: Q has seq_len=1, K/V hold full context."""
        _skip_bf16_on_rtx(self, dtype)

        class DecodeAttention(nn.Module):
            def forward(self, q, k, v):
                return torch.ops.aten.scaled_dot_product_attention.default(
                    q, k, v, None, 0.0, False, scale=None
                )

        q = torch.randn(batch, num_heads, 1, head_dim, dtype=dtype)
        k = torch.randn(batch, num_heads, context_len, head_dim, dtype=dtype)
        v = torch.randn(batch, num_heads, context_len, head_dim, dtype=dtype)
        self.run_test(
            DecodeAttention(),
            [q, k, v],
            rtol=1e-2,
            atol=1e-2,
            use_explicit_typing=True,
            enable_passes=True,
            decompose_attention=False,
        )

    # fmt: off
    @parameterized.expand(
        [
            # (name, batch, heads, seq_q, seq_k, head_dim, mask_shape, dtype, use_decompose)
            # Full (batch, heads, seq_q, seq_k) masks
            ("full_b2_h8_s32_fp16",  2, 8,  32,  32, 64, (2, 8,  32,  32), torch.float16, False),
            ("full_b2_h8_s32_fp32",  2, 8,  32,  32, 64, (2, 8,  32,  32), torch.float32, False),
            ("full_b4_h8_s64_fp16",  4, 8,  64,  64, 64, (4, 8,  64,  64), torch.float16, False),
            # Broadcast: (1, 1, seq_q, seq_k)
            ("bcast_1111_fp16",      2, 8,  32,  32, 64, (1, 1,  32,  32), torch.float16, False),
            ("bcast_1111_fp32",      2, 8,  32,  32, 64, (1, 1,  32,  32), torch.float32, False),
            # Broadcast: (batch, 1, seq_q, seq_k)
            ("bcast_b1sk_fp16",      2, 8,  32,  32, 64, (2, 1,  32,  32), torch.float16, False),
            # 2D mask (seq_q, seq_k) — broadcastable
            ("mask_2d_fp16",         1, 8,  32,  32, 64, (32,  32),        torch.float16, False),
            ("mask_2d_fp32",         2, 8, 128, 128, 64, (128, 128),       torch.float32, False),
            # Decode step (seq_q=1): IAttentionLayer handles non-square Q/K natively
            ("decode_full_fp16",     2, 8,   1,  32, 64, (2, 8,   1,  32), torch.float16, False),
            ("decode_bcast_fp16",    2, 8,   1,  32, 64, (1, 1,   1,  32), torch.float16, False),
            # Cross-attention (seq_q != seq_k): non-square
            ("cross_attn_fp16",      1, 8,  16,  64, 64, (1, 8,  16,  64), torch.float16, False),
        ]
    )
    # fmt: on
    def test_bool_mask(
        self,
        name,
        batch,
        num_heads,
        seq_q,
        seq_k,
        head_dim,
        mask_shape,
        dtype,
        use_decompose,
    ):
        _skip_bf16_on_rtx(self, dtype)

        class SDPABoolMask(nn.Module):
            def forward(self, q, k, v, mask):
                return torch.ops.aten.scaled_dot_product_attention.default(
                    q, k, v, mask, 0.0, False, scale=None
                )

        q = torch.randn(batch, num_heads, seq_q, head_dim, dtype=dtype)
        k = torch.randn(batch, num_heads, seq_k, head_dim, dtype=dtype)
        v = torch.randn(batch, num_heads, seq_k, head_dim, dtype=dtype)
        mask = torch.randint(0, 2, mask_shape, dtype=torch.bool)
        self.run_test(
            SDPABoolMask(),
            [q, k, v, mask],
            rtol=1e-2,
            atol=1e-2,
            use_explicit_typing=True,
            enable_passes=True,
            decompose_attention=use_decompose,
        )

    # fmt: off
    @parameterized.expand(
        [
            # (name, batch, heads, seq_q, seq_k, head_dim, scale, dtype, use_decompose, test_atol)
            ("basic_nc_fp16",   2,  8,  32,  32, 64, None, torch.float16, False, 1e-2),
            ("basic_nc_fp32",   2,  8,  32,  32, 64, None, torch.float32, False, 1e-2),
            ("basic_nc_bf16",   2,  8,  32,  32, 64, None, torch.bfloat16, False, 1e-2),
            # scale causes ~0.2% FP16 mismatch at atol=0.01; loosen to 0.05
            ("scale1_fp16",     2,  8, 128, 128, 64, 1.0,  torch.float16, False, 5e-2),
            ("large_seq_fp16",  1,  8, 512, 512, 64, None, torch.float16, False, 1e-2),
            ("b4_h16_fp16",     4, 16,  64,  64, 64, None, torch.float16, False, 1e-2),
            # Decode step (seq_q=1)
            ("decode_fp16",     2,  8,   1,  32, 64, None, torch.float16, False, 1e-2),
            ("decode_fp32",     2,  8,   1,  64, 64, None, torch.float32, False, 1e-2),
            # Non-standard head dim
            ("d48_fp16",        1,  4,  32,  32, 48, None, torch.float16, False, 1e-2),
        ]
    )
    # fmt: on
    def test_float_mask(
        self,
        name,
        batch,
        num_heads,
        seq_q,
        seq_k,
        head_dim,
        scale,
        dtype,
        use_decompose,
        test_atol,
    ):
        _skip_bf16_on_rtx(self, dtype)

        class SDPAFloatMask(nn.Module):
            def forward(self, q, k, v, mask):
                return torch.ops.aten.scaled_dot_product_attention.default(
                    q, k, v, mask, 0.0, False, scale=scale
                )

        q = torch.randn(batch, num_heads, seq_q, head_dim, dtype=dtype)
        k = torch.randn(batch, num_heads, seq_k, head_dim, dtype=dtype)
        v = torch.randn(batch, num_heads, seq_k, head_dim, dtype=dtype)
        mask = torch.randn(batch, num_heads, seq_q, seq_k, dtype=dtype)
        self.run_test(
            SDPAFloatMask(),
            [q, k, v, mask],
            rtol=1e-2,
            atol=test_atol,
            use_explicit_typing=True,
            enable_passes=True,
            decompose_attention=use_decompose,
        )

    # fmt: off
    @parameterized.expand(
        [
            # (name, batch, q_heads, kv_heads, seq_len, head_dim, is_causal, dtype, use_decompose)
            ("gqa_32q_8kv_s128_fp16",    1, 32, 8,  128, 128, True,  torch.float16, False),
            ("gqa_32q_8kv_s2048_fp16",   1, 32, 8, 2048, 128, True,  torch.float16, True),  # large causal in fp16
            ("gqa_32q_8kv_s2048_bf16",   1, 32, 8, 2048, 128, True,  torch.bfloat16,False),  # large causal in bf16
            ("gqa_16q_4kv_s128_fp16",    2, 16, 4,  128,  64, True,  torch.float16, False),
            ("gqa_8q_2kv_nc_fp16",       2,  8, 2,   64,  64, False, torch.float16, False),
            ("gqa_8q_4kv_fp32",          2,  8, 4,   64,  64, False, torch.float32, False),
            ("gqa_24q_8kv_fp16",         1, 24, 8,  128, 128, True,  torch.float16, False),  # Llama-3.2-3B
            ("gqa_14q_2kv_fp16",         1, 14, 2,  128,  64, True,  torch.float16, False),  # Qwen2.5-0.5B
            # MQA (kv_heads = 1)
            ("mqa_8q_1kv_nc_fp16",       2,  8, 1,   64,  64, False, torch.float16, False),
            ("mqa_16q_1kv_ca_fp16",      1, 16, 1,  128,  64, True,  torch.float16, False),
            # GQA decode (seq_q=1)
            ("gqa_decode_32q_8kv_fp16",  2, 32, 8,    1, 128, False, torch.float16, False),
            ("mqa_decode_32q_1kv_fp16",  2, 32, 1,    1, 128, False, torch.float16, False),
        ]
    )
    # fmt: on
    def test_gqa(
        self,
        name,
        batch,
        num_q_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        is_causal,
        dtype,
        use_decompose,
    ):
        _skip_bf16_on_rtx(self, dtype)

        class GQA(nn.Module):
            def forward(self, q, k, v):
                return torch.ops.aten.scaled_dot_product_attention.default(
                    q, k, v, None, 0.0, is_causal, scale=None, enable_gqa=True
                )

        q = torch.randn(batch, num_q_heads, seq_len, head_dim, dtype=dtype)
        k = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=dtype)
        v = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=dtype)
        self.run_test(
            GQA(),
            [q, k, v],
            rtol=1e-2,
            atol=1e-2,
            use_explicit_typing=True,
            enable_passes=True,
            decompose_attention=use_decompose,
        )


# ---------------------------------------------------------------------------
# Flash attention kernel
# ---------------------------------------------------------------------------


@_FLASH_ATTN_SKIP
@_WINDOWS_SKIP
class TestFlashAttention(DispatchTestCase):
    """_scaled_dot_product_flash_attention kernel (Ampere+ required).

    Mirrors the TestEfficientAttention coverage structure.  Flash attention
    has no attn_bias parameter so test_with_bias / test_with_bias_causal /
    test_attn_bias_is_causal_opt have no equivalent here.

    test_no_mask
        Standard MHA, no mask.

    test_decode
        Decode-phase (seq_q=1, seq_k>1) via IAttentionLayer.

    test_gqa
        GQA/MQA (Hq != Hkv).  IAttentionLayer native.
        Large causal (seq >= 512) are tested.
    """

    # ------------------------------------------------------------------
    # 1. No mask
    # ------------------------------------------------------------------

    # fmt: off
    @parameterized.expand(
        [
            # (name, batch, heads, seq_len, head_dim, is_causal, scale, dtype, use_decompose, atol)
            ("causal_fp16",       2,  8,  128,  64, True,  None,  torch.float16,  False, 1e-2),
            ("nc_fp16",           2,  8,  128,  64, False, None,  torch.float16,  False, 1e-2),
            ("scale_025_ca_fp16", 2,  8,  128,  64, True,  0.25,  torch.float16,  False, 1e-2),
            # scale=0.5 causes ~4-element FP16 mismatch; loosen atol
            ("scale_05_nc_fp16",  2,  8,  128,  64, False, 0.5,   torch.float16,  False, 2e-2),
            ("b4_h16_s128_fp16",  4, 16,  128,  64, True,  None,  torch.float16,  False, 1e-2),
            ("b1_h8_d128_ca_fp16",1,  8,  128, 128, True,  None,  torch.float16,  False, 1e-2),
            ("b1_h32_s256_ca_fp16",1,32,  256,  64, True,  None,  torch.float16,  False, 1e-2),
            # Non-power-of-2 head dim
            ("d48_fp16",          1,  4,   64,  48, False, None,  torch.float16,  False, 1e-2),
            ("d96_fp16",          1,  4,   64,  96, False, None,  torch.float16,  False, 1e-2),
            # BF16
            ("causal_bf16",       2,  8,  128,  64, True,  None,  torch.bfloat16, False, 1e-2),
            # Large causal in fp16
            ("s512_ca_fp16",      1,  8,  512,  64, True,  None,  torch.float16,  True, 0.1),
            ("s2048_ca_fp16",     1, 32, 2048,  64, True,  None,  torch.float16,  True, 0.1),
            # Large causal in bf16
            ("s512_ca_bf16",      1,  8,  512,  64, True,  None,  torch.bfloat16, False, 1e-2),
            ("s2048_ca_bf16",     1, 32, 2048,  64, True,  None,  torch.bfloat16, False, 1e-2),
        ]
    )
    # fmt: on
    def test_no_mask(
        self,
        name,
        batch,
        num_heads,
        seq_len,
        head_dim,
        is_causal,
        scale,
        dtype,
        use_decompose,
        atol,
    ):
        _skip_bf16_on_rtx(self, dtype)

        class FlashAttn(nn.Module):
            def forward(self, q, k, v):
                out = torch.ops.aten._scaled_dot_product_flash_attention.default(
                    q, k, v, 0.0, is_causal, False, scale=scale
                )
                return out[0]

        q = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype)
        k = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype)
        v = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype)
        self.run_test(
            FlashAttn(),
            [q, k, v],
            rtol=1e-2,
            atol=atol,
            use_explicit_typing=True,
            enable_passes=True,
            decompose_attention=use_decompose,
        )

    # ------------------------------------------------------------------
    # 2. Decode-phase (seq_q=1, seq_k>1) — IAttentionLayer native
    # ------------------------------------------------------------------

    # fmt: off
    @parameterized.expand(
        [
            # (name, batch, heads, context_len, head_dim, dtype, atol)
            ("b1_h8_ctx128_fp16",   1,  8,  128,  64, torch.float16, 1e-2),
            ("b1_h8_ctx512_fp16",   1,  8,  512,  64, torch.float16, 1e-2),
            ("b2_h8_ctx128_fp16",   2,  8,  128,  64, torch.float16, 1e-2),
            ("b1_h8_d128_ctx128_fp16", 1, 8, 128, 128, torch.float16, 1e-2),
            # LLM-realistic decode configs
            ("llama_1b_dec_fp16",   1, 32, 2048, 128, torch.float16, 1e-2),
            ("qwen_dec_fp16",       1, 14,  128,  64, torch.float16, 1e-2),
            ("mistral_dec_fp16",    1, 32,  512, 128, torch.float16, 1e-2),
        ]
    )
    # fmt: on
    def test_decode(self, name, batch, num_heads, context_len, head_dim, dtype, atol):
        """Single-token decode: Q has seq_len=1, K/V hold the full context."""
        _skip_bf16_on_rtx(self, dtype)

        class FlashAttnDecode(nn.Module):
            def forward(self, q, k, v):
                out = torch.ops.aten._scaled_dot_product_flash_attention.default(
                    q, k, v, 0.0, False, False, scale=None
                )
                return out[0]

        q = torch.randn(batch, num_heads, 1, head_dim, dtype=dtype)
        k = torch.randn(batch, num_heads, context_len, head_dim, dtype=dtype)
        v = torch.randn(batch, num_heads, context_len, head_dim, dtype=dtype)
        self.run_test(
            FlashAttnDecode(),
            [q, k, v],
            rtol=1e-2,
            atol=atol,
            use_explicit_typing=True,
            enable_passes=True,
            decompose_attention=False,
        )

    # ------------------------------------------------------------------
    # 3. GQA / MQA — IAttentionLayer accepts Hq != Hkv natively
    # ------------------------------------------------------------------

    # fmt: off
    @parameterized.expand(
        [
            # (name, batch, q_heads, kv_heads, seq_len, head_dim, is_causal, dtype, use_decompose, atol)
            ("gqa_32q_8kv_s128_fp16",   1, 32, 8,  128, 128, True,  torch.float16, False, 1e-2),
            ("gqa_16q_4kv_s128_fp16",   2, 16, 4,  128,  64, True,  torch.float16, False, 1e-2),
            ("gqa_8q_2kv_nc_fp16",      2,  8, 2,   64,  64, False, torch.float16, False, 1e-2),
            ("gqa_24q_8kv_fp16",        1, 24, 8,  128, 128, True,  torch.float16, False, 1e-2),
            # MQA (kv_heads = 1)
            ("mqa_8q_1kv_nc_fp16",      2,  8, 1,   64,  64, False, torch.float16, False, 1e-2),
            ("mqa_16q_1kv_ca_fp16",     1, 16, 1,  128,  64, True,  torch.float16, False, 1e-2),
            # GQA decode (seq_q=1)
            ("gqa_decode_32q_8kv_fp16", 2, 32, 8,    1, 128, False, torch.float16, False, 1e-2),
            ("mqa_decode_32q_1kv_fp16", 2, 32, 1,    1, 128, False, torch.float16, False, 1e-2),
        ]
    )
    # fmt: on
    def test_gqa(
        self,
        name,
        batch,
        num_q_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        is_causal,
        dtype,
        use_decompose,
        atol,
    ):
        """GQA/MQA via flash attention: Q has Hq heads, K/V have Hkv heads."""
        _skip_bf16_on_rtx(self, dtype)

        class FlashAttnGQA(nn.Module):
            def forward(self, q, k, v):
                out = torch.ops.aten._scaled_dot_product_flash_attention.default(
                    q, k, v, 0.0, is_causal, False, scale=None
                )
                return out[0]

        q = torch.randn(batch, num_q_heads, seq_len, head_dim, dtype=dtype)
        k = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=dtype)
        v = torch.randn(batch, num_kv_heads, seq_len, head_dim, dtype=dtype)
        self.run_test(
            FlashAttnGQA(),
            [q, k, v],
            rtol=1e-2,
            atol=atol,
            use_explicit_typing=True,
            enable_passes=True,
            decompose_attention=use_decompose,
        )


# ---------------------------------------------------------------------------
# Efficient attention kernel
# ---------------------------------------------------------------------------


class TestEfficientAttention(DispatchTestCase):
    """_scaled_dot_product_efficient_attention kernel — all attn_bias scenarios.

    Five test methods cover the distinct code paths through the converter:

    test_no_bias
        attn_bias=None; uses decompose_attention=True to exercise the
        matmul+softmax fallback path.

    test_with_bias
        attn_bias provided; uses the native IAttentionLayer.mask path
        (decompose_attention=False, attn_bias_is_causal=False).
        Includes cases with batch=1 or heads=1 to stress-test mask alignment.

    test_with_bias_causal
        Both is_causal=True and attn_bias set simultaneously.  The converter
        materialises a causal tril mask and combines it with the float bias
        via additive -inf before passing to IAttentionLayer.
        (decompose_attention=False, attn_bias_is_causal=False)

    test_attn_bias_is_causal_opt
        Exercises the force_causal_efficient_attention lowering pass
        (attn_bias_is_causal=True, default).  The pass strips attn_bias and
        sets is_causal=True; both TRT and the PyTorch reference see the same
        post-lowering graph so the comparison is valid.
        (decompose_attention=False, attn_bias_is_causal=True)

    Note: bool attn_bias is not accepted by _scaled_dot_product_efficient_attention
    (PyTorch requires bias dtype == query dtype), so the bool+causal combine path
    in the converter cannot be exercised through this op.

    Note: GQA/MQA is not testable via _scaled_dot_product_efficient_attention
    directly — PyTorch's eager kernel rejects Hq != Hkv at runtime, so no
    valid reference exists for output comparison.
    """

    # ------------------------------------------------------------------
    # 1. No bias — decompose fallback
    # ------------------------------------------------------------------

    # fmt: off
    @parameterized.expand(
        [
            # (name, batch, heads, seq, head_dim, is_causal, scale, dtype, atol)
            ("causal_fp16",    2,  8, 128,  64, True,  None, torch.float16, 1e-2),
            ("nc_fp16",        2,  8, 128,  64, False, None, torch.float16, 1e-2),
            ("causal_fp32",    1,  8,  64,  64, True,  None, torch.float32, 1e-2),
            # scale=0.5 causes ~3-element FP16 mismatch; loosen atol
            ("scale05_ca_fp16",2,  8, 128,  64, True,  0.5,  torch.float16, 2e-2),
            ("b4_h16_fp16",    4, 16, 128,  64, False, None, torch.float16, 1e-2),
            ("s512_ca_fp16",   1,  8, 512,  64, True,  None, torch.float16, 1e-2),
            ("d128_fp16",      1,  8,  64, 128, True,  None, torch.float16, 1e-2),
            ("d48_fp16",       1,  4,  32,  48, False, None, torch.float16, 1e-2),
        ]
    )
    # fmt: on
    def test_no_bias(
        self, name, batch, num_heads, seq, head_dim, is_causal, scale, dtype, atol
    ):
        class EfficientAttn(nn.Module):
            def forward(self, q, k, v):
                out = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    q, k, v, None, False, 0.0, is_causal, scale=scale
                )
                return out[0]

        q = torch.randn(batch, num_heads, seq, head_dim, dtype=dtype)
        k = torch.randn(batch, num_heads, seq, head_dim, dtype=dtype)
        v = torch.randn(batch, num_heads, seq, head_dim, dtype=dtype)
        self.run_test(
            EfficientAttn(),
            [q, k, v],
            rtol=1e-2,
            atol=atol,
            use_explicit_typing=True,
            enable_passes=True,
            decompose_attention=True,
        )

    # ------------------------------------------------------------------
    # 2. With bias — native IAttentionLayer.mask path
    #    Includes heads=1 cases to stress mask alignment.
    # ------------------------------------------------------------------

    # fmt: off
    @parameterized.expand(
        [
            # (name, batch, heads, seq, head_dim, scale, dtype, atol)
            # Standard shapes
            ("nc_b2_h8_fp16",  2,  8,  32,  64, None, torch.float16, 1e-2),
            ("nc_b2_h8_fp32",  2,  8,  32,  64, None, torch.float32, 1e-2),
            # scale with bias causes borderline FP16 mismatch; loosen slightly
            ("scale05_fp16",   2,  8,  32,  64, 0.5,  torch.float16, 2e-2),
            ("scale2_fp32",    1,  8,  32,  64, 2.0,  torch.float32, 2e-2),
            ("large_seq_fp16", 1,  8, 128,  64, None, torch.float16, 1e-2),
            ("b4_h16_fp16",    4, 16,  64,  64, None, torch.float16, 1e-2),
            # heads=1 — alignment stress test for IAttentionLayer.mask
            ("h1_b1_fp16",     1,  1,  32,  64, None, torch.float16, 1e-2),
            ("h1_b2_fp16",     2,  1,  32,  64, None, torch.float16, 1e-2),
            ("h1_d128_fp16",   1,  1,  32, 128, None, torch.float16, 1e-2),
            # batch=1
            ("b1_h8_fp16",     1,  8,  32,  64, None, torch.float16, 1e-2),
        ]
    )
    # fmt: on
    def test_with_bias(self, name, batch, num_heads, seq, head_dim, scale, dtype, atol):
        class EfficientAttnBias(nn.Module):
            def forward(self, q, k, v, bias):
                out = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    q, k, v, bias, False, 0.0, False, scale=scale
                )
                return out[0]

        q = torch.randn(batch, num_heads, seq, head_dim, dtype=dtype)
        k = torch.randn(batch, num_heads, seq, head_dim, dtype=dtype)
        v = torch.randn(batch, num_heads, seq, head_dim, dtype=dtype)
        bias = torch.randn(batch, num_heads, seq, seq, dtype=dtype)
        self.run_test(
            EfficientAttnBias(),
            [q, k, v, bias],
            rtol=1e-2,
            atol=atol,
            use_explicit_typing=True,
            enable_passes=True,
            decompose_attention=False,
            attn_bias_is_causal=False,
        )

    # ------------------------------------------------------------------
    # 3. With bias — decode-phase (seq_q=1, seq_k>1)
    # ------------------------------------------------------------------

    # fmt: off
    @parameterized.expand(
        [
            # (name, batch, heads, seq_k, head_dim, scale, dtype, atol)
            # seq_k >= 8 required: PyTorch's efficient-attention kernel enforces
            # attn_bias.stride(1) = seq_q * seq_k >= 8 for its eager reference run.
            ("decode_b4_h8_fp16", 4, 8, 8, 64, None, torch.float16, 1e-2),
        ]
    )
    # fmt: on
    def test_with_bias_decode(
        self, name, batch, num_heads, seq_k, head_dim, scale, dtype, atol
    ):
        """Decode-phase with 4D float bias: q=(B,H,1,D), bias=(B,H,1,Sk)."""
        seq_q = 1

        class EfficientAttnBiasDecode(nn.Module):
            def forward(self, q, k, v, bias):
                out = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    q, k, v, bias, False, 0.0, False, scale=scale
                )
                return out[0]

        q = torch.randn(batch, num_heads, seq_q, head_dim, dtype=dtype)
        k = torch.randn(batch, num_heads, seq_k, head_dim, dtype=dtype)
        v = torch.randn(batch, num_heads, seq_k, head_dim, dtype=dtype)
        bias = torch.randn(batch, num_heads, seq_q, seq_k, dtype=dtype)
        self.run_test(
            EfficientAttnBiasDecode(),
            [q, k, v, bias],
            rtol=1e-2,
            atol=atol,
            use_explicit_typing=True,
            enable_passes=True,
            decompose_attention=False,
            attn_bias_is_causal=False,
        )

    # ------------------------------------------------------------------
    # 4. With bias + is_causal=True — combined path in converter
    # ------------------------------------------------------------------

    # fmt: off
    @parameterized.expand(
        [
            # (name, batch, heads, seq, head_dim, scale, dtype, atol)
            ("ca_b2_h8_fp16",   2, 8,  32,  64, None, torch.float16, 1e-2),
            ("ca_b1_h8_fp32",   1, 8,  64,  64, None, torch.float32, 1e-2),
            ("ca_scale05_fp16", 2, 8,  32,  64, 0.5,  torch.float16, 1e-2),
            ("ca_large_fp16",   1, 8, 128,  64, None, torch.float16, 1e-2),
            ("ca_d128_fp16",    1, 8,  32, 128, None, torch.float16, 1e-2),
        ]
    )
    # fmt: on
    def test_with_bias_causal(
        self, name, batch, num_heads, seq, head_dim, scale, dtype, atol
    ):
        class EfficientAttnBiasCausal(nn.Module):
            def forward(self, q, k, v, bias):
                out = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    q, k, v, bias, False, 0.0, True, scale=scale
                )
                return out[0]

        q = torch.randn(batch, num_heads, seq, head_dim, dtype=dtype)
        k = torch.randn(batch, num_heads, seq, head_dim, dtype=dtype)
        v = torch.randn(batch, num_heads, seq, head_dim, dtype=dtype)
        bias = torch.randn(batch, num_heads, seq, seq, dtype=dtype)
        self.run_test(
            EfficientAttnBiasCausal(),
            [q, k, v, bias],
            rtol=1e-2,
            atol=atol,
            use_explicit_typing=True,
            enable_passes=True,
            decompose_attention=False,
            attn_bias_is_causal=False,
        )

    # ------------------------------------------------------------------
    # 5. attn_bias_is_causal=True — force_causal_efficient_attention pass
    # ------------------------------------------------------------------

    # fmt: off
    @parameterized.expand(
        [
            # (name, batch, heads, seq, head_dim, scale, dtype, atol)
            ("opt_b2_h8_fp16",  2,  8,  32,  64, None, torch.float16, 1e-2),
            ("opt_large_fp16",  1,  8, 128,  64, None, torch.float16, 1e-2),
            ("opt_b4_h16_fp16", 4, 16,  64,  64, None, torch.float16, 1e-2),
            ("opt_d128_fp16",   1,  8,  32, 128, None, torch.float16, 1e-2),
        ]
    )
    # fmt: on
    def test_attn_bias_is_causal_opt(
        self, name, batch, num_heads, seq, head_dim, scale, dtype, atol
    ):
        class EfficientAttnBiasOpt(nn.Module):
            def forward(self, q, k, v, bias):
                out = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    q, k, v, bias, False, 0.0, False, scale=scale
                )
                return out[0]

        q = torch.randn(batch, num_heads, seq, head_dim, dtype=dtype)
        k = torch.randn(batch, num_heads, seq, head_dim, dtype=dtype)
        v = torch.randn(batch, num_heads, seq, head_dim, dtype=dtype)
        bias = torch.randn(
            batch, num_heads, seq, seq, dtype=dtype
        )  # values ignored; pass replaces with is_causal=True
        self.run_test(
            EfficientAttnBiasOpt(),
            [q, k, v, bias],
            rtol=1e-2,
            atol=atol,
            use_explicit_typing=True,
            enable_passes=True,
            decompose_attention=False,
            attn_bias_is_causal=True,
        )


if __name__ == "__main__":
    run_tests()
