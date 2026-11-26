# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from unittest import mock

import torch
import torch.nn.functional as F
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo.conversion import aten_ops_converters, impl


class Attention(torch.nn.Module):
    def __init__(self, masked, gqa):
        super().__init__()
        self.masked, self.gqa = masked, gqa

    def forward(self, q, k, v, mask):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask if self.masked else None, enable_gqa=self.gqa
        )


def _kernel_spy():
    return mock.patch.object(
        impl.attention,
        "decode_attention_kernel",
        wraps=impl.attention.decode_attention_kernel,
    )


def _reference(q, k, v, mask):
    group = q.shape[1] // k.shape[1]
    w = (
        q.float()
        @ k.float().repeat_interleave(group, 1).transpose(-1, -2)
        * q.shape[-1] ** -0.5
    )
    if mask is not None:
        w = w + mask.float()
    return torch.softmax(w, -1) @ v.float().repeat_interleave(group, 1)


class TestDecodeAttentionKernel(TestCase):
    def setUp(self):
        reason = aten_ops_converters._grouped_mm_kernel_unavailable()
        if reason is not None:
            self.skipTest(reason)

    @parameterized.expand(
        [
            # name, batch, q heads, kv heads, head dim, cache length, mask heads (None: no mask), dtype
            ("gqa_masked_bf16", 1, 24, 8, 128, 2048, 24, torch.bfloat16),
            ("gqa_odd_length", 1, 24, 8, 128, 2049, 24, torch.bfloat16),
            ("gqa_no_mask", 1, 24, 8, 128, 1000, None, torch.bfloat16),
            ("mha_fp16_mask_over_heads", 2, 8, 8, 64, 777, 1, torch.float16),
        ]
    )
    def test_kernel_matches_pytorch(
        self, _, batch, heads, kv_heads, dim, length, mask_heads, dtype
    ):
        torch.manual_seed(0)
        q = torch.randn(batch, heads, 1, dim, dtype=dtype, device="cuda")
        k = torch.randn(batch, kv_heads, length, dim, dtype=dtype, device="cuda")
        v = torch.randn(batch, kv_heads, length, dim, dtype=dtype, device="cuda")
        mask = torch.randn(
            batch, mask_heads or 1, 1, length, dtype=dtype, device="cuda"
        )
        # The end of a static cache is masked out, as during decoding.
        mask[..., length * 3 // 4 :] = float("-inf")
        model = Attention(mask_heads is not None, heads != kv_heads).eval()
        inputs = [q, k, v, mask]
        with torch.no_grad(), _kernel_spy() as spy:
            compiled = torch_tensorrt.dynamo.compile(
                torch.export.export(model, tuple(inputs)),
                inputs=inputs,
                use_explicit_typing=True,
                min_block_size=1,
                decode_attention_kernel=True,
            )
            out = compiled(*inputs)
        self.assertEqual(spy.call_count, 1)
        expected = _reference(q, k, v, mask if mask_heads is not None else None)
        torch.testing.assert_close(out.float(), expected, rtol=0, atol=2e-2)

    def test_declines_more_than_one_query_row(self):
        q = torch.randn(1, 8, 2, 64, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(1, 8, 128, 64, dtype=torch.bfloat16, device="cuda")
        mask = torch.zeros(1, 1, 2, 128, dtype=torch.bfloat16, device="cuda")
        inputs = [q, k, k, mask]
        with torch.no_grad(), _kernel_spy() as spy:
            compiled = torch_tensorrt.dynamo.compile(
                torch.export.export(Attention(True, False).eval(), tuple(inputs)),
                inputs=inputs,
                use_explicit_typing=True,
                min_block_size=1,
                decode_attention_kernel=True,
            )
            out = compiled(*inputs)
        self.assertEqual(spy.call_count, 0)
        torch.testing.assert_close(
            out.float(), _reference(q, k, k, mask), rtol=0, atol=2e-2
        )


if __name__ == "__main__":
    run_tests()
