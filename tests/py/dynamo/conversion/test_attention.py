import unittest

import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestScaledDotProductAttention(DispatchTestCase):
    @parameterized.expand([((32, 8, 128, 64), (32, 8, 128, 64))])
    def test_sdpa_no_causal(self, query_shape, key_shape):
        class SDPA(nn.Module):
            def forward(self, query, key, value):
                return torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, None, 0.0, False, scale=None
                )

        inputs = []
        query = torch.randn(query_shape, dtype=torch.float16)
        key = torch.rand(key_shape, dtype=torch.float16)
        value = torch.rand(key_shape, dtype=torch.float16)
        inputs.extend([query, key, value])
        self.run_test(
            SDPA(),
            inputs,
            rtol=1e-2,
            atol=1e-2,
            precision=torch.float16,
            enable_passes=True,
            decompose_attention=True,
        )


@unittest.skipIf(
    torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8,
    "GPU compute capability is too low to run flash attention, need Ampere (8.0) or greater",
)
class TestFlashAttention(DispatchTestCase):
    @parameterized.expand([((32, 8, 128, 64), (32, 8, 128, 64))])
    def test_sdpa_causal(self, query_shape, key_shape):
        class SDPA(nn.Module):
            def forward(self, query, key, value):
                attn = torch.ops.aten._scaled_dot_product_flash_attention.default(
                    query,
                    key,
                    value,
                    0,
                    True,  # is_causal
                    False,
                    scale=0.25,
                )
                return attn[0]

        inputs = []
        query = torch.randn(query_shape, dtype=torch.float16)
        key = torch.rand(key_shape, dtype=torch.float16)
        value = torch.rand(key_shape, dtype=torch.float16)
        inputs.extend([query, key, value])
        self.run_test(
            SDPA(),
            inputs,
            rtol=1e-2,
            atol=1e-2,
            precision=torch.float16,
            enable_passes=True,
            decompose_attention=True,
        )


class TestEfficientAttention(DispatchTestCase):
    @parameterized.expand([((32, 8, 128, 64), (32, 8, 128, 64))])
    def test_sdpa_causal(self, query_shape, key_shape):
        class SDPA(nn.Module):
            def forward(self, query, key, value):
                attn = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    query,
                    key,
                    value,
                    None,
                    False,
                    0,
                    True,  # is_causal
                    scale=0.5,
                )
                return attn[0]

        inputs = []
        query = torch.randn(query_shape, dtype=torch.float16)
        key = torch.rand(key_shape, dtype=torch.float16)
        value = torch.rand(key_shape, dtype=torch.float16)
        inputs.extend([query, key, value])
        self.run_test(
            SDPA(),
            inputs,
            rtol=1e-2,
            atol=1e-2,
            precision=torch.float16,
            enable_passes=True,
            decompose_attention=True,
        )

    @parameterized.expand([((1, 2, 16, 32), (1, 2, 16, 32))])
    def test_sdpa_no_causal_with_bias(self, query_shape, key_shape):
        class SDPA(nn.Module):
            def forward(self, query, key, value, attn_bias):
                attn = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    query,
                    key,
                    value,
                    attn_bias,
                    False,
                    0,
                    False,  # is_causal
                    scale=0.5,
                )
                return attn[0]

        inputs = []
        query = torch.randn(query_shape, dtype=torch.float16)
        key = torch.rand(key_shape, dtype=torch.float16)
        value = torch.rand(key_shape, dtype=torch.float16)
        # Regression test: an attn_bias whose dtype doesn't match query's dtype
        # (e.g. an int32 padding mask, as HF BERT passes) used to be handed
        # straight to TensorRT's IAttention layer with no cast, which TRT's
        # AttentionInput layer rejects for anything but Float/Half/BFloat16/
        # Bool matching the attention dtype. int32 (not e.g. float32) is used
        # here since PyTorch's own eager kernel accepts an int32 bias but
        # rejects a mismatched float dtype outright.
        attn_bias = torch.zeros((query_shape[0], 1, 1, key_shape[2]), dtype=torch.int32)
        inputs.extend([query, key, value, attn_bias])
        self.run_test(
            SDPA(),
            inputs,
            rtol=1e-2,
            atol=1e-2,
            precision=torch.float16,
            enable_passes=True,
            decompose_attention=True,
        )


if __name__ == "__main__":
    run_tests()
