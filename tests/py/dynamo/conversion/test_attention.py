import unittest

import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from ..testing_utilities import DECIMALS_OF_AGREEMENT
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
        self.run_test(SDPA(), inputs, rtol=1e-2, atol=1e-2, precision=torch.float16)

    @parameterized.expand(
        [
            (
                "4d-2d",
                (4, 2, 128, 64),
                (6, 3, 128, 64),
                (32, 8, 128, 64),
                (4, 64),
                (4, 64),
                (16, 64),
            ),
            (
                "4d-4d",
                (4, 2, 12, 16),
                (6, 3, 16, 32),
                (32, 8, 18, 64),
                (4, 2, 4, 16),
                (6, 3, 8, 32),
                (32, 8, 12, 64),
            ),
        ]
    )
    def test_sdpa_no_causal_dynamic_shape_with_scale(
        self,
        _,
        query_min_shape,
        query_opt_shape,
        query_max_shape,
        key_min_shape,
        key_opt_shape,
        key_max_shape,
    ):
        class SDPA(nn.Module):
            def forward(self, query, key, value):
                return torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    None,
                    0.0,
                    is_causal=False,
                    scale=-0.5,
                )

        inputs = [
            # query
            Input(
                dtype=torch.float32,
                min_shape=query_min_shape,
                opt_shape=query_opt_shape,
                max_shape=query_max_shape,
            ),
            # key
            Input(
                dtype=torch.float32,
                min_shape=key_min_shape,
                opt_shape=key_opt_shape,
                max_shape=key_max_shape,
            ),
            # value
            Input(
                dtype=torch.float32,
                min_shape=key_min_shape,
                opt_shape=key_opt_shape,
                max_shape=key_max_shape,
            ),
        ]

        self.run_test_with_dynamic_shape(SDPA(), inputs)

    @parameterized.expand(
        [
            (
                "4d-2d",
                (4, 2, 128, 64),
                (6, 3, 128, 64),
                (32, 8, 128, 64),
                (4, 64),
                (4, 64),
                (16, 64),
            ),
            (
                "4d-4d",
                (4, 2, 12, 16),
                (6, 3, 16, 16),
                (32, 8, 18, 16),
                (4, 2, 4, 16),
                (6, 3, 8, 16),
                (32, 8, 12, 16),
            ),
        ]
    )
    def test_sdpa_no_causal_dynamic_shape(
        self,
        _,
        query_min_shape,
        query_opt_shape,
        query_max_shape,
        key_min_shape,
        key_opt_shape,
        key_max_shape,
    ):
        class SDPA(nn.Module):
            def forward(self, query, key, value):
                return torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    None,
                    0.0,
                    is_casual=False,
                    scale=None,
                )

        inputs = [
            # query
            Input(
                dtype=torch.float32,
                min_shape=query_min_shape,
                opt_shape=query_opt_shape,
                max_shape=query_max_shape,
            ),
            # key
            Input(
                dtype=torch.float32,
                min_shape=key_min_shape,
                opt_shape=key_opt_shape,
                max_shape=key_max_shape,
            ),
            # value
            Input(
                dtype=torch.float32,
                min_shape=key_min_shape,
                opt_shape=key_opt_shape,
                max_shape=key_max_shape,
            ),
        ]

        self.run_test_with_dynamic_shape(SDPA(), inputs)

    @parameterized.expand(
        [
            (
                "4d-2d",
                (4, 2, 128, 64),
                (6, 3, 128, 64),
                (32, 8, 128, 64),
                (4, 64),
                (4, 64),
                (4, 64),
            ),
            (
                "4d-4d",
                (4, 2, 12, 16),
                (6, 3, 12, 16),
                (32, 8, 12, 16),
                (4, 2, 4, 16),
                (6, 3, 4, 16),
                (32, 8, 4, 16),
            ),
        ]
    )
    def test_sdpa_causal_dynamic_shape(
        self,
        _,
        query_min_shape,
        query_opt_shape,
        query_max_shape,
        key_min_shape,
        key_opt_shape,
        key_max_shape,
    ):
        class SDPA(nn.Module):
            def forward(self, query, key, value):
                return torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, None, 0.0, True, scale=None
                )

        inputs = [
            # query
            Input(
                dtype=torch.float32,
                min_shape=query_min_shape,
                opt_shape=query_opt_shape,
                max_shape=query_max_shape,
            ),
            # key
            Input(
                dtype=torch.float32,
                min_shape=key_min_shape,
                opt_shape=key_opt_shape,
                max_shape=key_max_shape,
            ),
            # value
            Input(
                dtype=torch.float32,
                min_shape=key_min_shape,
                opt_shape=key_opt_shape,
                max_shape=key_max_shape,
            ),
        ]

        self.run_test_with_dynamic_shape(SDPA(), inputs)

    @parameterized.expand([((32, 8, 128, 64), (32, 8, 128, 64))])
    def test_sdpa_causal(self, query_shape, key_shape):
        class SDPA(nn.Module):
            def forward(self, query, key, value):
                return torch.nn.functional.scaled_dot_product_attention(
                    query, key, value, None, 0.0, True, scale=None
                )

        inputs = []
        query = torch.randn(query_shape, dtype=torch.float16)
        key = torch.rand(key_shape, dtype=torch.float16)
        value = torch.rand(key_shape, dtype=torch.float16)
        inputs.extend([query, key, value])
        self.run_test(SDPA(), inputs, rtol=1e-2, atol=1e-2, precision=torch.float16)


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
        )


if __name__ == "__main__":
    run_tests()
