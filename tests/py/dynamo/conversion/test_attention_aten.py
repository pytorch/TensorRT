import unittest

import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestScaledDotProductAttention(DispatchTestCase):
    @parameterized.expand(
        [
            (
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                None,
                False,
                None,
                torch.float16,
                0.0,
                False,
            ),  # flash attention
            (
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                None,
                False,
                0.1,
                torch.float16,
                0.0,
                False,
            ),  # flash attention
            (
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                None,
                True,
                None,
                torch.float32,
                0.0,
                False,
            ),  # flash attention
            (
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                None,
                True,
                0.5,
                torch.float32,
                0.0,
                False,
            ),  # flash attention
            (
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 32),
                False,
                None,
                torch.float16,
                0.0,
                False,
            ),  # efficient attention
            (
                (2, 8, 128, 64),
                (2, 8, 128, 64),
                (2, 8, 128, 64),
                (2, 8, 128, 128),
                False,
                1.0,
                torch.float16,
                0.0,
                False,
            ),  # efficient attention
            (
                (2, 8, 128, 64),
                (2, 8, 128, 64),
                (2, 8, 128, 64),
                (2, 8, 128, 128),
                False,
                2.0,
                torch.float32,
                0.0,
                False,
            ),  # efficient attention
        ]
    )
    def test_sdpa_bool_mask(
        self,
        q_shape,
        k_shape,
        v_shape,
        attn_mask_shape,
        is_causal,
        scale,
        dtype,
        dropout_p=0.0,
        enable_gqa=False,
    ):
        class SDPA(nn.Module):
            def forward(self, query, key, value, attn_mask=None):
                return torch.ops.aten.scaled_dot_product_attention.default(
                    query,
                    key,
                    value,
                    attn_mask,
                    dropout_p,
                    is_causal,
                    scale=scale,
                    enable_gqa=enable_gqa,
                )

        inputs = []
        query = torch.randn(q_shape, dtype=dtype)
        key = torch.rand(k_shape, dtype=dtype)
        value = torch.rand(v_shape, dtype=dtype)
        inputs.extend([query, key, value])
        if attn_mask_shape is not None:
            # bool mask
            attn_mask = torch.randint(0, 2, attn_mask_shape, dtype=torch.bool)
            inputs.append(attn_mask)
        self.run_test(
            SDPA(),
            inputs,
            rtol=1e-2,
            atol=1e-2,
            precision=dtype,
            enable_passes=True,
            use_explicit_typing=True,
        )

    @parameterized.expand(
        [
            (
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                None,
                False,
                None,
                torch.float16,
                0.0,
                False,
            ),  # flash attention
            (
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                None,
                False,
                0.1,
                torch.float16,
                0.0,
                False,
            ),  # flash attention
            (
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                None,
                True,
                None,
                torch.float32,
                0.0,
                False,
            ),  # flash attention
            (
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                None,
                True,
                0.5,
                torch.float32,
                0.0,
                False,
            ),  # flash attention
            (
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 32),
                False,
                None,
                torch.float16,
                0.0,
                False,
            ),  # efficient attention
            (
                (2, 8, 128, 64),
                (2, 8, 128, 64),
                (2, 8, 128, 64),
                (2, 8, 128, 128),
                False,
                1.0,
                torch.float16,
                0.0,
                False,
            ),  # efficient attention
            (
                (2, 8, 128, 64),
                (2, 8, 128, 64),
                (2, 8, 128, 64),
                (2, 8, 128, 128),
                False,
                2.0,
                torch.float32,
                0.0,
                False,
            ),  # efficient attention
        ]
    )
    def test_sdpa_fp_mask(
        self,
        q_shape,
        k_shape,
        v_shape,
        attn_mask_shape,
        is_causal,
        scale,
        dtype,
        dropout_p=0.0,
        enable_gqa=False,
    ):
        class SDPA(nn.Module):
            def forward(self, query, key, value, attn_mask=None):
                return torch.ops.aten.scaled_dot_product_attention.default(
                    query,
                    key,
                    value,
                    attn_mask,
                    dropout_p,
                    is_causal,
                    scale=scale,
                    enable_gqa=enable_gqa,
                )

        inputs = []
        query = torch.randn(q_shape, dtype=dtype)
        key = torch.rand(k_shape, dtype=dtype)
        value = torch.rand(v_shape, dtype=dtype)
        inputs.extend([query, key, value])
        if attn_mask_shape is not None:
            # fp mask
            attn_mask = torch.randn(attn_mask_shape, dtype=dtype)
            inputs.append(attn_mask)
        self.run_test(
            SDPA(),
            inputs,
            rtol=1e-2,
            atol=1e-2,
            precision=dtype,
            enable_passes=True,
            use_explicit_typing=True,
        )


if __name__ == "__main__":
    run_tests()
