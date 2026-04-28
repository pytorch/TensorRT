import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

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
            ),
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
            ),
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
            ),
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
            ),
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
            ),
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
            ),
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
            ),
            (
                (4, 8, 1, 64),
                (4, 8, 4, 64),
                (4, 8, 4, 64),
                (1, 1, 4),
                False,
                None,
                torch.float16,
                0.0,
                False,
            ),  # decoder-style single-token attention
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
            ),
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
            ),
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
            ),
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
            ),
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
            ),
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
            ),
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
            ),
            (
                (4, 8, 1, 64),
                (4, 8, 4, 64),
                (4, 8, 4, 64),
                (1, 1, 4),
                False,
                None,
                torch.float16,
                0.0,
                False,
            ),  # decoder-style single-token attention
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
        )

    @parameterized.expand(
        [
            (
                [(2, 8, 32, 16), (4, 8, 32, 16), (16, 8, 32, 16)],
                [(2, 8, 32, 16), (4, 8, 32, 16), (16, 8, 32, 16)],
                [(2, 8, 32, 16), (4, 8, 32, 16), (16, 8, 32, 16)],
                None,
                False,
                None,
                torch.float16,
                0.0,
                False,
            ),
            (
                [(2, 8, 32, 16), (4, 8, 32, 16), (16, 8, 32, 16)],
                [(2, 8, 32, 16), (4, 8, 32, 16), (16, 8, 32, 16)],
                [(2, 8, 32, 16), (4, 8, 32, 16), (16, 8, 32, 16)],
                None,
                True,
                None,
                torch.float32,
                0.0,
                False,
            ),
            (
                [(2, 8, 32, 16), (4, 8, 32, 16), (16, 8, 32, 16)],
                [(2, 8, 32, 16), (4, 8, 32, 16), (16, 8, 32, 16)],
                [(2, 8, 32, 16), (4, 8, 32, 16), (16, 8, 32, 16)],
                [(2, 8, 32, 32), (4, 8, 32, 32), (16, 8, 32, 32)],
                False,
                None,
                torch.float16,
                0.0,
                False,
            ),
            (
                [(2, 4, 128, 64), (4, 4, 128, 64), (8, 4, 128, 64)],
                [(2, 4, 128, 64), (4, 4, 128, 64), (8, 4, 128, 64)],
                [(2, 4, 128, 64), (4, 4, 128, 64), (8, 4, 128, 64)],
                [(2, 4, 128, 128), (4, 4, 128, 128), (8, 4, 128, 128)],
                True,
                2.0,
                torch.float32,
                0.0,
                False,
            ),
            (
                [(2, 8, 1, 64), (4, 8, 1, 64), (8, 8, 1, 64)],
                [(2, 8, 4, 64), (4, 8, 4, 64), (8, 8, 4, 64)],
                [(2, 8, 4, 64), (4, 8, 4, 64), (8, 8, 4, 64)],
                [(1, 1, 4), (1, 1, 4), (1, 1, 4)],
                False,
                None,
                torch.float16,
                0.0,
                False,
            ),  # decoder-style single-token attention
        ]
    )
    def test_dynamic_sdpa_fp_mask(
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

        input_specs = [
            Input(
                min_shape=q_shape[0],
                opt_shape=q_shape[1],
                max_shape=q_shape[2],
                dtype=dtype,
            ),
            Input(
                min_shape=k_shape[0],
                opt_shape=k_shape[1],
                max_shape=k_shape[2],
                dtype=dtype,
            ),
            Input(
                min_shape=v_shape[0],
                opt_shape=v_shape[1],
                max_shape=v_shape[2],
                dtype=dtype,
            ),
        ]
        if attn_mask_shape is not None:
            input_specs.append(
                Input(
                    min_shape=attn_mask_shape[0],
                    opt_shape=attn_mask_shape[1],
                    max_shape=attn_mask_shape[2],
                    dtype=dtype,
                ),
            )
        self.run_test_with_dynamic_shape(SDPA(), input_specs, output_dtypes=[dtype])


class TestScaledDotProductEfficientAttention(DispatchTestCase):
    @parameterized.expand(
        [
            (
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                None,
                True,
                1.0,
                torch.float16,
                0.0,
            ),
            (
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                None,
                False,
                None,
                torch.float16,
                0.0,
            ),
            (
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 32),
                True,
                0.5,
                torch.float16,
                0.0,
            ),
            (
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 32),
                True,
                2.0,
                torch.float32,
                0.0,
            ),
            (
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 32),
                False,
                2.0,
                torch.float32,
                0.0,
            ),
            (
                (4, 8, 1, 64),
                (4, 8, 4, 64),
                (4, 8, 4, 64),
                (1, 1, 4),
                False,
                None,
                torch.float16,
                0.0,
            ),  # decoder-style single-token attention
        ]
    )
    def test_efficient_sdpa(
        self,
        q_shape,
        k_shape,
        v_shape,
        attn_bias_shape,
        is_causal,
        scale,
        dtype,
        dropout_p=0.0,
    ):
        class EfficientSDPA(nn.Module):
            def forward(self, query, key, value, attn_bias=None):
                attn = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    query,
                    key,
                    value,
                    attn_bias,
                    False,
                    dropout_p,
                    is_causal,
                    scale=scale,
                )
                return attn[0]

        inputs = []
        query = torch.randn(q_shape, dtype=dtype)
        key = torch.rand(k_shape, dtype=dtype)
        value = torch.rand(v_shape, dtype=dtype)
        inputs.extend([query, key, value])
        if attn_bias_shape is not None:
            # create a lower triangular mask that is 0 for lower and -inf for upper
            attn_bias = torch.zeros(attn_bias_shape, dtype=dtype)
            upper = torch.triu(
                torch.ones(attn_bias_shape, dtype=torch.bool), diagonal=1
            )
            attn_bias = attn_bias.masked_fill(upper, float("-inf"))
            inputs.append(attn_bias)
        self.run_test(
            EfficientSDPA(),
            inputs,
            rtol=1e-2,
            atol=1e-2,
            precision=dtype,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            (
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                None,
                True,
                1.0,
                torch.float16,
                0.0,
            ),
            (
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                None,
                False,
                None,
                torch.float16,
                0.0,
            ),
            (
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 32),
                True,
                0.5,
                torch.float16,
                0.0,
            ),
            (
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 32),
                True,
                2.0,
                torch.float32,
                0.0,
            ),
            (
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 16),
                (4, 8, 32, 32),
                False,
                2.0,
                torch.float32,
                0.0,
            ),
            (
                (4, 8, 1, 64),
                (4, 8, 4, 64),
                (4, 8, 4, 64),
                (1, 1, 4),
                False,
                None,
                torch.float16,
                0.0,
            ),  # decoder-style single-token attention
        ]
    )
    def test_efficient_sdpa_random_attn_bias(
        self,
        q_shape,
        k_shape,
        v_shape,
        attn_bias_shape,
        is_causal,
        scale,
        dtype,
        dropout_p=0.0,
    ):
        class EfficientSDPA(nn.Module):
            def forward(self, query, key, value, attn_bias=None):
                attn = torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    query,
                    key,
                    value,
                    attn_bias,
                    False,
                    dropout_p,
                    is_causal,
                    scale=scale,
                )
                return attn[0]

        inputs = []
        query = torch.randn(q_shape, dtype=dtype)
        key = torch.rand(k_shape, dtype=dtype)
        value = torch.rand(v_shape, dtype=dtype)
        inputs.extend([query, key, value])
        if attn_bias_shape is not None:
            attn_bias = torch.randn(attn_bias_shape, dtype=dtype)
            inputs.append(attn_bias)
        self.run_test(
            EfficientSDPA(),
            inputs,
            rtol=1e-2,
            atol=1e-2,
            precision=dtype,
            enable_passes=True,
        )


if __name__ == "__main__":
    run_tests()
