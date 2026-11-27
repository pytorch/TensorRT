# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from unittest import mock

import tensorrt as trt
import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import param, parameterized
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_CUDNN_ATTENTION,
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
)
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import aten_ops_converters, impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
    attention_capability_validator,
    scaled_dot_product_attention_validator,
    scaled_dot_product_efficient_attention_validator,
)

from .harness import DispatchTestCase, skip_if_trt_rtx_turing


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
        if dtype == torch.float32:
            skip_if_trt_rtx_turing(self, "FP32 scaled dot-product attention")

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
        if dtype == torch.float32:
            skip_if_trt_rtx_turing(self, "FP32 scaled dot-product attention")

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
                # is_causal must be False when an explicit attn_mask is provided:
                # newer torch rejects passing both to scaled_dot_product_attention
                # ("Explicit attn_mask should not be set when is_causal=True").
                False,
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
        if dtype == torch.float32:
            skip_if_trt_rtx_turing(self, "FP32 scaled dot-product attention")

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

    @parameterized.expand(["sdpa", "efficient"])
    def test_sdpa_default_scale_bf16(self, op):
        skip_if_trt_rtx_turing(self, "bfloat16")

        class SDPA(nn.Module):
            def forward(self, query, key, value):
                if op == "efficient":
                    return (
                        torch.ops.aten._scaled_dot_product_efficient_attention.default(
                            query, key, value, None, False
                        )[0]
                    )
                return torch.ops.aten.scaled_dot_product_attention.default(
                    query, key, value
                )

        # A peaked softmax magnifies any bfloat16 rounding of the scaled query,
        # and sqrt(80) is not a power of two, so dividing by it rounds.
        shape = (2, 8, 64, 80)
        query = 4 * torch.randn(shape, dtype=torch.bfloat16)
        key = 4 * torch.randn(shape, dtype=torch.bfloat16)
        value = torch.randn(shape, dtype=torch.bfloat16)
        self.run_test(
            SDPA(), [query, key, value], rtol=1e-2, atol=1e-2, enable_passes=True
        )

    @parameterized.expand(
        [(trt.float16, True), (trt.float32, True), (trt.bfloat16, False)]
    )
    def test_default_scale_comes_from_query_shape_except_in_bf16(
        self, dtype, from_shape
    ):
        for attention in (
            impl.attention.scaled_dot_product_attention,
            impl.attention.scaled_dot_product_efficient_attention,
        ):
            network = trt.Builder(trt.Logger(trt.Logger.ERROR)).create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
            )
            query, key, value = (
                network.add_input(n, dtype, (2, 8, 64, 80)) for n in "qkv"
            )
            attention(
                ConversionContext(network),
                torch.ops.aten.scaled_dot_product_attention.default,
                SourceIR.ATEN,
                "attention",
                query,
                key,
                value,
            )
            layers = {network.get_layer(i).type for i in range(network.num_layers)}
            with self.subTest(attention=attention.__name__):
                self.assertEqual(trt.LayerType.SHAPE in layers, from_shape)

    @parameterized.expand([("query", 0), ("value", 2)])
    def test_sdpa_constant_input(self, _, index):
        shape = (2, 8, 32, 16)

        class SDPA(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "constant", torch.randn(shape, dtype=torch.float16)
                )

            def forward(self, query, other):
                args = [query, other, other]
                args[index] = self.constant
                return torch.ops.aten.scaled_dot_product_attention.default(*args)

        inputs = [torch.randn(shape, dtype=torch.float16) for _ in range(2)]
        self.run_test(SDPA(), inputs, rtol=1e-2, atol=1e-2, enable_passes=True)

    def test_only_a_constant_key_is_declined(self):
        graph = torch.fx.Graph()
        query, value, start = (graph.placeholder(n) for n in ("q", "v", "start"))
        key = graph.get_attr("k")
        is_constant = {
            key: True,
            graph.call_function(torch.ops.aten.view.default, (key, [-1])): True,
            graph.call_function(torch.ops.aten.slice.Tensor, (key, 0, 0, 1)): True,
            # The slice offset comes from an input, so the key is not a constant.
            graph.call_function(torch.ops.aten.slice.Tensor, (key, 0, start)): False,
        }
        sdpa = torch.ops.aten.scaled_dot_product_attention.default
        for k, constant in is_constant.items():
            k.meta["val"] = torch.empty(1, dtype=torch.float16, device="meta")
            with self.subTest(key=k.name):
                node = graph.call_function(sdpa, (query, k, value))
                self.assertEqual(
                    attention_capability_validator(node, CompilationSettings()),
                    not constant,
                )

    @parameterized.expand(
        [
            ("grouped_query", "sdpa", 1, 32, 2, "shared"),
            ("grouped_query_mask_per_head", "sdpa", 1, 32, 2, "per_head"),
            ("grouped_query_mask_without_batch", "sdpa", 2, 32, 2, "rank_3"),
            ("multi_head_batch_2", "sdpa", 2, 4, 4, "per_head"),
            ("efficient", "efficient", 2, 4, 4, "per_head"),
            ("cudnn", "cudnn", 2, 4, 4, "per_head"),
        ]
    )
    def test_single_query_mask(self, _, op, batch, q_heads, kv_heads, mask_kind):
        if op == "cudnn" and not PLATFORM_SUPPORTS_CUDNN_ATTENTION:
            self.skipTest("Platform doesn't support cuDNN attention")

        # A decode step over a long cache, where the mask hides the slots not written yet.
        class Attention(nn.Module):
            def forward(self, query, key, value, mask):
                if op == "sdpa":
                    return torch.ops.aten.scaled_dot_product_attention.default(
                        query, key, value, mask, enable_gqa=q_heads != kv_heads
                    )
                attention = (
                    torch.ops.aten._scaled_dot_product_efficient_attention.default
                    if op == "efficient"
                    else torch.ops.aten._scaled_dot_product_cudnn_attention.default
                )
                return attention(query, key, value, mask, False)[0]

        key_len = 2048
        query = torch.randn(batch, q_heads, 1, 64, dtype=torch.float16)
        key, value = (
            torch.randn(batch, kv_heads, key_len, 64, dtype=torch.float16)
            for _ in range(2)
        )
        value[:, :, 1700:] = 4
        # A different number of visible keys per head shows a mixed-up head order.
        visible = torch.arange(q_heads).view(-1, 1, 1) * 50 + 100
        mask = torch.arange(key_len) < (100 if mask_kind == "shared" else visible)
        if mask_kind != "rank_3":
            mask = mask.expand(
                batch, 1 if mask_kind == "shared" else q_heads, 1, key_len
            )
        if op != "sdpa":
            mask = torch.zeros(mask.shape, dtype=torch.float16).masked_fill(
                ~mask, float("-inf")
            )
        inputs = [query, key, value, mask.contiguous()]
        # run_test compares TensorRT with the rewritten graph, so compare that graph with the op first.
        rewritten = self.generate_graph(
            Attention(), inputs, use_dynamo_tracer=True, enable_passes=True
        )
        cuda_inputs = [tensor.cuda() for tensor in inputs]
        torch.testing.assert_close(
            rewritten(*cuda_inputs), Attention()(*cuda_inputs), rtol=1e-2, atol=1e-2
        )
        self.run_test(Attention(), inputs, rtol=1e-2, atol=1e-2, enable_passes=True)

    @parameterized.expand(
        [
            param("bfloat16", dtype=torch.bfloat16, key_len=60000),
            param(
                "bfloat16_hardware_compatible",
                dtype=torch.bfloat16,
                key_len=8192,
                hardware_compatible=True,
            ),
            param("float16_hardware_compatible", hardware_compatible=True),
            param("head_size_not_a_multiple_of_8", head_size=100),
            param("float32_cast_from_float16", cast="sdpa"),
            param("float32_efficient_cast_from_float16", cast="efficient"),
            param("dynamic_key", key_len=2048, dynamic=True),
        ]
    )
    def test_single_query_mask_compiled(
        self,
        _,
        dtype=torch.float16,
        key_len=1024,
        head_size=64,
        cast=None,
        dynamic=False,
        hardware_compatible=False,
    ):
        if dtype == torch.bfloat16:
            skip_if_trt_rtx_turing(self, "bfloat16")

        # Compiled the way a user compiles, so a dynamic key goes through every lowering pass.
        class Attention(nn.Module):
            def forward(self, query, key, value, mask):
                if cast:
                    # TensorRT would fold these casts and run the attention layer in FP16, so the pass decomposes
                    # it in FP32.
                    query, key, value = (t.float() for t in (query, key, value))
                    if cast == "efficient":
                        efficient = (
                            torch.ops.aten._scaled_dot_product_efficient_attention
                        )
                        attention = efficient.default(
                            query, key, value, None, False, 0.0, True, scale=0.125
                        )[0]
                    else:
                        attention = torch.ops.aten.scaled_dot_product_attention.default(
                            query, key, value, mask, scale=0.125
                        )
                    return attention.half()
                return torch.ops.aten.scaled_dot_product_attention.default(
                    query, key, value, mask, scale=0.3, enable_gqa=True
                )

        kv_heads = 8 if cast else 2
        query = torch.randn(1, 8, 1, head_size, dtype=dtype, device="cuda")
        key, value = (
            torch.randn(1, kv_heads, key_len, head_size, dtype=dtype, device="cuda")
            for _ in range(2)
        )
        value[:, :, 100:] = 4
        mask = (torch.arange(key_len, device="cuda") < 100).expand(1, 1, 1, key_len)
        inputs = (query, key, value, mask.contiguous())
        dynamic_shapes = None
        if dynamic:
            length = torch.export.Dim("length", min=2, max=4096)
            dynamic_shapes = (None, {2: length}, {2: length}, {3: length})
        compiled = torch_tensorrt.dynamo.compile(
            torch.export.export(Attention(), inputs, dynamic_shapes=dynamic_shapes),
            arg_inputs=inputs,
            min_block_size=1,
            hardware_compatible=hardware_compatible,
        )
        torch.testing.assert_close(
            compiled(*inputs), Attention()(*inputs), rtol=1e-2, atol=1e-2
        )

    @parameterized.expand(
        [
            param("float32", dtype=torch.float32),
            param("float32_head_size_12", dtype=torch.float32, head_size=12),
            param(
                "bfloat16_five_queries", dtype=torch.bfloat16, query_len=5, head_size=6
            ),
            param("float16", dtype=torch.float16),
            param("float16_head_size_32", dtype=torch.float16, head_size=32),
            # In FP16 the product of this query and key overflows, though the scaled score does not.
            param(
                "float16_head_size_32_large_scores",
                dtype=torch.float16,
                head_size=32,
                fill=50.0,
            ),
            param(
                "float16_head_size_32_hidden_key",
                dtype=torch.float16,
                head_size=32,
                hidden_key=True,
            ),
            param(
                "efficient_bias",
                op="efficient",
                dtype=torch.float16,
                query_len=5,
                bias=True,
            ),
            param(
                "cudnn_bias",
                op="cudnn",
                dtype=torch.float16,
                query_len=5,
                bias=True,
            ),
            param("flash", op="flash", dtype=torch.bfloat16),
            # TensorRT would fold the casts and run the attention layer in FP16.
            param(
                "float32_cast_from_float16",
                op="cast",
                dtype=torch.float16,
                head_size=32,
            ),
        ]
    )
    def test_single_key(
        self,
        _,
        dtype,
        op="sdpa",
        query_len=1,
        head_size=8,
        fill=None,
        hidden_key=False,
        bias=False,
    ):
        skip_if_trt_rtx_turing(self, "FP32 decomposition of single-key attention")
        if op == "flash" and not PLATFORM_SUPPORTS_FLASH_ATTENTION:
            self.skipTest("Platform doesn't support flash attention")
        if op == "cudnn" and not PLATFORM_SUPPORTS_CUDNN_ATTENTION:
            self.skipTest("Platform doesn't support cuDNN attention")

        class Attention(nn.Module):
            def forward(self, query, key, value, *mask):
                if op in ("efficient", "cudnn"):
                    attention = (
                        torch.ops.aten._scaled_dot_product_efficient_attention
                        if op == "efficient"
                        else torch.ops.aten._scaled_dot_product_cudnn_attention
                    )
                    # Sliced in the graph, as export slices a padded bias, so its strides stay aligned.
                    attn_bias = mask[0][..., :1] if mask else None
                    return attention.default(query, key, value, attn_bias, False)[0]
                if op == "flash":
                    flash = torch.ops.aten._scaled_dot_product_flash_attention
                    return flash.default(query, key, value)[0]
                if op == "cast":
                    query, key, value = (t.float() for t in (query, key, value))
                    return torch.ops.aten.scaled_dot_product_attention.default(
                        query, key, value, scale=0.125
                    ).half()
                return torch.ops.aten.scaled_dot_product_attention.default(
                    query, key, value, *mask
                )

        # With more heads TensorRT gets some of these shapes right, so one head shows the bug best.
        query = torch.randn(1, 1, query_len, head_size, dtype=dtype, device="cuda")
        key, value = (
            torch.randn(1, 1, 1, head_size, dtype=dtype, device="cuda")
            for _ in range(2)
        )
        if fill is not None:
            query.fill_(fill)
            key.fill_(fill)
        inputs = (query, key, value)
        if hidden_key:
            inputs += (
                torch.zeros(1, 1, query_len, 1, dtype=torch.bool, device="cuda"),
            )
        if bias:
            inputs += (torch.randn(1, 1, query_len, 8, dtype=dtype, device="cuda"),)
        compiled = torch_tensorrt.dynamo.compile(
            torch.export.export(Attention(), inputs),
            arg_inputs=inputs,
            min_block_size=1,
            # Flash attention is left to PyTorch; the others are decomposed into the engine.
            require_full_compilation=op != "flash",
        )
        torch.testing.assert_close(
            compiled(*inputs), Attention()(*inputs), rtol=1e-2, atol=1e-2
        )

    def test_short_keys_validator(self):
        skip_if_trt_rtx_turing(self, "FP32 decomposition of single-key attention")
        sdpa = torch.ops.aten.scaled_dot_product_attention.default
        efficient = torch.ops.aten._scaled_dot_product_efficient_attention.default
        rows = (
            (torch.bfloat16, 1, 1, 1, 16, True),
            (torch.bfloat16, 1, 1, 2, 8, True),
            (torch.float16, 1, 2, 1, 64, True),
            (torch.bfloat16, 1, 1, 1, 64, True),
            (torch.float16, 1, 1, 1, 16, True),
            (torch.float16, 1, 1, 1, 128, True),
            (torch.float16, 1, 2, 1, 12, True),
            (torch.float16, 1, 2, 1, 9, True),
            # With 2 heads, a batch of 2 gives the 4 heads in all that TensorRT needs for these.
            (torch.float16, 2, 1, 1, 64, True),
            (torch.bfloat16, 2, 1, 1, 8, True),
            (torch.float32, 1, 2, 1, 1, True),
            (torch.float32, 1, 2, 1, 16, True),
            (torch.float32, 1, 2, 1, 2, False),
            (torch.float32, 1, 2, 1, 15, False),
            (torch.float16, 1, 1, 1, 8, False),
            (torch.float16, 1, 1, 1, 64, False),
            # 4 heads in all keep only a one-row query with a head size of 8.
            (torch.float16, 2, 2, 1, 8, False),
            (torch.float32, 2, 1, 1, 8, True),
            (torch.float32, 2, 1, 1, 4, False),
        )
        for op, validator, extra in (
            (sdpa, scaled_dot_product_attention_validator, ()),
            (
                efficient,
                scaled_dot_product_efficient_attention_validator,
                (None, False),
            ),
        ):
            for dtype, batch, query_len, key_len, head_size, kept in rows:
                graph = torch.fx.Graph()
                query, key, value = (graph.placeholder(n) for n in "qkv")
                query.meta["val"] = torch.empty(
                    batch, 2, query_len, head_size, dtype=dtype, device="meta"
                )
                key.meta["val"] = value.meta["val"] = torch.empty(
                    batch, 2, key_len, head_size, dtype=dtype, device="meta"
                )
                node = graph.call_function(op, (query, key, value, *extra))
                with self.subTest(
                    op=op,
                    dtype=dtype,
                    batch=batch,
                    query_len=query_len,
                    key_len=key_len,
                    head_size=head_size,
                ):
                    self.assertEqual(validator(node, CompilationSettings()), kept)
                    # TensorRT-RTX keeps 16-bit attention for Turing, where it cannot run the FP32 decomposition.
                    with mock.patch.object(
                        aten_ops_converters, "trt_rtx_targets_turing", return_value=True
                    ):
                        self.assertEqual(
                            validator(node, CompilationSettings()),
                            dtype != torch.float32,
                        )

        class Attention(nn.Module):
            def forward(self, query, key, value):
                return sdpa(query, key, value)

        # Export takes a dynamic batch to be at least 2, but the engine's shape range can start at 1.
        batch = torch.export.Dim("batch", min=1, max=8)
        for heads, kept in ((2, False), (4, True)):
            query = torch.randn(2, heads, 1, 8, dtype=torch.float16)
            key = torch.randn(2, heads, 1, 8, dtype=torch.float16)
            ep = torch.export.export(
                Attention(), (query, key, key), dynamic_shapes=({0: batch},) * 3
            )
            (node,) = [n for n in ep.graph.nodes if n.target == sdpa]
            with self.subTest(heads=heads, batch="dynamic"):
                self.assertEqual(
                    scaled_dot_product_attention_validator(node, CompilationSettings()),
                    kept,
                )

    @parameterized.expand(
        [("sdpa", 1, 8), ("sdpa", 1, 2048), ("efficient", 6, 4), ("efficient", 1, 2048)]
    )
    def test_causal_with_unequal_lengths(self, op, query_len, key_len):
        # PyTorch aligns the causal mask to the top left, so a decode step attends to the first key only.
        class Causal(nn.Module):
            def forward(self, query, key, value):
                if op == "sdpa":
                    return torch.ops.aten.scaled_dot_product_attention.default(
                        query, key, value, None, 0.0, True
                    )
                return torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    query, key, value, None, False, 0.0, True
                )[0]

        query = torch.randn(2, 4, query_len, 32, dtype=torch.float16)
        key, value = (
            torch.randn(2, 4, key_len, 32, dtype=torch.float16) for _ in range(2)
        )
        inputs = [query, key, value]
        # run_test compares TensorRT with the rewritten graph, so compare that graph with the op first.
        rewritten = self.generate_graph(
            Causal(), inputs, use_dynamo_tracer=True, enable_passes=True
        )
        cuda_inputs = [tensor.cuda() for tensor in inputs]
        torch.testing.assert_close(
            rewritten(*cuda_inputs), Causal()(*cuda_inputs), rtol=1e-2, atol=1e-2
        )
        self.run_test(Causal(), inputs, rtol=1e-2, atol=1e-2, enable_passes=True)

    def test_causal_mask_kind(self):
        # TensorRT's LOWER_RIGHT is PyTorch's mask for equal lengths and for its flash kernel.
        sdpa = torch.ops.aten.scaled_dot_product_attention.default
        flash = torch.ops.aten._scaled_dot_product_flash_attention.default
        efficient = torch.ops.aten._scaled_dot_product_efficient_attention.default

        class Attention(nn.Module):
            def forward(self, x, memory):
                return (
                    sdpa(x, x, x, None, 0.0, True),
                    efficient(x, memory, memory, None, False, 0.0, True)[0],
                )

        # Equal example lengths, so only the symbols can tell the second attention's lengths apart.
        ep = torch.export.export(
            Attention(),
            (torch.randn(1, 2, 4, 8), torch.randn(1, 2, 4, 8)),
            dynamic_shapes=({2: torch.export.Dim("x")}, {2: torch.export.Dim("m")}),
        )
        nodes = {n.target: n for n in ep.graph.nodes}
        same, different = nodes[sdpa], nodes[efficient]
        network = trt.Builder(trt.Logger(trt.Logger.ERROR)).create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        four, six, dynamic = (
            network.add_input(name, trt.float16, (1, 2, length, 8))
            for name, length in (("four", 4), ("six", 6), ("dynamic", -1))
        )
        lower, upper = trt.CausalMaskKind.LOWER_RIGHT, trt.CausalMaskKind.UPPER_LEFT
        for node, target, query, key, kind in (
            (None, sdpa, four, four, lower),
            (None, sdpa, four, six, upper),
            (None, flash, four, six, lower),
            (same, sdpa, dynamic, dynamic, lower),
            (different, efficient, dynamic, dynamic, upper),
        ):
            with self.subTest(target=target, query=query.name, key=key.name, node=node):
                ctx = ConversionContext(network, current_node=node)
                self.assertEqual(
                    impl.attention._causal_mask_kind(ctx, target, query, key), kind
                )


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
                (4, 8, 8, 64),
                (4, 8, 8, 64),
                # attn_bias needs to match query @ key.transpose(-2, -1), and seq_k>=8 for kernel layout constraint to the FP16
                (4, 8, 1, 8),
                False,
                None,
                torch.float16,
                0.0,
            ),  # decoder-style single-token attention
            (
                (4, 8, 1, 64),
                (4, 8, 4, 64),
                (4, 8, 4, 64),
                (4, 8, 1, 4),  # need to match query @ key.transpose(-2, -1)
                False,
                2.0,
                torch.float32,
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
        if dtype == torch.float32:
            skip_if_trt_rtx_turing(self, "FP32 scaled dot-product attention")

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
                (4, 8, 8, 64),
                (4, 8, 8, 64),
                # attn_bias needs to match query @ key.transpose(-2, -1), and seq_k>=8 for kernel layout constraint to the FP16
                (4, 8, 1, 8),
                False,
                None,
                torch.float16,
                0.0,
            ),  # decoder-style single-token attention
            (
                (4, 8, 1, 64),
                (4, 8, 4, 64),
                (4, 8, 4, 64),
                (4, 8, 1, 4),  # need to match query @ key.transpose(-2, -1)
                False,
                2.0,
                torch.float32,
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
        if dtype == torch.float32:
            skip_if_trt_rtx_turing(self, "FP32 scaled dot-product attention")

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
            enable_passes=True,
        )

    def test_efficient_sdpa_constant_value(self):
        shape = (2, 8, 32, 16)

        class EfficientSDPA(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("value", torch.randn(shape, dtype=torch.float16))

            def forward(self, query, key):
                return torch.ops.aten._scaled_dot_product_efficient_attention.default(
                    query, key, self.value, None, False
                )[0]

        inputs = [torch.randn(shape, dtype=torch.float16) for _ in range(2)]
        self.run_test(EfficientSDPA(), inputs, rtol=1e-2, atol=1e-2, enable_passes=True)


if __name__ == "__main__":
    run_tests()
