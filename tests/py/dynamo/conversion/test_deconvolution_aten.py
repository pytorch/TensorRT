# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch_tensorrt
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase, skip_if_trt_rtx_turing


def _any_gt_1(v):
    return any(x > 1 for x in v) if isinstance(v, (tuple, list)) else v > 1


def _skip_if_rtx_strided_dilated_deconv(testcase, stride, dilation):
    """Mirror convolution_capability_validator: skip on TRT-RTX when a
    deconv combines stride > 1 with dilation > 1.

    The converter test harness drives TRTInterpreter directly and skips
    the partitioner, so a validator-rejected op raises
    UnsupportedOperatorException here instead of falling back to PyTorch
    as it would in torch_tensorrt.compile.
    """
    if (
        torch_tensorrt.ENABLED_FEATURES.tensorrt_rtx
        and _any_gt_1(stride)
        and _any_gt_1(dilation)
    ):
        testcase.skipTest("Strided + dilated deconv falls back to PyTorch on TRT-RTX")


class TestDeconvolutionConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("default", 1),
            param("no_bias", 1, bias=False),
            ("tuple_parameters", 1, (1), (1)),
            param("non_zero_padding", 1, padding=1),
            param("dilation", 1, dilation=2),
            param("groups", 1, groups=3),
            param("output_padding_1", 3, stride=2, padding=1, output_padding=1),
            param("output_padding_2", 3, stride=2, padding=2, output_padding=1),
            param("output_padding_3", 3, stride=2, padding=3, output_padding=1),
            param("output_padding_4", 3, stride=3, padding=2, output_padding=1),
            param("output_padding_5", 3, stride=3, padding=3, output_padding=1),
            param("output_padding_6", 3, stride=3, padding=3, output_padding=2),
            # tensorrt-rtx: does not support both strided and dilated deconv due to cuDNN limitation
            param(
                "combined_params",
                3,
                stride=1,
                padding=3,
                dilation=2,
                groups=3,
                output_padding=1,
            ),
            param("strided_dilated", 3, stride=2, padding=2, dilation=2),
        ]
    )
    def test_deconv1d(
        self,
        _,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        output_padding=0,
    ):
        _skip_if_rtx_strided_dilated_deconv(self, stride, dilation)

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.deconv = torch.nn.ConvTranspose1d(
                    in_channels=3,
                    out_channels=3,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    groups=groups,
                    bias=bias,
                    dilation=dilation,
                )

            def forward(self, x):
                return self.deconv(x)

        inputs = [torch.randn(1, 3, 32)]
        self.run_test(
            TestModule(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    def test_deconv1d_with_dynamic_shape(
        self,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.deconv = torch.nn.ConvTranspose1d(
                    in_channels=3,
                    out_channels=3,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    bias=bias,
                )

            def forward(self, x):
                return self.deconv(x)

        input_specs = [
            Input(
                shape=(-1, 3, 3),
                dtype=torch.float32,
                shape_ranges=[((1, 3, 3), (3, 3, 3), (5, 3, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(),
            input_specs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            ("default", 1),
            param("no_bias", 1, bias=False),
            ("tuple_parameters", 1, (1, 1), (1, 1)),
            param("non_zero_padding", 1, padding=1),
            param("dilation", 1, dilation=2),
            param("groups", 1, groups=3),
            param("output_padding_1", 3, stride=2, padding=1, output_padding=1),
            param("output_padding_2", 3, stride=2, padding=1, output_padding=1),
            param("output_padding_3", 3, stride=2, padding=2, output_padding=1),
            param("output_padding_4", 3, stride=2, padding=3, output_padding=1),
            param("output_padding_5", 3, stride=3, padding=2, output_padding=1),
            param("output_padding_6", 3, stride=3, padding=3, output_padding=1),
            param("output_padding_7", 3, stride=3, padding=3, output_padding=2),
            # tensorrt-rtx: does not support both strided and dilated deconv due to cuDNN limitation
            param(
                "combined_params",
                3,
                stride=1,
                padding=3,
                dilation=2,
                groups=3,
                output_padding=1,
            ),
            param("strided_dilated", 3, stride=2, padding=2, dilation=2),
        ]
    )
    def test_deconv2d(
        self,
        _,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        output_padding=0,
    ):
        _skip_if_rtx_strided_dilated_deconv(self, stride, dilation)

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.deconv = torch.nn.ConvTranspose2d(
                    in_channels=3,
                    out_channels=3,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    groups=groups,
                    bias=bias,
                    dilation=dilation,
                )

            def forward(self, x):
                return self.deconv(x)

        inputs = [torch.randn(1, 3, 32, 32)]
        self.run_test(
            TestModule(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    # Testing with (-1, -1, -1, -1) results into Error:
    # AssertionError: Channel dim can't be dynamic for deconvolution.

    def test_deconv2d_with_dynamic_shape(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.deconv = torch.nn.ConvTranspose2d(3, 3, 1)

            def forward(self, x):
                return self.deconv(x)

        input_specs = [
            Input(
                shape=(-1, 3, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 3, 1, 1), (2, 3, 4, 4), (32, 3, 128, 128))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(),
            input_specs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            ("default", 1),
            param("no_bias", 1, bias=False),
            ("tuple_parameters", 1, (1, 1, 1), (1, 1, 1)),
            param("non_zero_padding", 1, padding=1),
            param("dilation", 1, dilation=2),
            param("groups", 1, groups=3),
            param("output_padding_1", 3, stride=2, padding=1, output_padding=1),
            param("output_padding_2", 3, stride=2, padding=2, output_padding=1),
            param("output_padding_3", 3, stride=3, padding=3, output_padding=1),
            param("output_padding_4", 3, stride=3, padding=3, output_padding=2),
            param(
                "combined_params",
                3,
                stride=3,
                padding=3,
                dilation=2,
                groups=3,
                output_padding=2,
            ),
        ]
    )
    def test_deconv3d(
        self,
        _,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        output_padding=0,
    ):
        _skip_if_rtx_strided_dilated_deconv(self, stride, dilation)
        skip_if_trt_rtx_turing(self, "FP32 3D deconvolution")

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.deconv = torch.nn.ConvTranspose3d(
                    in_channels=3,
                    out_channels=3,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    groups=groups,
                    bias=bias,
                    dilation=dilation,
                )

            def forward(self, x):
                return self.deconv(x)

        inputs = [torch.randn(1, 3, 32, 32, 32)]
        self.run_test(
            TestModule(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    # Testing with (-1, -1, -1, -1, -1) results into Error:
    # AssertionError: Channel dim can't be dynamic for deconvolution.

    def test_deconv3d_with_dynamic_shape(self):
        skip_if_trt_rtx_turing(self, "FP32 3D deconvolution")

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.deconv = torch.nn.ConvTranspose3d(3, 3, 1)

            def forward(self, x):
                return self.deconv(x)

        input_specs = [
            Input(
                shape=(-1, 3, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 3, 1, 1, 1), (2, 3, 4, 4, 4), (8, 3, 32, 32, 32))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(),
            input_specs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            ("1d", torch.nn.ConvTranspose1d, (1, 16, 32)),
            ("2d", torch.nn.ConvTranspose2d, (1, 16, 16, 16)),
            ("3d", torch.nn.ConvTranspose3d, (1, 16, 8, 8, 8)),
        ]
    )
    def test_deconv_bf16(self, _, deconv_cls, shape):
        skip_if_trt_rtx_turing(self, "bfloat16")

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.deconv = deconv_cls(16, 16, kernel_size=2, stride=2)

            def forward(self, x):
                return self.deconv(x)

        inputs = [torch.randn(shape, dtype=torch.bfloat16)]
        # bfloat16 keeps about 2 digits.
        self.run_test(
            TestModule().to(torch.bfloat16),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
            rtol=1.6e-2,
            atol=1.6e-2,
        )

    @parameterized.expand(
        [
            ("1d", (1, 16, 32), (16, 8, 2), [2], [0], [1], [0]),
            ("2d", (1, 16, 16, 16), (16, 8, 2, 2), [2, 2], [0, 0], [1, 1], [0, 0]),
            (
                "3d",
                (1, 16, 8, 8, 8),
                (16, 8, 2, 2, 2),
                [2, 2, 2],
                [0, 0, 0],
                [1, 1, 1],
                [0, 0, 0],
            ),
        ]
    )
    def test_deconv_bf16_tensor_weight_and_bias(
        self, _, shape, weight_shape, stride, padding, dilation, output_padding
    ):
        skip_if_trt_rtx_turing(self, "bfloat16")

        class TestModule(torch.nn.Module):
            def forward(self, x, weight, bias):
                return torch.ops.aten.convolution.default(
                    x, weight, bias, stride, padding, dilation, True, output_padding, 1
                )

        inputs = [
            torch.randn(shape, dtype=torch.bfloat16),
            torch.randn(weight_shape, dtype=torch.bfloat16),
            torch.randn(weight_shape[1], dtype=torch.bfloat16),
        ]
        self.run_test(
            TestModule(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
            rtol=1.6e-2,
            atol=1.6e-2,
        )

    def test_deconv_bf16_feeds_another_deconv_with_values_past_float16(self):
        skip_if_trt_rtx_turing(self, "bfloat16")

        class TestModule(torch.nn.Module):
            def forward(self, x, first, second):
                y = torch.ops.aten.convolution.default(
                    x, first, None, [2, 2], [0, 0], [1, 1], True, [0, 0], 1
                )
                return torch.ops.aten.convolution.default(
                    y, second, None, [2, 2], [0, 0], [1, 1], True, [0, 0], 1
                )

        # The second layer fails to build unless the first casts its result back to
        # bfloat16. The values go above float16's largest value 65504, which bfloat16
        # holds. Non-negative values do not cancel, so the bfloat16 relative tolerance
        # holds for every element.
        inputs = [
            torch.rand(1, 16, 8, 8, dtype=torch.bfloat16) * 1e5,
            torch.rand(16, 16, 2, 2, dtype=torch.bfloat16),
            torch.rand(16, 16, 2, 2, dtype=torch.bfloat16),
        ]
        self.run_test(
            TestModule(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
            rtol=1.6e-2,
            atol=1.6e-2,
        )

    def test_deconv1d_tensor_weight(self):
        class TestModule(torch.nn.Module):
            def forward(self, x, weight):
                return torch.ops.aten.convolution.default(
                    x, weight, None, [2], [0], [1], True, [0], 1
                )

        inputs = [torch.randn(1, 16, 32), torch.randn(16, 8, 2)]
        self.run_test(
            TestModule(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )


if __name__ == "__main__":
    run_tests()
