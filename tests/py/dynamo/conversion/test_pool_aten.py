# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest
from unittest import mock

import tensorrt as trt
import torch
import torch_tensorrt
from parameterized import param, parameterized
from torch.fx.passes.shape_prop import ShapeProp
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import UnsupportedOperatorException, impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
    avg_pool_param_validator,
    max_pool_param_validator,
)

from .harness import DispatchTestCase, skip_if_trt_rtx_turing

CEIL_MODE_POOL2D = [
    (
        "avg",
        lambda x: torch.ops.aten.avg_pool2d.default(x, (3, 3), (2, 2), (1, 1), True),
    ),
    (
        "max",
        lambda x: torch.ops.aten.max_pool2d.default(x, (3, 3), (2, 2), (1, 1), 1, True),
    ),
]


class TestPoolConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (3, 1, 0),
            ((3,), (1,), (1,)),
            ((2,), [], (0,)),
            ((4,), (1,), (1,)),
            ((5,), (2,), (0,)),
            ((7,), (2,), (1,)),
        ]
    )
    def test_avg_pool1d(
        self,
        kernel_size,
        stride=1,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.avg_pool1d.default(
                    x, kernel_size, stride, padding, ceil_mode, count_include_pad
                )

        inputs = [torch.randn(1, 3, 32)]
        self.run_test(
            TestModule(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            (3, 1, 0),
            (3, 1, 1),
            ((2, 2), [], (1, 0)),
            ((4, 3), (1, 1), (1, 1)),
            ((5, 4), (2, 1), (1, 0)),
            ((7, 7), (1, 2), (0, 1)),
        ]
    )
    def test_avg_pool2d(
        self,
        kernel_size,
        stride,
        padding,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.avg_pool2d.default(
                    x,
                    kernel_size,
                    stride,
                    padding,
                    ceil_mode,
                    count_include_pad,
                    divisor_override,
                )

        inputs = [torch.randn(1, 3, 32, 32)]
        self.run_test(TestModule(), inputs, use_dynamo_tracer=True)

    @parameterized.expand(
        [
            (3, 1, 0),
            (3, 1, 1),
            ((2, 2, 3), [], (1, 0, 1)),
            ((4, 3, 2), (1, 1, 1), (1, 1, 0)),
            ((5, 4, 3), (2, 1, 2), (1, 0, 1)),
            ((7, 7, 7), (1, 2, 1), (0, 1, 1)),
        ]
    )
    def test_avg_pool3d(
        self,
        kernel_size,
        stride,
        padding,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.avg_pool3d.default(
                    x,
                    kernel_size,
                    stride,
                    padding,
                    ceil_mode,
                    count_include_pad,
                    divisor_override,
                )

        inputs = [torch.randn(1, 3, 32, 32, 32)]
        self.run_test(TestModule(), inputs, use_dynamo_tracer=True)

    @parameterized.expand(
        [
            (
                (1, 1, 1),
                (2, 2, 2),
                (3, 3, 3),
                torch.float,
                (3,),
                (1,),
                (1,),
            ),
        ]
    )
    def test_dynamic_shape_pool1d(
        self,
        min_shape,
        opt_shape,
        max_shape,
        type,
        kernel_size,
        stride=1,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
    ):
        class pool1d(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.avg_pool1d.default(
                    x, kernel_size, stride, padding, ceil_mode, count_include_pad
                )

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(
            pool1d(), input_specs, use_dynamo_tracer=True, enable_passes=True
        )

    @parameterized.expand(
        [
            (
                (1, 1, 1, 1),
                (2, 2, 2, 2),
                (3, 3, 3, 3),
                torch.float,
                3,
                1,
                1,
            ),
            (
                (1, 1, 1, 1),
                (2, 2, 2, 2),
                (3, 3, 3, 3),
                torch.float,
                (3, 3),
                (1, 1),
                (1, 1),
            ),
        ]
    )
    def test_dynamic_shape_pool2d(
        self,
        min_shape,
        opt_shape,
        max_shape,
        type,
        kernel_size,
        stride=1,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
    ):
        class pool2d(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.avg_pool2d.default(
                    x, kernel_size, stride, padding, ceil_mode, count_include_pad
                )

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(pool2d(), input_specs, use_dynamo_tracer=True)

    @parameterized.expand(
        [
            (
                (1, 1, 1, 1, 1),
                (2, 2, 2, 2, 2),
                (3, 3, 3, 3, 3),
                torch.float,
                2,
                1,
                1,
            ),
            (
                (1, 1, 1, 1, 1),
                (2, 2, 2, 2, 2),
                (3, 3, 3, 3, 3),
                torch.float,
                (2, 2, 2),
                (1, 1, 1),
                (1, 1, 1),
            ),
        ]
    )
    def test_dynamic_shape_pool3d(
        self,
        min_shape,
        opt_shape,
        max_shape,
        type,
        kernel_size,
        stride=1,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
    ):
        class pool3d(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.avg_pool3d.default(
                    x, kernel_size, stride, padding, ceil_mode, count_include_pad
                )

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]

        self.run_test_with_dynamic_shape(pool3d(), input_specs, use_dynamo_tracer=True)

    @parameterized.expand(
        [
            (3, 1, 0),
            ((3,), (1,), (1,)),
            ((2,), [], (0,)),
            ((4,), (1,), (1,)),
            ((5,), (2,), (0,)),
            ((7,), (2,), (1,)),
        ]
    )
    def test_max_pool1d(
        self,
        kernel_size,
        stride,
        padding,
        dilation=1,
        ceil_mode=False,
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.max_pool1d.default(
                    x, kernel_size, stride, padding, dilation, ceil_mode
                )

        inputs = [torch.randn(1, 3, 32)]
        self.run_test(
            TestModule(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            (3, 1, 0),
            (3, 1, 1),
            ((2, 2), [], (1, 0)),
            ((4, 3), (1, 1), (1, 1)),
            ((5, 4), (2, 1), (1, 0)),
            ((7, 7), (1, 2), (0, 1)),
        ]
    )
    def test_max_pool2d(
        self,
        kernel_size,
        stride,
        padding,
        dilation=1,
        ceil_mode=False,
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.max_pool2d.default(
                    x, kernel_size, stride, padding, dilation, ceil_mode
                )

        inputs = [torch.randn(1, 3, 32, 32)]
        self.run_test(
            TestModule(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            (3, 1, 0),
            (3, 1, 1),
            ((2, 2, 3), [], (1, 0, 1)),
            ((4, 3, 2), (1, 1, 1), (1, 1, 0)),
            ((5, 4, 3), (2, 1, 2), (1, 0, 1)),
            ((7, 7, 7), (1, 2, 1), (0, 1, 1)),
        ]
    )
    def test_max_pool3d(
        self,
        kernel_size,
        stride,
        padding,
        dilation=1,
        ceil_mode=False,
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.max_pool3d.default(
                    x, kernel_size, stride, padding, dilation, ceil_mode
                )

        inputs = [torch.randn(1, 3, 32, 32, 32)]
        self.run_test(
            TestModule(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    # PyTorch uses a kernel size of one element for every pooled dimension.
    @parameterized.expand(
        [
            (
                "avg2d",
                lambda x: torch.ops.aten.avg_pool2d.default(x, (3,), (2,), (1,)),
                (1, 3, 8, 8),
            ),
            (
                "max3d",
                lambda x: torch.ops.aten.max_pool3d.default(x, (3,), (2,), (1,)),
                (1, 2, 8, 8, 8),
            ),
            (
                "avg2d_ceil_mode",
                lambda x: torch.ops.aten.avg_pool2d.default(x, (3,), (2,), (1,), True),
                (1, 3, 8, 8),
            ),
            (
                "max3d_ceil_mode",
                lambda x: torch.ops.aten.max_pool3d.default(
                    x, (3,), (2,), (1,), 1, True
                ),
                (1, 2, 8, 8, 8),
            ),
        ]
    )
    def test_pool_one_kernel_size(self, _, pool, shape):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return pool(x)

        inputs = [torch.randn(*shape)]
        self.run_test(TestModule(), inputs, use_dynamo_tracer=True)

    def test_max_pool2d_dynamic_channels(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.max_pool2d.default(x, (3, 3), (2, 2), (1, 1))

        input_specs = [
            Input(
                min_shape=(1, 2, 8, 8),
                opt_shape=(1, 3, 8, 8),
                max_shape=(1, 5, 8, 8),
                dtype=torch.float,
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, use_dynamo_tracer=True
        )

    # A symbolic trace keeps stride=(), which means a stride equal to the kernel size.
    @parameterized.expand(
        [
            (
                "avg",
                lambda x: torch.ops.aten.avg_pool2d.default(x, (3, 3), (), (1, 1)),
            ),
            (
                "max",
                lambda x: torch.ops.aten.max_pool2d.default(x, (3, 3), (), (1, 1)),
            ),
            (
                "avg_ceil_mode",
                lambda x: torch.ops.aten.avg_pool2d.default(
                    x, (3, 3), (), (1, 1), True
                ),
            ),
            (
                "max_ceil_mode",
                lambda x: torch.ops.aten.max_pool2d.default(
                    x, (3, 3), (), (1, 1), 1, True
                ),
            ),
        ]
    )
    def test_pool2d_empty_stride(self, _, pool):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return pool(x)

        inputs = [torch.randn(1, 3, 8, 8)]
        self.run_test(
            TestModule(), inputs, use_dynamo_tracer=False, propagate_shapes=True
        )

    # ceil_mode, on inputs where it adds a window (some reaching past the padding),
    # where it adds none, and where PyTorch drops the extra window because it would
    # start in the right padding.
    @parameterized.expand(
        [
            ((2,), (2,), (0,), True, 32),
            ((3,), (2,), (0,), True, 32),
            ((3,), (2,), (1,), True, 32),
            ((3,), (2,), (1,), False, 32),
            ((3,), (3,), (1,), True, 32),
            ((3,), (3,), (1,), False, 32),
            ((4,), (3,), (2,), True, 31),
        ]
    )
    def test_avg_pool1d_ceil_mode(
        self, kernel_size, stride, padding, count_include_pad, length
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.avg_pool1d.default(
                    x, kernel_size, stride, padding, True, count_include_pad
                )

        inputs = [torch.randn(1, 3, length)]
        self.run_test(
            TestModule(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            ((2, 2), (2, 2), (0, 0), True, (32, 32)),
            ((3, 3), (2, 2), (0, 0), True, (32, 32)),
            ((3, 3), (2, 2), (1, 1), True, (32, 32)),
            ((3, 3), (2, 2), (1, 1), False, (32, 32)),
            ((3, 3), (3, 3), (1, 1), True, (32, 32)),
            ((3, 3), (3, 3), (1, 1), False, (32, 32)),
            ((4, 3), (3, 2), (1, 0), True, (33, 30)),
            ((2, 2), (2, 2), (0, 0), True, (33, 31)),
            ((2, 2), (3, 3), (1, 1), True, (5, 5)),
            ((2, 2), (3, 3), (1, 1), False, (5, 5)),
            ((2, 2), (1, 1), (0, 0), True, (32, 31)),
            ((3, 3), (), (1, 1), True, (32, 31)),
        ]
    )
    def test_avg_pool2d_ceil_mode(
        self, kernel_size, stride, padding, count_include_pad, size
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.avg_pool2d.default(
                    x, kernel_size, stride, padding, True, count_include_pad, None
                )

        inputs = [torch.randn(1, 3, *size)]
        self.run_test(TestModule(), inputs, use_dynamo_tracer=True)

    @parameterized.expand(
        [
            ((2, 2, 2), (2, 2, 2), (0, 0, 0), True, (16, 16, 16)),
            ((3, 3, 3), (2, 2, 2), (1, 1, 1), True, (16, 16, 16)),
            ((3, 3, 3), (2, 2, 2), (1, 1, 1), False, (16, 16, 16)),
            ((3, 2, 3), (3, 2, 2), (1, 0, 1), True, (16, 15, 17)),
        ]
    )
    def test_avg_pool3d_ceil_mode(
        self, kernel_size, stride, padding, count_include_pad, size
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.avg_pool3d.default(
                    x, kernel_size, stride, padding, True, count_include_pad, None
                )

        inputs = [torch.randn(1, 2, *size)]
        self.run_test(TestModule(), inputs, use_dynamo_tracer=True)

    @parameterized.expand(
        [
            ((3,), (2,), (0,), 32),
            ((3,), (2,), (1,), 32),
            ((3,), (3,), (1,), 32),
        ]
    )
    def test_max_pool1d_ceil_mode(self, kernel_size, stride, padding, length):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.max_pool1d.default(
                    x, kernel_size, stride, padding, 1, True
                )

        inputs = [torch.randn(1, 3, length)]
        self.run_test(
            TestModule(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            ((2, 2), (2, 2), (0, 0), (32, 32)),
            ((3, 3), (2, 2), (0, 0), (32, 32)),
            ((3, 3), (2, 2), (0, 0), (31, 31)),
            ((3, 3), (2, 2), (1, 1), (32, 32)),
            ((3, 3), (3, 3), (1, 1), (32, 32)),
            ((4, 3), (3, 2), (1, 0), (33, 30)),
            ((2, 2), (3, 3), (1, 1), (5, 5)),
            ((2, 2), (1, 1), (0, 0), (32, 31)),
            ((3, 3), (), (1, 1), (32, 31)),
        ]
    )
    def test_max_pool2d_ceil_mode(self, kernel_size, stride, padding, size):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.max_pool2d.default(
                    x, kernel_size, stride, padding, 1, True
                )

        inputs = [torch.randn(1, 3, *size)]
        self.run_test(
            TestModule(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            ((2, 2, 2), (2, 2, 2), (0, 0, 0), (16, 16, 16)),
            ((3, 3, 3), (2, 2, 2), (1, 1, 1), (16, 16, 16)),
            ((3, 2, 3), (3, 2, 2), (1, 0, 1), (16, 15, 17)),
            ((47, 47, 47), (2, 2, 2), (0, 0, 0), (50, 50, 50)),
        ]
    )
    def test_max_pool3d_ceil_mode(self, kernel_size, stride, padding, size):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.max_pool3d.default(
                    x, kernel_size, stride, padding, 1, True
                )

        inputs = [torch.randn(1, 2, *size)]
        self.run_test(
            TestModule(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    @parameterized.expand(CEIL_MODE_POOL2D)
    def test_pool2d_ceil_mode_fp16(self, _, pool):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return pool(x)

        inputs = [torch.randn(1, 3, 32, 32).half()]
        self.run_test(TestModule(), inputs, use_dynamo_tracer=True)

    @parameterized.expand(CEIL_MODE_POOL2D)
    def test_pool2d_ceil_mode_bf16(self, _, pool):
        skip_if_trt_rtx_turing(self, "bfloat16")
        if trt.__version__.startswith("10.3."):
            self.skipTest("TensorRT 10.3 cannot build a strongly typed bfloat16 pool")

        class TestModule(torch.nn.Module):
            def forward(self, x):
                return pool(x)

        inputs = [torch.randn(1, 3, 32, 32).bfloat16()]
        self.run_test(TestModule(), inputs, use_dynamo_tracer=True)

    # The harness builds with truncate_double=True, so TensorRT returns float32.
    @parameterized.expand(CEIL_MODE_POOL2D)
    def test_pool2d_ceil_mode_fp64(self, _, pool):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return pool(x)

        inputs = [torch.randn(1, 3, 32, 32).double()]
        self.run_test(TestModule(), inputs, use_dynamo_tracer=True, check_dtype=False)

    # ShapeProp on a symbolic trace records the input shape only in tensor_meta.
    @parameterized.expand(CEIL_MODE_POOL2D)
    def test_pool2d_ceil_mode_shape_prop_metadata(self, _, pool):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return pool(x)

        inputs = [torch.randn(1, 3, 32, 32)]
        self.run_test(
            TestModule(), inputs, use_dynamo_tracer=False, propagate_shapes=True
        )

    # A symbolic trace keeps 1D pools, which TensorRT cannot build.
    @parameterized.expand(
        [
            (
                "avg_1d",
                lambda x: torch.ops.aten.avg_pool1d.default(x, (3,), (2,), (0,), True),
                (1, 3, 8),
            ),
            (
                "max_1d",
                lambda x: torch.ops.aten.max_pool1d.default(
                    x, (3,), (2,), (0,), (1,), True
                ),
                (1, 3, 8),
            ),
        ]
    )
    def test_pool_ceil_mode_symbolic_trace_not_converted(self, _, pool, shape):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return pool(x)

        with self.assertRaises(UnsupportedOperatorException):
            self.run_test(
                TestModule(),
                [torch.randn(shape)],
                use_dynamo_tracer=False,
                propagate_shapes=True,
            )

    # The symbolic trace also keeps a getattr node with no converter, so call the
    # validator directly to reach its check for sizes taken from graph values.
    def test_max_pool2d_ceil_mode_graph_kernel_size_not_converted(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.max_pool2d.default(
                    x, (x.shape[-2], x.shape[-1]), (2, 2), (0, 0), 1, True
                )

        traced = torch.fx.symbolic_trace(TestModule())
        ShapeProp(traced).propagate(torch.randn(1, 3, 7, 7))
        (node,) = [
            n
            for n in traced.graph.nodes
            if n.target == torch.ops.aten.max_pool2d.default
        ]
        self.assertFalse(max_pool_param_validator(node))

    def test_max_pool2d_ceil_mode_dilation_not_converted(self):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.max_pool2d.default(
                    x, (3, 3), (2, 2), (1, 1), (2, 2), True
                )

        inputs = [torch.randn(1, 3, 32, 32)]
        with self.assertRaises(UnsupportedOperatorException):
            self.run_test(TestModule(), inputs, use_dynamo_tracer=True)

    @parameterized.expand(
        [
            (*CEIL_MODE_POOL2D[0], (1, 3, 32, 30), (2, 3, 32, 30), (4, 3, 32, 30)),
            (*CEIL_MODE_POOL2D[1], (1, 3, 32, 30), (2, 3, 32, 30), (4, 3, 32, 30)),
            (
                "avg_dynamic_channels",
                CEIL_MODE_POOL2D[0][1],
                (1, 2, 8, 8),
                (1, 3, 8, 8),
                (1, 5, 8, 8),
            ),
            (
                "max_dynamic_channels",
                CEIL_MODE_POOL2D[1][1],
                (1, 2, 8, 8),
                (1, 3, 8, 8),
                (1, 5, 8, 8),
            ),
        ]
    )
    def test_pool2d_ceil_mode_dynamic_shape(
        self, _, pool, min_shape, opt_shape, max_shape
    ):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return pool(x)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=torch.float,
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, use_dynamo_tracer=True
        )

    @parameterized.expand(CEIL_MODE_POOL2D)
    def test_pool2d_ceil_mode_unbatched_not_converted(self, _, pool):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return pool(x)

        inputs = [torch.randn(3, 8, 8)]
        with self.assertRaises(UnsupportedOperatorException):
            self.run_test(TestModule(), inputs, use_dynamo_tracer=True)

    @parameterized.expand(
        [
            (
                "avg",
                lambda x: torch.ops.aten.avg_pool2d.default(
                    x, (317, 317), (1, 1), (0, 0), True
                ),
                (1, 1, 317, 317),
            ),
            (
                "max",
                lambda x: torch.ops.aten.max_pool2d.default(
                    x, (317, 317), (1, 1), (0, 0), 1, True
                ),
                (1, 1, 317, 317),
            ),
            (
                "max_3d",
                lambda x: torch.ops.aten.max_pool3d.default(
                    x, (317, 317, 317), (317, 317, 317), (0, 0, 0), 1, True
                ),
                (1, 1, 3, 3, 3),
            ),
        ]
    )
    def test_pool_ceil_mode_large_window_not_converted(self, _, pool, shape):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return pool(x)

        inputs = [torch.randn(shape)]
        with self.assertRaises(UnsupportedOperatorException):
            self.run_test(TestModule(), inputs, use_dynamo_tracer=True)

    @parameterized.expand(
        [
            (*CEIL_MODE_POOL2D[0], avg_pool_param_validator),
            (*CEIL_MODE_POOL2D[1], max_pool_param_validator),
        ]
    )
    def test_pool2d_ceil_mode_input_dtype(self, _, pool, validator):
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return pool(x)

        for dtype, truncate_double, trt_version, supported in (
            (torch.float64, False, trt.__version__, False),
            (torch.float64, True, trt.__version__, True),
            (torch.int64, True, trt.__version__, False),
            (torch.bfloat16, False, "10.3.0", False),
            (torch.bfloat16, False, "11.4.0", True),
        ):
            x = torch.randint(0, 9, (1, 3, 8, 8)).to(dtype)
            exported = torch.export.export(TestModule(), (x,))
            (node,) = [n for n in exported.graph.nodes if n.op == "call_function"]
            settings = CompilationSettings(truncate_double=truncate_double)
            with mock.patch("tensorrt.__version__", trt_version):
                self.assertEqual(validator(node, settings), supported)

    @parameterized.expand(
        [
            ((3, 3), (1, 1), (1, 1), False),
            ((2, 2), (2, 2), (1, 1), False),
            ((3, 3), (2, 2), (1, 1), True),
        ]
    )
    def test_avg_pool2d_ceil_mode_pads_only_when_it_can_add_a_window(
        self, kernel_size, stride, padding, can_add_window
    ):
        network = trt.Builder(trt.Logger(trt.Logger.ERROR)).create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        input = network.add_input("input", trt.float32, (1, 3, 8, 8))
        impl.pool.avg_poolNd(
            ConversionContext(network),
            torch.ops.aten.avg_pool2d.default,
            SourceIR.ATEN,
            "pool",
            input,
            kernel_size,
            stride,
            padding,
            ceil_mode=True,
        )
        self.assertEqual(network.num_layers > 1, can_add_window)


CEIL_MODE_POOL2D_END_TO_END = [
    ("max", lambda x: torch.nn.functional.max_pool2d(x, 3, 2, 1, ceil_mode=True)),
    ("avg", lambda x: torch.nn.functional.avg_pool2d(x, 3, 2, 1, ceil_mode=True)),
    (
        "avg_excluding_padding",
        lambda x: torch.nn.functional.avg_pool2d(
            x, 3, 2, 1, ceil_mode=True, count_include_pad=False
        ),
    ),
    ("max_stride_1", lambda x: torch.nn.functional.max_pool2d(x, 2, 1, ceil_mode=True)),
    # Drops the extra window at some sizes and keeps it at others.
    (
        "avg_stride_3",
        lambda x: torch.nn.functional.avg_pool2d(x, 3, 3, 1, ceil_mode=True),
    ),
]


@unittest.skipIf(not torch.cuda.is_available(), "Skip because CUDA is not available")
class TestCeilModePoolEndToEnd(TestCase):
    def _assert_one_engine_matches(self, model, trt_mod, inputs):
        names = [name for name, _ in trt_mod.named_children()]
        self.assertEqual(sum("_run_on_acc" in name for name in names), 1)
        self.assertFalse(any("_run_on_gpu" in name for name in names))
        for x in inputs:
            torch.testing.assert_close(trt_mod(x), model(x))

    @parameterized.expand(CEIL_MODE_POOL2D_END_TO_END)
    def test_dynamic_pooled_sizes(self, _, pool):
        class Model(torch.nn.Module):
            def forward(self, x):
                return pool(x)

        model = Model().eval().cuda()
        # torch.export rejects named ranges on the pooled dimensions of ceil_mode pools
        # with a stride above 1, so the sizes are automatic. For avg_stride_3, the
        # exported program accepts only sizes that keep or drop the extra window like the
        # example does, so the height always drops it and the width never does.
        exported = torch.export.export(
            model,
            (torch.randn(1, 2, 20, 21).cuda(),),
            dynamic_shapes=({2: torch.export.Dim.AUTO, 3: torch.export.Dim.AUTO},),
        )
        trt_mod = torch_tensorrt.dynamo.compile(
            exported,
            arg_inputs=[
                Input(
                    min_shape=(1, 2, 8, 8),
                    opt_shape=(1, 2, 20, 21),
                    max_shape=(1, 2, 33, 33),
                )
            ],
            min_block_size=1,
        )
        sizes = [(8, 9), (11, 13), (17, 16), (20, 21), (32, 33)]
        self._assert_one_engine_matches(
            model, trt_mod, [torch.randn(1, 2, h, w).cuda() for h, w in sizes]
        )

    @parameterized.expand(CEIL_MODE_POOL2D_END_TO_END)
    def test_static_crop_of_a_dynamic_input(self, _, pool):
        # The crop is 8x8, but its start depends on the input size, so TensorRT cannot
        # infer the crop size.
        class Model(torch.nn.Module):
            def forward(self, x):
                h, w = x.shape[-2:]
                return pool(x.narrow(2, h - 8, 8).narrow(3, w - 8, 8))

        model = Model().eval().cuda()
        trt_mod = torch_tensorrt.compile(
            model,
            ir="dynamo",
            inputs=[
                Input(
                    min_shape=(1, 2, 9, 9),
                    opt_shape=(1, 2, 12, 12),
                    max_shape=(1, 2, 32, 32),
                )
            ],
            min_block_size=1,
        )
        self._assert_one_engine_matches(
            model, trt_mod, [torch.randn(1, 2, s, s).cuda() for s in (9, 12, 20, 32)]
        )


if __name__ == "__main__":
    run_tests()
