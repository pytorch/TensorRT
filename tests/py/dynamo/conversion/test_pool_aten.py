# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch_tensorrt
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


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

    @parameterized.expand(
        [
            ("max_int32", torch.ops.aten.max_pool2d.default, lambda x: x[..., 0]),
            (
                "max_int32_round",
                torch.ops.aten.max_pool2d.default,
                lambda x: torch.round(x[..., 0]),
            ),
            ("max_int64_prod", torch.ops.aten.max_pool2d.default, lambda x: x.prod(-1)),
            ("avg_int64_prod", torch.ops.aten.avg_pool2d.default, lambda x: x.prod(-1)),
            # An average of -15.25 truncates to -15, as in PyTorch.
            (
                "avg_int64_negative",
                torch.ops.aten.avg_pool2d.default,
                lambda x: x.sum(-1) // 3,
            ),
        ]
    )
    def test_pool2d_integer(self, _, pool, make):
        # CUDA has no integer pooling, so the graph is exported and checked on the CPU.
        class TestModule(torch.nn.Module):
            def forward(self, x):
                return pool(make(x), [2, 2])

        model = TestModule().eval()
        x = torch.arange(-40, 40, 5, dtype=torch.int32).reshape(1, 1, 4, 2, 2)
        x = torch.cat([x, x.flip(-1)], -2)
        compiled = torch_tensorrt.dynamo.compile(
            torch.export.export(model, (x,)),
            arg_inputs=[x.cuda()],
            min_block_size=1,
            require_full_compilation=True,
        )
        torch.testing.assert_close(compiled(x.cuda()).cpu(), model(x), rtol=0, atol=0)


if __name__ == "__main__":
    run_tests()
