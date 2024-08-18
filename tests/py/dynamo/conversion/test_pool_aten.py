import torch
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
            ((3,), (1,), (1,), 0, True),
            ((7,), (2,), (1,), 0, True),
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
        )

    @parameterized.expand(
        [
            (3, 1, 0),
            (3, 1, 1),
            ((2, 2), [], (1, 0)),
            ((4, 3), (1, 1), (1, 1)),
            ((4, 3), (1, 1), (1, 1), True),
            ((5, 4), (2, 1), (1, 0)),
            ((5, 4), (2, 1), (1, 0), True),
            ((7, 7), (1, 2), (0, 1)),
            ((7, 7), (1, 2), (0, 1), True),
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
        self.run_test(TestModule(), inputs, rtol=5e-03, atol=5e-03, use_dynamo_tracer=True)

    @parameterized.expand(
        [
            (3, 1, 0),
            (3, 1, 1),
            ((2, 2, 3), [], (1, 0, 1)),
            ((4, 3, 2), (1, 1, 1), (1, 1, 0)),
            ((5, 4, 3), (2, 1, 2), (1, 0, 1)),
            ((7, 7, 7), (1, 2, 1), (0, 1, 1)),
            ((7, 7, 7), (1, 2, 1), (0, 1, 1), True),
            ((5, 4, 3), (2, 1, 2), (1, 0, 1), True),
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

        self.run_test_with_dynamic_shape(pool1d(), input_specs, use_dynamo_tracer=True)

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
            (
                (1, 1, 1, 1),
                (2, 2, 2, 2),
                (3, 3, 3, 3),
                torch.float,
                (3, 3),
                (1, 1),
                (1, 1),
                True
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
            ((7,), (2,), (1,), 1, True),
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
            ((4, 3), (1, 1), (1, 1), 1, True),
            ((5, 4), (2, 1), (1, 0), 1, True),
            ((7, 7), (1, 2), (0, 1), 1, True),
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
            ((7, 7, 7), (1, 2, 1), (0, 1, 1), 1, True),
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


if __name__ == "__main__":
    run_tests()
