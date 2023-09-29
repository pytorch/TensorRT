import torch
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestPoolConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (3, 1, 0),
            (3, 1, 1),
            (2, None, 0),
            (4, 1, 1),
            (5, 2, 0),
            (7, 2, 1),
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
            def __init__(self):
                super().__init__()
                self.pool = torch.nn.AvgPool1d(
                    kernel_size, stride, padding, ceil_mode, count_include_pad
                )

            def forward(self, x):
                return self.pool(x)

        inputs = [torch.randn(1, 3, 32)]
        self.run_test(
            TestModule(),
            inputs,
            expected_ops={torch.ops.aten.avg_pool2d.default},
        )

    @parameterized.expand(
        [
            (3, 1, 0),
            (3, 1, 1),
            ((2, 2), None, (1, 0)),
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
            def __init__(self):
                super().__init__()
                self.pool = torch.nn.AvgPool2d(
                    kernel_size,
                    stride,
                    padding,
                    ceil_mode,
                    count_include_pad,
                    divisor_override,
                )

            def forward(self, x):
                return self.pool(x)

        inputs = [torch.randn(1, 3, 32, 32)]
        self.run_test(
            TestModule(), inputs, expected_ops={torch.ops.aten.avg_pool2d.default}
        )

    @parameterized.expand(
        [
            (3, 1, 0),
            (3, 1, 1),
            ((2, 2, 3), None, (1, 0, 1)),
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
            def __init__(self):
                super().__init__()
                self.pool = torch.nn.AvgPool3d(
                    kernel_size,
                    stride,
                    padding,
                    ceil_mode,
                    count_include_pad,
                    divisor_override,
                )

            def forward(self, x):
                return self.pool(x)

        inputs = [torch.randn(1, 3, 32, 32, 32)]
        self.run_test(
            TestModule(), inputs, expected_ops={torch.ops.aten.avg_pool3d.default}
        )

    @parameterized.expand(
        [
            (3, 1, 0),
            (3, 1, 1),
            (2, None, 0),
            (4, 1, 1),
            (5, 2, 0),
            (7, 2, 1),
        ]
    )
    def test_max_pool1d(
        self,
        kernel_size,
        stride,
        padding,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = torch.nn.MaxPool1d(
                    kernel_size, stride, padding, dilation, return_indices, ceil_mode
                )

            def forward(self, x):
                return self.pool(x)

        inputs = [torch.randn(1, 3, 32)]
        self.run_test(
            TestModule(),
            inputs,
            expected_ops={torch.ops.aten.max_pool2d},
        )

    @parameterized.expand(
        [
            (3, 1, 0),
            (3, 1, 1),
            ((2, 2), None, (1, 0)),
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
        return_indices=False,
        ceil_mode=False,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = torch.nn.MaxPool2d(
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    return_indices,
                    ceil_mode,
                )

            def forward(self, x):
                return self.pool(x)

        inputs = [torch.randn(1, 3, 32, 32)]
        self.run_test(TestModule(), inputs, expected_ops={torch.ops.aten.max_pool2d})

    @parameterized.expand(
        [
            (3, 1, 0),
            (3, 1, 1),
            ((2, 2, 3), None, (1, 0, 1)),
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
        return_indices=False,
        ceil_mode=False,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = torch.nn.MaxPool3d(
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    return_indices,
                    ceil_mode,
                )

            def forward(self, x):
                return self.pool(x)

        inputs = [torch.randn(1, 3, 32, 32, 32)]
        self.run_test(TestModule(), inputs, expected_ops={torch.ops.aten.max_pool3d})


if __name__ == "__main__":
    run_tests()
