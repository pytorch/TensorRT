import unittest

import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.dynamo.fx_ts_compat.tools.common_fx2trt import DispatchTestCase, InputTensorSpec


class TestMaxPoolConverter(DispatchTestCase):
    # TODO max_pool1d. It needs support of squeeze and unsqueeze

    @parameterized.expand(
        [
            ("default", 1),
            ("stride", 1, 2),
            ("tuple_parameters", 2, (1, 1), (1, 1)),
            param("padding", 2, padding=1),
            param("ceil_mode", 1, ceil_mode=True),
        ]
    )
    def test_max_pool2d(
        self,
        test_name,
        kernel_size,
        stride=1,
        padding=0,
        ceil_mode=False,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.max_pool = torch.nn.MaxPool2d(
                    kernel_size, stride, padding, ceil_mode=ceil_mode
                )

            def forward(self, x):
                return self.max_pool(x)

        inputs = [torch.randn(1, 3, 224, 224)]
        self.run_test(TestModule(), inputs, expected_ops={torch.ops.aten.max_pool2d})

    def test_max_pool2d_with_dynamic_shape(
        self,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.max_pool = torch.nn.MaxPool2d(1, 1)

            def forward(self, x):
                return self.max_pool(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (1, 2, 4, 4), (2, 4, 4, 4))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(),
            input_specs,
            expected_ops={torch.ops.aten.max_pool2d},
        )

    @parameterized.expand(
        [
            ("default", 1),
            # ("stride", 1, 2),
            # ("tuple_parameters", 2, (1, 1, 1), (1, 1, 1)),
            # param("padding", 2, padding=1),
            # param("ceil_mode", 1, ceil_mode=True),
        ]
    )
    @unittest.skip("PT2 tracer issue")
    def test_max_pool3d(
        self,
        test_name,
        kernel_size,
        stride=1,
        padding=0,
        ceil_mode=False,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.max_pool = torch.nn.MaxPool3d(
                    kernel_size, stride, padding, ceil_mode=ceil_mode
                )

            def forward(self, x):
                return self.max_pool(x)

        inputs = [torch.randn(1, 3, 32, 32, 32)]
        self.run_test(TestModule(), inputs, expected_ops={})

    @unittest.skip("PT2 tracer issue")
    def test_max_pool3d_with_dynamic_shape(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.max_pool = torch.nn.MaxPool3d(1, 1)

            def forward(self, x):
                return self.max_pool(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1, 1), (1, 2, 4, 4, 4), (2, 4, 4, 4, 4))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={torch.ops.aten.max_pool3d}
        )

    @parameterized.expand(
        [
            ("default", 1),
            # param("stride", 2, stride=()),  #PT2 tracer issue
        ]
    )
    def test_stride_none_max_pool2d(
        self,
        test_name,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.nn.functional.max_pool2d(
                    x, kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode
                )

        inputs = [torch.randn(1, 3, 224, 224)]
        self.run_test(TestModule(), inputs, expected_ops={torch.ops.aten.max_pool2d})

    @parameterized.expand(
        [
            ("default", 1),
            param("stride", 2, stride=()),
        ]
    )
    @unittest.skip("PT2 tracer issue")
    def test_stride_none_max_pool3d(
        self,
        test_name,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.nn.functional.max_pool3d(
                    x, kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode
                )

        inputs = [torch.randn(1, 3, 32, 32, 32)]
        self.run_test(TestModule(), inputs, expected_ops={torch.ops.aten.max_pool3d})

    @parameterized.expand(
        [
            ("default", 1),
            param("stride", 2, stride=()),
        ]
    )
    def test_stride_none_max_pool2d_with_dynamic_shape(
        self,
        test_name,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.nn.functional.max_pool2d(
                    x, kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode
                )

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (1, 2, 4, 4), (2, 4, 4, 4))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={torch.ops.aten.max_pool2d}
        )

    @parameterized.expand(
        [
            ("default", 1),
            param("stride", 2, stride=()),
        ]
    )
    @unittest.skip("PT2 tracer issue")
    def test_stride_none_max_pool3d_with_dynamic_shape(
        self,
        test_name,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.nn.functional.max_pool3d(
                    x, kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode
                )

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1, 1), (1, 2, 4, 4, 4), (2, 4, 4, 4, 4))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={torch.ops.aten.max_pool3d}
        )


if __name__ == "__main__":
    run_tests()
