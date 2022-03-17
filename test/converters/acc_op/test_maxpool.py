# Owner(s): ["oncall: aiacc"]

import torch
import fx2trt_oss.tracer.acc_tracer.acc_ops as acc_ops
from torch.testing._internal.common_fx2trt import AccTestCase, InputTensorSpec
from parameterized import parameterized, param
from torch.testing._internal.common_utils import run_tests


class TestMaxPoolConverter(AccTestCase):
    @parameterized.expand(
        [
            ("default", 1),
            ("kernel_3", 3),
            ("stride", 1, 2),
            param("padding", 2, padding=1),
            param("padding_even", 5, padding=2),
            param("ceil_mode", 1, ceil_mode=True),
        ]
    )
    def test_max_pool1d(self,
        test_name,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        ceil_mode=False,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.max_pool = torch.nn.MaxPool1d(
                    kernel_size, stride, padding, ceil_mode=ceil_mode, dilation=dilation
                )

            def forward(self, x):
                return self.max_pool(x)

        inputs = [torch.randn(1, 3, 224)]
        self.run_test(TestModule(), inputs, expected_ops={acc_ops.max_pool1d}, test_explicit_batch_dim=False,)


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
        self.run_test(TestModule(), inputs, expected_ops={acc_ops.max_pool2d})

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
            TestModule(), input_specs, expected_ops={acc_ops.max_pool2d}
        )

    @parameterized.expand(
        [
            ("default", 1),
            param("stride", 2, stride=()),
        ]
    )
    def test_stride_none_max_pool1d(self,
        test_name,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        ceil_mode=False,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.nn.functional.max_pool1d(
                    x, kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, dilation=dilation
                )

        inputs = [torch.randn(1, 3, 224)]
        self.run_test(TestModule(), inputs, expected_ops={acc_ops.max_pool1d}, test_explicit_batch_dim=False,)


    @parameterized.expand(
        [
            ("default", 1),
            param("stride", 2, stride=()),
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
                return  torch.nn.functional.max_pool2d(
                    x, kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode
                )

        inputs = [torch.randn(1, 3, 224, 224)]
        self.run_test(TestModule(), inputs, expected_ops={acc_ops.max_pool2d})

if __name__ == '__main__':
    run_tests()
