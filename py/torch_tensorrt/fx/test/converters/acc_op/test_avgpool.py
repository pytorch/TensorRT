import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


class TestAvgPoolConverter(AccTestCase):
    @parameterized.expand(
        [
            ("default", 1),
            ("kernal_size", 3),
            ("stride", 1, 2),
            ("tuple_parameters", 2, (1,), (1,)),
            param("padding", 2, padding=1),
            param("ceil_mode", 1, ceil_mode=True),
            param("include_pad", 2, padding=1, count_include_pad=False),
        ]
    )
    def test_avg_pool1d(
        self,
        test_name,
        kernel_size,
        stride=1,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.avg_pool = torch.nn.AvgPool1d(
                    kernel_size, stride, padding, ceil_mode, count_include_pad
                )

            def forward(self, x):
                return self.avg_pool(x)

        inputs = [torch.randn(1, 3, 224)]
        self.run_test(TestModule(), inputs, expected_ops={acc_ops.avg_pool1d})

    @parameterized.expand(
        [
            ("default", 1),
            ("kernal_size", 3),
            ("stride", 1, 2),
            ("tuple_parameters", 2, (1,), (1,)),
            param("padding", 2, padding=1),
            param("ceil_mode", 1, ceil_mode=True),
            param("include_pad", 2, padding=1, count_include_pad=False),
        ]
    )
    def test_avg_pool1d_with_dynamic_shape(
        self,
        test_name="default",
        kernel_size=1,
        stride=1,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.avg_pool = torch.nn.AvgPool1d(
                    kernel_size, stride, padding, ceil_mode, count_include_pad
                )

            def forward(self, x):
                return self.avg_pool(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, 3, 3),
                dtype=torch.float32,
                shape_ranges=[((1, 3, 3), (3, 3, 3), (3, 3, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={acc_ops.avg_pool1d}
        )

    def test_avg_pool2d_with_dynamic_shape_four_dimensions(
        self,
        test_name="default",
        kernel_size=1,
        stride=1,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.avg_pool = torch.nn.AvgPool2d(
                    kernel_size,
                    stride,
                    padding,
                    ceil_mode,
                    count_include_pad,
                    divisor_override,
                )

            def forward(self, x):
                return self.avg_pool(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (3, 3, 3, 3), (5, 5, 5, 5))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={acc_ops.avg_pool2d}
        )

    @parameterized.expand(
        [
            ("default", 1),
            ("kernal_size", 3),
            ("stride", 1, 2),
            ("tuple_parameters", 2, (1, 1), (1, 1)),
            param("padding", 2, padding=1),
            param("ceil_mode", 1, ceil_mode=True),
            param("include_pad", 2, padding=1, count_include_pad=False),
        ]
    )
    def test_avg_pool2d(
        self,
        test_name,
        kernel_size,
        stride=1,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.avg_pool = torch.nn.AvgPool2d(
                    kernel_size,
                    stride,
                    padding,
                    ceil_mode,
                    count_include_pad,
                    divisor_override,
                )

            def forward(self, x):
                return self.avg_pool(x)

        inputs = [torch.randn(1, 3, 224, 224)]
        self.run_test(TestModule(), inputs, expected_ops={acc_ops.avg_pool2d})

    @parameterized.expand(
        [
            ("kernal_size", 1),
            param("stride", 2, stride=()),
        ]
    )
    def test_stride_none_avg_pool1d(
        self,
        test_name,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.nn.functional.avg_pool1d(
                    x,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    ceil_mode=ceil_mode,
                    count_include_pad=count_include_pad,
                )

        inputs = [torch.randn(1, 3, 224)]
        self.run_test(TestModule(), inputs, expected_ops={acc_ops.avg_pool1d})

    @parameterized.expand(
        [
            ("kernal_size", 2),
            param("stride", 2, stride=()),
        ]
    )
    def test_stride_none_avg_pool2d(
        self,
        test_name,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.nn.functional.avg_pool2d(
                    x,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    ceil_mode=ceil_mode,
                    count_include_pad=count_include_pad,
                    divisor_override=divisor_override,
                )

        inputs = [torch.randn(1, 3, 224, 224)]
        self.run_test(TestModule(), inputs, expected_ops={acc_ops.avg_pool2d})

    def test_stride_none_avg_pool2d_with_dynamic_shape_four_dimensions(
        self,
        test_name="default",
        kernel_size=1,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.nn.functional.avg_pool2d(
                    x,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    ceil_mode=ceil_mode,
                    count_include_pad=count_include_pad,
                    divisor_override=divisor_override,
                )

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (3, 3, 3, 3), (5, 5, 5, 5))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={acc_ops.avg_pool2d}
        )


if __name__ == "__main__":
    run_tests()
