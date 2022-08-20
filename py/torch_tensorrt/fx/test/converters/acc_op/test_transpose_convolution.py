# Owner(s): ["oncall: gpu_enablement"]

import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


class TestTransposeConvolutionConverter(AccTestCase):
    @parameterized.expand(
        [
            ("default", 1),
            param("no_bias", 1, bias=False),
            ("tuple_parameters", 1, (1, 1), (1, 1)),
            param("non_zero_padding", 1, padding=1),
            param("dilation", 1, dilation=2),
            param("groups", 1, groups=3),
        ]
    )
    def test_conv_transpose2d(
        self,
        _,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_transpose = torch.nn.ConvTranspose2d(
                    in_channels=3,
                    out_channels=6,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    groups=groups,
                    bias=bias,
                    dilation=dilation,
                )

            def forward(self, x):
                return self.conv_transpose(x)

        inputs = [torch.randn(1, 3, 224, 224)]
        self.run_test(TestModule(), inputs, expected_ops={acc_ops.conv_transpose2d})

    def test_conv_transpose2d_with_dynamic_shape(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, 1)

            def forward(self, x):
                return self.conv_transpose(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, 3, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 3, 1, 1), (1, 3, 4, 4), (32, 3, 128, 128))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={acc_ops.conv_transpose2d}
        )

    @parameterized.expand(
        [
            ("default", 1),
            param("no_bias", 1, bias=False),
            ("tuple_parameters", 1, (1, 1, 1), (1, 1, 1)),
            param("non_zero_padding", 1, padding=1),
            param("dilation", 1, dilation=2),
            param("groups", 1, groups=3),
        ]
    )
    def test_conv_transpose3d(
        self,
        _,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_transpose = torch.nn.ConvTranspose3d(
                    in_channels=3,
                    out_channels=6,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    groups=groups,
                    bias=bias,
                    dilation=dilation,
                )

            def forward(self, x):
                return self.conv_transpose(x)

        inputs = [torch.randn(1, 3, 32, 32, 32)]
        self.run_test(TestModule(), inputs, expected_ops={acc_ops.conv_transpose3d})

    def test_conv_transpose3d_with_dynamic_shape(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_transpose = torch.nn.ConvTranspose3d(3, 6, 1)

            def forward(self, x):
                return self.conv_transpose(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, 3, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 3, 1, 1, 1), (1, 3, 4, 4, 4), (8, 3, 32, 32, 32))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={acc_ops.conv_transpose3d}
        )


if __name__ == "__main__":
    run_tests()
