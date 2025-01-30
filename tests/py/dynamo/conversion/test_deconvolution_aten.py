import torch
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests

from torch_tensorrt import Input

from .harness import DispatchTestCase


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


if __name__ == "__main__":
    run_tests()
