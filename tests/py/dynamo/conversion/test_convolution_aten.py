import torch
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestConvolutionConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("default", 1),
            param("no_bias", 1, bias=False),
            ("tuple_parameters", 1, (1), (1)),
            param("non_zero_padding", 1, padding=1),
            param("dilation", 1, dilation=2),
            param("groups", 1, groups=3),
        ]
    )
    def test_conv1d(
        self,
        _,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(
                    3, 6, kernel_size, stride, padding, dilation, groups, bias
                )

            def forward(self, x):
                return self.conv(x)

        inputs = [torch.randn(1, 3, 32)]
        self.run_test(
            TestModule(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    @parameterized.expand(
        [
            ("default", 1),
            param("no_bias", 1, bias=False),
            ("tuple_parameters", 1, (1), (1)),
            param("non_zero_padding", 1, padding=1),
            param("dilation", 1, dilation=2),
        ]
    )
    def test_conv1d_TRTTensor_weight(
        self,
        _,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, w):
                return torch.ops.aten.convolution.default(
                    x,
                    w,
                    None,
                    (stride,) if isinstance(stride, int) else stride,
                    (padding,) if isinstance(padding, int) else padding,
                    (dilation,) if isinstance(dilation, int) else dilation,
                    False,
                    (0,),
                    groups,
                )

        inputs = [
            torch.randn(1, 3, 32),
            torch.randn(
                6, 3, 1
            ),  # Conv1d weight shape: (out_channels, in_channels, kernel_size)
        ]
        self.run_test(
            TestModule(),
            inputs,
            use_dynamo_tracer=True,
        )

    def test_conv1d_with_dynamic_shape(
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
                self.conv = torch.nn.Conv1d(
                    3, 6, kernel_size, stride, padding, dilation, groups, bias
                )

            def forward(self, x):
                return self.conv(x)

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
            param("list_stride", 2, stride=[2]),
            param("non_zero_padding", 1, padding=1),
            param("list_zero_padding", 1, padding=[0]),
            param("list_non_padding", 1, padding=[1]),
            param("dilation", 2, dilation=3),
            param("tuple_dilation", 2, dilation=(3, 3)),
            param("list_dilation", 2, dilation=[3]),
            param("groups", 1, groups=3),
        ]
    )
    def test_conv2d(
        self,
        _,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(
                    3,
                    6,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    groups,
                    bias,
                )

            def forward(self, x):
                return self.conv(x)

        inputs = [torch.randn(1, 3, 32, 32)]
        self.run_test(
            TestModule(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    # Testing with (-1, -1, -1, -1) results into Error:
    # AssertionError: Channel dim can't be dynamic for convolution.

    def test_conv2d_with_dynamic_shape(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 6, 1)

            def forward(self, x):
                return self.conv(x)

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
            param("list_stride", 2, stride=[2]),
            param("non_zero_padding", 1, padding=1),
            param("list_zero_padding", 1, padding=[0]),
            param("list_non_padding", 1, padding=[1]),
            param("dilation", 2, dilation=2),
            param("list_dilation", 2, dilation=[2]),
            ## TODO TRT 8.4.1 will trigger issue with this test. T127981773
            # param("groups", 1, groups=3),
        ]
    )
    def test_conv3d(
        self,
        _,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv3d(
                    3, 6, kernel_size, stride, padding, dilation, groups, bias
                )

            def forward(self, x):
                return self.conv(x)

        inputs = [torch.randn(1, 3, 32, 32, 32)]
        self.run_test(
            TestModule(),
            inputs,
            use_dynamo_tracer=True,
            enable_passes=True,
        )

    # Testing with (-1, -1, -1, -1, -1) results into Error:
    # AssertionError: Channel dim can't be dynamic for convolution.

    def test_conv3d_with_dynamic_shape(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv3d(3, 6, 1)

            def forward(self, x):
                return self.conv(x)

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
