import unittest

import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import DispatchTestCase, InputTensorSpec

class TestMatMulConverter(DispatchTestCase):
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
            expected_ops={torch.ops.aten.convolution.default},
            test_explicit_precision=True,
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
            InputTensorSpec(
                shape=(-1, 3, 3),
                dtype=torch.float32,
                shape_ranges=[((1, 3, 3), (3, 3, 3), (5, 3, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={torch.ops.aten.convolution.default}
        )

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