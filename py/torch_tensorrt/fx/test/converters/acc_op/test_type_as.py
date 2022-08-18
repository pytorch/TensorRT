import unittest

import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec
from torch_tensorrt.fx.utils import LowerPrecision


class TestTypeAsConverter(AccTestCase):
    def test_device_fp32(self):
        class Type_as(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.randn(2, 2)

            def forward(self, x):
                b = self.a.type_as(x)
                return b

                # self.a = self.a.type_as(x) # error is throw
                # return self.a

        input = torch.randn(2, 2).cuda()
        inputs = [
            input,
        ]
        self.run_test(
            Type_as(),
            inputs,
            expected_ops={acc_ops.to_dtype, acc_ops.device, acc_ops.dtype},
            test_implicit_batch_dim=False,
        )

    def test_device_fp16(self):
        class Type_as(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.randn(2, 2)

            def forward(self, x):
                return self.a.type_as(x)

        input = torch.randn(2, 2).half().cuda()
        inputs = [
            input,
        ]
        self.run_test(
            Type_as(),
            inputs,
            expected_ops={acc_ops.to_dtype, acc_ops.device, acc_ops.dtype},
            test_implicit_batch_dim=False,
            precision=LowerPrecision.FP16,
        )

    def test_device_fp32_tensor(self):
        class Type_as(torch.nn.Module):
            def forward(self, input, other):
                return other.type_as(input)

        input = torch.randn(2, 2).cuda()
        other = torch.randn(2, 2)
        inputs = [
            input,
            other,
        ]
        self.run_test(
            Type_as(),
            inputs,
            expected_ops={acc_ops.to_dtype, acc_ops.device, acc_ops.dtype},
        )

    def test_device_fp16_tensor(self):
        class Type_as(torch.nn.Module):
            def forward(self, input, other):
                return other.type_as(input)

        input = torch.randn(2, 2).half().cuda()
        other = torch.randn(2, 2)
        inputs = [
            input,
            other,
        ]
        self.run_test(
            Type_as(),
            inputs,
            expected_ops={acc_ops.to_dtype, acc_ops.device, acc_ops.dtype},
            precision=LowerPrecision.FP16,
        )

    def test_type_tensor(self):
        class Type_as(torch.nn.Module):
            def forward(self, input):
                return input.type(dtype=torch.float16)

        input = torch.randn(2, 2)

        inputs = [
            input,
        ]
        self.run_test(
            Type_as(),
            inputs,
            expected_ops={acc_ops.to_dtype},
            precision=LowerPrecision.FP16,
        )

    @unittest.skip("Does not pass in TRT 8.4.1 T127981773")
    def test_type_tensor_with_dynamic_shape_four_dimensions(self):
        class Type_as(torch.nn.Module):
            def forward(self, input):
                return input.type(dtype=torch.float32)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.int,
                shape_ranges=[((1, 1, 1, 1), (3, 3, 3, 3), (3, 3, 3, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            Type_as(),
            input_specs,
            expected_ops={acc_ops.to_dtype},
        )

    def test_type_tensor_ext(self):
        class Type_as(torch.nn.Module):
            def forward(self, input, other):
                t = input.type()
                return other.type(t)

        input = torch.randn(2, 2).to(dtype=torch.float16)
        other = torch.randn(2, 2)

        inputs = [
            input,
            other,
        ]
        self.run_test(
            Type_as(),
            inputs,
            expected_ops={acc_ops.to_dtype, acc_ops.dtype},
            precision=LowerPrecision.FP16,
        )


if __name__ == "__main__":
    run_tests()
