import unittest

import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import DispatchTestCase, InputTensorSpec

class TestMatMulConverter(DispatchTestCase):
    def test_matmul(self):
        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                return torch.matmul(x, y)
        inputOne = torch.randn(1, 32)
        inputTwo = torch.randn(32, 3)
        inputs = [inputOne, inputTwo]
        self.run_test(
            TestModule(), inputs, expected_ops={torch.ops.aten.matmul},
        )

    def test_matmul_with_dynamic_shape(self):
        class TestModule(torch.nn.Module):
            def forward(self, x, y):
                return torch.matmul(x, y)

        input_specs = [
            InputTensorSpec(
                shape=(-1, 3),
                dtype=torch.float32,
                shape_ranges=[((3, 3, 1), (3, 3, 3))],
            ),
            InputTensorSpec(
                shape=(3, -1),
                dtype=torch.float32,
                shape_ranges=[((3, 3, 3), (5, 3, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={torch.ops.aten.mm},
        )

if __name__ == "__main__":
    run_tests()