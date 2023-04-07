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

        inputOne = torch.randn(2, 32)
        inputTwo = torch.randn(32, 2)
        inputs = [inputOne, inputTwo]
        self.run_test(
            TestModule(),
            inputs,
            expected_ops={torch.ops.aten.mm.default},
        )


if __name__ == "__main__":
    run_tests()
