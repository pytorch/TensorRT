import unittest

import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import DispatchTestCase, InputTensorSpec

class TestSelectConverter(DispatchTestCase):
    def test_select(self):
        class TestModule(torch.nn.Module):
            def forward(self, input, dim, index):
                return torch.select(input, dim, index)
        input = [torch.randn(1, 3, 32)]
        dim = 2
        index = 1
        inputs = (input, dim, index)
        self.run_test(
            TestModule(), input, expected_ops={torch.ops.aten.select.Tensor}, test_explicit_precision=True,
        )

    def test_select_with_dynamic_shape(self, x, y):
        class TestModule(torch.nn.Module):
            def forward(self, input, dim, index):
                return torch.select(input, dim, index)

        input_spec = [
            InputTensorSpec(
                shape=(-1, 3, 32),
                dtype=torch.float32,
                shape_ranges=[((1, 3, 3), (3, 3, 3), (32, 32, 32))],
            ),
        ]
        dim = 2
        index = 1
        inputs_spec = (input_spec, dim, index)
        self.run_test_with_dynamic_shape(
            TestModule(), inputs_spec, expected_ops={torch.ops.aten.select.Tensor}
        )