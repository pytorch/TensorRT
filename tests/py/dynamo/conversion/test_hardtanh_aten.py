import unittest

import torch
import torch.nn as nn
import torch_tensorrt
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestHardTanHConverter(DispatchTestCase):
    def test_hardtanh(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.ops.aten.hardtanh.default(x, -1.0, 1.0)

        inputs = [torch.randn(1, 10)]
        self.run_test(TestModule(), inputs)

    def test_hardtanh_with_dynamic_shape(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.ops.aten.hardtanh.default(x, -1.0, 1.0)

        input_specs = [
            Input(
                shape=(-1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1), (1, 2, 3), (3, 3, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    def test_hardtanh_with_dynamic_shape_four_dimensions(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.ops.aten.hardtanh.default(x, -1.0, 1.0)

        input_specs = [
            Input(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 5), (1, 2, 3, 5), (3, 3, 3, 5))],
            ),
        ]

        self.run_test_with_dynamic_shape(TestModule(), input_specs)


if __name__ == "__main__":
    run_tests()
