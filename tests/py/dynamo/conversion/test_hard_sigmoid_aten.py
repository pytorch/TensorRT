import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestHardSigmoidConverter(DispatchTestCase):
    def test_hardsigmoid(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.ops.aten.hardsigmoid.default(x)

        inputs = [torch.randn(1, 10)]
        self.run_test(TestModule(), inputs)

    def test_hardsigmoid_with_dynamic_shape(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.ops.aten.hardsigmoid.default(x)

        input_specs = [
            Input(
                shape=(-1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1), (1, 2, 3), (3, 3, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    def test_hardsigmoid_with_dynamic_shape_four_dimensions(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.ops.aten.hardsigmoid.default(x)

        input_specs = [
            Input(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 5), (1, 2, 3, 5), (3, 3, 3, 5))],
            ),
        ]

        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    def test_hardsigmoid_fp16(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.ops.aten.hardsigmoid.default(x)

        inputs = [torch.randn(1, 10)]
        self.run_test(
            TestModule(),
            inputs,
            precision=torch.half,
            check_dtype=False,
        )


if __name__ == "__main__":
    run_tests()
