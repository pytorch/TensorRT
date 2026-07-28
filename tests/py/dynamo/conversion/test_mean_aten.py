import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestMeanDimConverter(DispatchTestCase):
    def test_mean_dim_keepdims(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.ops.aten.mean.dim(x, [0, 1], True)

        inputs = [torch.randn(1, 10)]
        self.run_test(TestModule(), inputs)

    def test_mean_dim_keepdims_with_dynamic_shape(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.ops.aten.mean.dim(x, [0, 1, 2], True)

        input_specs = [
            Input(
                shape=(-1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1), (1, 2, 3), (3, 3, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    def test_mean_dim_keepdims_false(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.ops.aten.mean.dim(x, 0, False)

        inputs = [torch.randn(3, 5, 7)]
        self.run_test(TestModule(), inputs)

    def test_mean_dim_keepdims_false_with_dynamic_shape(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.ops.aten.mean.dim(x, -1, False)

        input_specs = [
            Input(
                shape=(-1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1), (1, 2, 3), (3, 3, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)


class TestMeanConverter(DispatchTestCase):
    def test_mean(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.ops.aten.mean.default(x)

        inputs = [torch.randn(3, 8, 5, 7, 1)]
        self.run_test(TestModule(), inputs)

    def test_mean_with_dynamic_shape(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.ops.aten.mean.default(x)

        input_specs = [
            Input(
                shape=(-1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1), (1, 5, 8), (3, 10, 10))],
            ),
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)


if __name__ == "__main__":
    run_tests()
