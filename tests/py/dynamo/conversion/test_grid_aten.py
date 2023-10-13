import pytest
import torch
import torch.nn as nn
from .harness import DispatchTestCase
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input


class TestGridConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (
                "input_grid_interpolation_nearest_sample_fill",
                [1, 1, 5, 5],
                [1, 5, 2, 2],
                0,
                0,
            ),
            (
                "input_grid_interpolation_nearest_sample_clamp",
                [1, 1, 5, 5],
                [1, 5, 2, 2],
                0,
                1,
            ),
            (
                "input_grid_interpolation_nearest_sample_reflect",
                [1, 1, 5, 5],
                [1, 5, 2, 2],
                0,
                2,
            ),
            (
                "input_grid_interpolation_linear_sample_fill",
                [1, 1, 5, 5],
                [1, 5, 2, 2],
                1,
                0,
            ),
            (
                "input_grid_interpolation_linear_sample_clamp",
                [1, 1, 5, 5],
                [1, 5, 2, 2],
                1,
                1,
            ),
            (
                "input_grid_interpolation_linear_sample_reflect",
                [1, 1, 5, 5],
                [1, 5, 2, 2],
                1,
                2,
            ),
            (
                "input_grid_interpolation_cubic_sample_fill",
                [1, 1, 5, 5],
                [1, 5, 2, 2],
                2,
                0,
            ),
            (
                "input_grid_interpolation_cubic_sample_clamp",
                [1, 1, 5, 5],
                [1, 5, 2, 2],
                2,
                1,
            ),
            (
                "input_grid_interpolation_cubic_sample_reflect",
                [1, 1, 5, 5],
                [1, 5, 2, 2],
                2,
                2,
            ),
        ]
    )
    def test_grid(self, _, input_shape, dim_shape, interpolation, sample):
        class TestModule(nn.Module):
            def forward(self, x):
                grid = torch.randint(-1, 1, dim_shape, dtype=torch.float32)
                return torch.ops.aten.grid_sampler(x, grid, interpolation, sample, True)

        inputs = [torch.randn(input_shape, dtype=torch.float32)]
        self.run_test(TestModule(), inputs)


if __name__ == "__main__":
    run_tests()
