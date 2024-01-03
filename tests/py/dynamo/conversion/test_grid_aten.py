import pytest
import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase

grid_sampler_ops = [
    (
        "input_grid_interpolation_nearest_sample_fill",
        (lambda x, grid: torch.ops.aten.grid_sampler(x, grid, 0, 0, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
    (
        "input_grid_interpolation_nearest_sample_clamp",
        (lambda x, grid: torch.ops.aten.grid_sampler(x, grid, 0, 1, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
    (
        "input_grid_interpolation_nearest_sample_reflect",
        (lambda x, grid: torch.ops.aten.grid_sampler(x, grid, 0, 2, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
    (
        "input_grid_interpolation_linear_sample_fill",
        (lambda x, grid: torch.ops.aten.grid_sampler(x, grid, 0, 0, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
    (
        "input_grid_interpolation_linear_sample_clamp",
        (lambda x, grid: torch.ops.aten.grid_sampler(x, grid, 0, 1, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
    (
        "input_grid_interpolation_linear_sample_reflect",
        (lambda x, grid: torch.ops.aten.grid_sampler(x, grid, 0, 2, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
    (
        "input_grid_interpolation_cubic_sample_fill",
        (lambda x, grid: torch.ops.aten.grid_sampler(x, grid, 0, 0, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
    (
        "input_grid_interpolation_cubic_sample_clamp",
        (lambda x, grid: torch.ops.aten.grid_sampler(x, grid, 0, 1, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
    (
        "input_grid_interpolation_cubic_sample_reflect",
        (lambda x, grid: torch.ops.aten.grid_sampler(x, grid, 0, 2, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
    (
        "input_grid_interpolation_nearest_sample_fill_2d",
        (lambda x, grid: torch.ops.aten.grid_sampler_2d(x, grid, 0, 0, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
    (
        "input_grid_interpolation_nearest_sample_clamp_2d",
        (lambda x, grid: torch.ops.aten.grid_sampler_2d(x, grid, 0, 1, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
    (
        "input_grid_interpolation_nearest_sample_reflect_2d",
        (lambda x, grid: torch.ops.aten.grid_sampler_2d(x, grid, 0, 2, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
    (
        "input_grid_interpolation_linear_sample_fill_2d",
        (lambda x, grid: torch.ops.aten.grid_sampler_2d(x, grid, 0, 0, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
    (
        "input_grid_interpolation_linear_sample_clamp_2d",
        (lambda x, grid: torch.ops.aten.grid_sampler_2d(x, grid, 0, 1, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
    (
        "input_grid_interpolation_linear_sample_reflect_2d",
        (lambda x, grid: torch.ops.aten.grid_sampler_2d(x, grid, 0, 2, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
    (
        "input_grid_interpolation_cubic_sample_fill_2d",
        (lambda x, grid: torch.ops.aten.grid_sampler_2d(x, grid, 0, 0, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
    (
        "input_grid_interpolation_cubic_sample_clamp_2d",
        (lambda x, grid: torch.ops.aten.grid_sampler_2d(x, grid, 0, 1, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
    (
        "input_grid_interpolation_cubic_sample_reflect_2d",
        (lambda x, grid: torch.ops.aten.grid_sampler_2d(x, grid, 0, 2, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
]


class TestGridConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (
                grid_sampler_op[0],
                grid_sampler_op[1],
                grid_sampler_op[2],
                grid_sampler_op[3],
            )
            for grid_sampler_op in grid_sampler_ops
        ]
    )
    def test_grid(self, _, op, input_shape, dim_shape):
        class TestModule(nn.Module):
            def __init__(self, grid_sampler_op):
                super().__init__()
                self.grid_sampler_op = grid_sampler_op

            def forward(self, x):
                grid = torch.randint(-1, 1, dim_shape, dtype=torch.float32)
                return self.grid_sampler_op(x, grid)

        inputs = [torch.randn(input_shape, dtype=torch.float32)]
        grid_model = TestModule(op)
        self.run_test(grid_model, inputs)


if __name__ == "__main__":
    run_tests()
