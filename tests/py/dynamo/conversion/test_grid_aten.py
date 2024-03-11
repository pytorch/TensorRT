import pytest
import torch
import torch.nn as nn
from .harness import DispatchTestCase
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

grid_sampler_aten_ops = {
    "torch.ops.aten.grid_sampler": torch.ops.aten.grid_sampler,
    "torch.ops.aten.grid_sampler_2d": torch.ops.aten.grid_sampler_2d,
    "torch.ops.aten.grid_sampler.default": torch.ops.aten.grid_sampler.default,
    "torch.ops.aten.grid_sampler_2d.default": torch.ops.aten.grid_sampler_2d.default,
}

grid_sampler_ops = [
    (
        "input_grid_interpolation_nearest_sample_fill",
        "torch.ops.aten.grid_sampler",
        (lambda x, grid, op: op(x, grid, 0, 0, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
    (
        "input_grid_interpolation_nearest_sample_clamp",
        "torch.ops.aten.grid_sampler",
        (lambda x, grid, op: op(x, grid, 0, 1, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
    (
        "input_grid_interpolation_nearest_sample_reflect",
        "torch.ops.aten.grid_sampler",
        (lambda x, grid, op: op(x, grid, 0, 2, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
    (
        "input_grid_interpolation_linear_sample_fill",
        "torch.ops.aten.grid_sampler",
        (lambda x, grid, op: op(x, grid, 1, 0, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
    (
        "input_grid_interpolation_linear_sample_clamp",
        "torch.ops.aten.grid_sampler",
        (lambda x, grid, op: op(x, grid, 1, 1, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
    (
        "input_grid_interpolation_linear_sample_reflect",
        "torch.ops.aten.grid_sampler",
        (lambda x, grid, op: op(x, grid, 1, 2, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
    (
        "input_grid_interpolation_cubic_sample_fill",
        "torch.ops.aten.grid_sampler",
        (lambda x, grid, op: op(x, grid, 2, 0, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
    (
        "input_grid_interpolation_cubic_sample_clamp",
        "torch.ops.aten.grid_sampler",
        (lambda x, grid, op: op(x, grid, 2, 1, True)),
        [1, 1, 5, 5],
        [1, 5, 2, 2],
    ),
    (
        "input_grid_interpolation_cubic_sample_reflect",
        "torch.ops.aten.grid_sampler",
        (lambda x, grid, op: op(x, grid, 2, 2, True)),
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
                grid_sampler_op[4],
            )
            for grid_sampler_op in grid_sampler_ops
        ]
    )
    def test_grid(self, _, op_name, op, input_shape, dim_shape):
        class TestModule(nn.Module):
            def __init__(self, grid_sampler_op):
                super().__init__()
                self.grid_sampler_op = grid_sampler_op

            def forward(self, x):
                grid = torch.randint(-1, 1, dim_shape, dtype=torch.float32)
                return self.grid_sampler_op(x, grid, grid_sampler_aten_ops[op_name])

        inputs = [torch.randn(input_shape, dtype=torch.float32)]
        grid_model = TestModule(op)
        self.run_test(grid_model, inputs)

    @parameterized.expand(
        [
            (
                grid_sampler_op[0],
                grid_sampler_op[1] + "_2d",
                grid_sampler_op[2],
                grid_sampler_op[3],
                grid_sampler_op[4],
            )
            for grid_sampler_op in grid_sampler_ops
        ]
    )
    def test_grid_2d(self, _, op_name, op, input_shape, dim_shape):
        class TestModule(nn.Module):
            def __init__(self, grid_sampler_op):
                super().__init__()
                self.grid_sampler_op = grid_sampler_op

            def forward(self, x):
                grid = torch.randint(-1, 1, dim_shape, dtype=torch.float32)
                return self.grid_sampler_op(x, grid, grid_sampler_aten_ops[op_name])

        inputs = [torch.randn(input_shape, dtype=torch.float32)]
        grid_model = TestModule(op)
        self.run_test(grid_model, inputs)

    @parameterized.expand(
        [
            (
                grid_sampler_op[0],
                grid_sampler_op[1] + ".default",
                grid_sampler_op[2],
                grid_sampler_op[3],
                grid_sampler_op[4],
            )
            for grid_sampler_op in grid_sampler_ops
        ]
    )
    def test_grid_default(self, _, op_name, op, input_shape, dim_shape):
        class TestModule(nn.Module):
            def __init__(self, grid_sampler_op):
                super().__init__()
                self.grid_sampler_op = grid_sampler_op

            def forward(self, x):
                grid = torch.randint(-1, 1, dim_shape, dtype=torch.float32)
                return self.grid_sampler_op(x, grid, grid_sampler_aten_ops[op_name])

        inputs = [torch.randn(input_shape, dtype=torch.float32)]
        grid_model = TestModule(op)
        self.run_test(grid_model, inputs)

    @parameterized.expand(
        [
            (
                grid_sampler_op[0],
                grid_sampler_op[1] + "_2d.default",
                grid_sampler_op[2],
                grid_sampler_op[3],
                grid_sampler_op[4],
            )
            for grid_sampler_op in grid_sampler_ops
        ]
    )
    def test_grid_2d_default(self, _, op_name, op, input_shape, dim_shape):
        class TestModule(nn.Module):
            def __init__(self, grid_sampler_op):
                super().__init__()
                self.grid_sampler_op = grid_sampler_op

            def forward(self, x):
                grid = torch.randint(-1, 1, dim_shape, dtype=torch.float32)
                return self.grid_sampler_op(x, grid, grid_sampler_aten_ops[op_name])

        inputs = [torch.randn(input_shape, dtype=torch.float32)]
        grid_model = TestModule(op)
        self.run_test(grid_model, inputs)


if __name__ == "__main__":
    run_tests()
