import pytest
import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase

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

    @parameterized.expand(
        [
            (
                (1, 1, 2, 2),
                (2, 2, 3, 3),
                (3, 3, 5, 5),
                (1, 2, 2, 2),
                (2, 3, 3, 2),
                (3, 5, 5, 2),
                0,
                0,
                True,
            ),
            (
                (1, 1, 2, 2),
                (2, 2, 3, 3),
                (3, 3, 5, 5),
                (1, 2, 2, 2),
                (2, 3, 3, 2),
                (3, 5, 5, 2),
                0,
                2,
                True,
            ),
            (
                (1, 1, 2, 2),
                (1, 1, 3, 3),
                (1, 1, 5, 5),
                (1, 3, 3, 2),
                (1, 4, 4, 2),
                (1, 5, 5, 2),
                0,
                1,
                True,
            ),
            (
                (1, 1, 2, 2),
                (2, 2, 3, 3),
                (3, 3, 5, 5),
                (1, 4, 2, 2),
                (2, 4, 3, 2),
                (3, 4, 5, 2),
                1,
                0,
                True,
            ),
            (
                (1, 1, 2, 2),
                (2, 2, 3, 3),
                (3, 3, 5, 5),
                (1, 4, 2, 2),
                (2, 5, 3, 2),
                (3, 5, 5, 2),
                1,
                1,
                False,
            ),
        ]
    )
    def test_grid_2d_default_dynamic_shape(
        self,
        input_min_shape,
        input_opt_shape,
        input_max_shape,
        grid_min_shape,
        grid_opt_shape,
        grid_max_shape,
        interpolation_mode,
        padding_mode,
        align_corners,
    ):
        class Grid_SAMPLER_2D(nn.Module):
            def forward(self, input, grid):
                return torch.ops.aten.grid_sampler_2d(
                    input, grid, interpolation_mode, padding_mode, align_corners
                )

        class Grid_SAMPLER_2D_default(nn.Module):
            def forward(self, input, grid):
                return torch.ops.aten.grid_sampler_2d.default(
                    input, grid, interpolation_mode, padding_mode, align_corners
                )

        class Grid_SAMPLER(nn.Module):
            def forward(self, input, grid):
                return torch.ops.aten.grid_sampler(
                    input, grid, interpolation_mode, padding_mode, align_corners
                )

        class Grid_SAMPLER_default(nn.Module):
            def forward(self, input, grid):
                return torch.ops.aten.grid_sampler.default(
                    input, grid, interpolation_mode, padding_mode, align_corners
                )

        inputs = [
            torch_tensorrt.Input(
                min_shape=input_min_shape,
                opt_shape=input_opt_shape,
                max_shape=input_max_shape,
                dtype=torch.float32,
                torch_tensorrt=torch.randn(input_opt_shape, dtype=torch.float32),
            ),
            torch_tensorrt.Input(
                min_shape=grid_min_shape,
                opt_shape=grid_opt_shape,
                max_shape=grid_max_shape,
                dtype=torch.float32,
                torch_tensor=torch.randint(-1, 1, grid_opt_shape, dtype=torch.float32),
            ),
        ]
        self.run_test_with_dynamic_shape(
            Grid_SAMPLER_2D(),
            inputs,
            use_example_tensors=False,
        )
        self.run_test_with_dynamic_shape(
            Grid_SAMPLER_2D_default(),
            inputs,
            use_example_tensors=False,
        )
        self.run_test_with_dynamic_shape(
            Grid_SAMPLER(),
            inputs,
            use_example_tensors=False,
        )
        self.run_test_with_dynamic_shape(
            Grid_SAMPLER_default(),
            inputs,
            use_example_tensors=False,
        )


if __name__ == "__main__":
    run_tests()
