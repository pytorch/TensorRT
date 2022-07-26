import torch
import torch.nn as nn
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


class TestTile(AccTestCase):
    @parameterized.expand(
        [
            ("same_num_dims", (2, 2, 3), (1, 2, 2)),
            ("less_dims", (2, 2, 3), (2,)),
            ("more_dims", (2, 3), (1, 2, 2, 1)),
        ]
    )
    def test_tile(self, _, input_shape, dims):
        class Tile(nn.Module):
            def __init__(self, dims):
                super().__init__()
                self.dims = dims

            def forward(self, x):
                return torch.tile(x, self.dims)

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            Tile(dims),
            inputs,
            expected_ops={acc_ops.tile},
            test_implicit_batch_dim=(
                len(input_shape) > len(dims)
                or (len(input_shape) == len(dims) and dims[0] == 1)
            ),
        )

    @parameterized.expand(
        [
            ("same_num_dims", (-1, 2, 3), (1, 2, 2)),
            ("less_dims", (-1, 2, 3), (2,)),
            ("more_dims", (-1, 3), (1, 2, 2, 1)),
            ("all_dynamic_dim", (-1, -1), (1, 2, 2, 1)),
        ]
    )
    def test_tile_with_dynamic_shape(self, _, shape, dims):
        class Tile(nn.Module):
            def __init__(self, dims):
                super().__init__()
                self.dims = dims

            def forward(self, x):
                return torch.tile(x, self.dims)

        input_specs = [
            InputTensorSpec(
                shape=shape,
                dtype=torch.float32,
                shape_ranges=[
                    (
                        tuple(i if i != -1 else 1 for i in shape),
                        tuple(i if i != -1 else 2 for i in shape),
                        tuple(i if i != -1 else 3 for i in shape),
                    )
                ],
            ),
        ]
        self.run_test_with_dynamic_shape(
            Tile(dims), input_specs, expected_ops={acc_ops.tile}
        )

    @parameterized.expand(
        [
            ("all_dynamic_dim", (-1, -1), (1, 2, 2, 1)),
        ]
    )
    def test_tile_with_dynamic_shape_four_dimensions(self, _, shape, dims):
        class Tile(nn.Module):
            def __init__(self, dims):
                super().__init__()
                self.dims = dims

            def forward(self, x):
                return torch.tile(x, self.dims)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 3), (3, 3, 3, 3), (3, 3, 3, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            Tile(dims), input_specs, expected_ops={acc_ops.tile}
        )

    def test_tile_non_int_dims(self):
        class Tile(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                y = y * 2
                return torch.tile(x, (1, y.shape[1], y.shape[1]))

        inputs = [torch.randn(2, 2, 3), torch.randn(2, 2, 3)]
        batch_size_range = (1, 2, 3)
        input_specs = InputTensorSpec.from_tensors_with_dynamic_batch_size(
            inputs, batch_size_range
        )
        self.run_test_with_dynamic_shape(
            Tile(),
            input_specs,
            expected_ops={acc_ops.tile},
        )

    def test_tile_non_int_dims_with_dynamic_shape_four_dimensions(self):
        class Tile(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                y = y * 2
                return torch.tile(x, (1, y.shape[1], y.shape[1]))

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 3), (3, 3, 3, 3), (3, 3, 3, 3))],
            ),
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 3), (3, 3, 3, 3), (3, 3, 3, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            Tile(), input_specs, expected_ops={acc_ops.tile}
        )


if __name__ == "__main__":
    run_tests()
