import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestTileConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((3,), (1,)),
            ((3,), (0,)),
            ((3,), (2,)),
            ((2,), (2, 2)),
            ((2,), (0, 2)),
        ]
    )
    def test_tile_1D(self, shape, dims):
        class Tile(nn.Module):
            def forward(self, x):
                return torch.ops.aten.tile.default(x, dims)

        inputs = [torch.randn(shape)]
        self.run_test(
            Tile(),
            inputs,
        )

    @parameterized.expand(
        [
            ((3, 1), (0,)),
            ((3, 1), (2,)),
            ((2, 3), (2, 2)),
            ((2, 3), (1, 0)),
            ((2, 3), (0, 2)),
            ((2, 3), (4, 2, 3)),
            ((2, 3), (0, 0, 3)),
            ((2, 3), (4, 2, 3, 1, 2)),
        ]
    )
    def test_tile_2D(self, shape, dims):
        class Tile(nn.Module):
            def forward(self, x):
                return torch.ops.aten.tile.default(x, dims)

        inputs = [torch.randn(shape)]
        self.run_test(
            Tile(),
            inputs,
        )

    @parameterized.expand(
        [
            ((4, 2, 3), (2,)),
            ((4, 2, 3), (1, 2)),
            ((1, 2, 3), (2, 3)),
            ((1, 2, 3), (2, 3, 4)),
            ((1, 2, 3), (2, 3, 4, 5)),
        ]
    )
    def test_tile_3D(self, shape, dims):
        class Tile(nn.Module):
            def forward(self, x):
                return torch.ops.aten.tile.default(x, dims)

        inputs = [torch.randn(shape)]
        self.run_test(
            Tile(),
            inputs,
        )


class TestTileConverterDynamicShape(DispatchTestCase):
    @parameterized.expand(
        [
            ((3,), (3,), (6,), (1,)),
            ((3,), (3,), (6,), (0,)),
            ((3,), (3,), (6,), (2,)),
            ((2,), (3,), (6,), (2, 2)),
            ((2,), (3,), (6,), (0, 2)),
            # 2d cases
            ((3, 1), (3, 1), (6, 1), (0,)),
            ((3, 1), (3, 1), (6, 1), (2,)),
            ((2, 3), (2, 3), (4, 3), (2, 2)),
            ((2, 3), (2, 3), (4, 3), (1, 0)),
            ((2, 3), (2, 3), (4, 3), (0, 2)),
            ((2, 3), (2, 3), (4, 3), (4, 2, 3)),
            ((2, 3), (2, 3), (4, 3), (0, 0, 3)),
            ((2, 3), (2, 3), (4, 3), (4, 2, 3, 1, 2)),
            # 3d cases
            ((4, 2, 3), (4, 2, 3), (6, 2, 3), (2,)),
            ((4, 2, 3), (4, 2, 3), (6, 2, 3), (1, 2)),
            ((1, 2, 3), (1, 2, 3), (6, 2, 3), (2, 3)),
            ((1, 2, 3), (1, 2, 3), (6, 2, 3), (2, 3, 4)),
            ((1, 2, 3), (1, 2, 3), (6, 2, 3), (2, 3, 4, 5)),
        ]
    )
    def test_tile_input_dynamic(self, min_shape, opt_shape, max_shape, dims):
        class Tile(nn.Module):
            def forward(self, x):
                return torch.ops.aten.tile.default(x, dims)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=torch.float32,
            ),
        ]
        self.run_test_with_dynamic_shape(
            Tile(),
            input_specs,
        )


if __name__ == "__main__":
    run_tests()
