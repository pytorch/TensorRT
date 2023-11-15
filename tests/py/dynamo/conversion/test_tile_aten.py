import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

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


if __name__ == "__main__":
    run_tests()
