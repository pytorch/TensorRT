# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

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

    @parameterized.expand(
        [
            # The tiled tensor has a static shape, only the count is symbolic.
            # "n" stands for the dynamic dim of x.
            ("same_rank", (1, 1, 4), (1, "n", 1)),
            ("fewer_dims_than_rank", (2, 3, 4), ("n",)),
            ("more_dims_than_rank", (3, 4), ("n", 2, 1)),
        ]
    )
    def test_tile_static_input_dynamic_dims(self, _, weight_shape, dims):
        class Tile(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(weight_shape))

            def forward(self, x):
                n = x.shape[1]
                return torch.ops.aten.tile.default(
                    self.weight, [n if d == "n" else d for d in dims]
                )

        input_specs = [
            Input(
                min_shape=(1, 1, 4),
                opt_shape=(1, 3, 4),
                max_shape=(1, 8, 4),
                dtype=torch.float32,
            ),
        ]
        self.run_test_with_dynamic_shape(Tile(), input_specs, use_dynamo_tracer=True)


if __name__ == "__main__":
    run_tests()
