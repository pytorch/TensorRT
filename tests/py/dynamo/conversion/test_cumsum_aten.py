import unittest

import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.utils import is_tegra_platform, is_thor

from .harness import DispatchTestCase


@unittest.skipIf(
    is_thor() or is_tegra_platform(),
    "Skipped on Thor and Tegra platforms",
)
class TestCumsumConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((1,), 0),
            ((2,), 0),
            ((3,), -1),
        ]
    )
    def test_cumsum_1D(self, shape, dim):
        class Cumsum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.cumsum.default(x, dim)

        inputs = [torch.randn(shape)]
        self.run_test(
            Cumsum(),
            inputs,
            immutable_weights=True,
        )

    @parameterized.expand(
        [
            ((3, 1), 0),
            ((3, 1), 1),
            ((2, 3), -1),
            ((2, 3), -2),
        ]
    )
    def test_cumsum_2D(self, shape, dims):
        class Cumsum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.cumsum.default(x, dims)

        inputs = [torch.randn(shape)]
        self.run_test(
            Cumsum(),
            inputs,
            immutable_weights=True,
        )

    @parameterized.expand(
        [
            ((2, 3, 3), 0),
            ((4, 2, 3), 1),
            ((1, 2, 3), 2),
            ((1, 2, 3), -1),
            ((1, 2, 3), -2),
        ]
    )
    def test_cumsum_3D(self, shape, dims):
        class Cumsum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.cumsum.default(x, dims)

        inputs = [torch.randn(shape)]
        self.run_test(
            Cumsum(),
            inputs,
            immutable_weights=True,
        )

    @parameterized.expand(
        [
            ((1,), (2,), (3,), 0),
            ((1,), (2,), (3,), -1),
            ((2, 3), (2, 4), (2, 5), 0),
            ((2, 3), (3, 4), (4, 5), -1),
            ((1, 2, 2), (2, 2, 3), (3, 3, 3), 0),
            ((1, 2, 2), (2, 2, 3), (3, 2, 3), -2),
            ((1, 2, 2, 3), (2, 2, 3, 4), (3, 3, 4, 5), -3),
            ((1, 2, 2, 3), (2, 2, 3, 4), (3, 3, 4, 5), -2),
        ]
    )
    def test_cumsum_dynamic_shape(self, min_shape, opt_shape, max_shape, dims):
        class Cumsum(nn.Module):
            def forward(self, x):
                return torch.ops.aten.cumsum.default(x, dims)

        inputs = [
            torch_tensorrt.Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            ),
        ]
        self.run_test_with_dynamic_shape(
            Cumsum(),
            inputs,
            immutable_weights=True,
        )


if __name__ == "__main__":
    run_tests()
