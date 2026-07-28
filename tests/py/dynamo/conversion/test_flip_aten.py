import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestFlipConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((3,), [0]),
            ((3,), [-1]),
            ((3,), []),
            ((3, 3), [0, 1]),
            ((3, 3), [-2, 1]),
            ((2, 3, 4), [0]),
            ((3, 3, 3), (0, 1)),
            ((2, 3, 4), [0, 1, 2]),
            ((2, 3, 4), [-3, -2, -1]),
            ((3, 3, 3, 3), [0]),
            ((2, 3, 4, 5), [0, 1, 2, 3]),
            ((2, 3, 4, 5), [-4, 1, -2, 3]),
            ((2, 3, 4, 5), []),
        ]
    )
    def test_flip(self, shape, dims):
        class Flip(nn.Module):
            def forward(self, x):
                return torch.ops.aten.flip.default(x, dims)

        inputs = [torch.randn(shape)]
        self.run_test(Flip(), inputs)


class TestFlipConverterDynamic(DispatchTestCase):
    @parameterized.expand(
        [
            (
                "3d_dynamic",
                (2, 1, 1),
                (2, 2, 1),
                (3, 2, 4),
                [2, 1, 0],
            ),
            (
                "3d_dynamic_negative_dim",
                (2, 1, 1),
                (2, 2, 1),
                (3, 2, 4),
                [-1, 1],
            ),
            (
                "4d_dynamic_static_dim",
                (3, 1, 1, 1),
                (3, 2, 1, 2),
                (3, 2, 4, 5),
                [0, 2, 3],
            ),
            (
                "3d_dynamic_no_dim",
                (2, 1, 1),
                (2, 2, 1),
                (3, 2, 4),
                [],
            ),
        ]
    )
    def test_flip_dynamic(self, _, min_shape, opt_shape, max_shape, dims):
        class Flip(nn.Module):
            def forward(self, x):
                return torch.ops.aten.flip.default(x, dims)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=torch.float,
            ),
        ]
        self.run_test_with_dynamic_shape(
            Flip(),
            input_specs,
        )


if __name__ == "__main__":
    run_tests()
