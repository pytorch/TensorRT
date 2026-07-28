import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestRollConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((4,), (2,), 0),
            ((4,), [2], [0]),
            ((4,), [3], [0]),
            ((4,), [-3, 2], [0, 0]),
            ((4,), [-2], []),
            ((4, 2), [2, 1], [0, 1]),
            ((3, 3), [2, 1], [1, 1]),
            ((4, 2), [2, -1], [-2, -1]),
            ((4, 2), [4], []),
            ((3, 4, 2), [1, 0, 2], [2, 0, -2]),
            ((3, 4, 2), [1, -0, 2], [1, 1, 1]),
            (
                (3, 4, 2),
                [
                    5,
                ],
                [],
            ),
            ((2, 3), [1], [1]),
        ]
    )
    def test_roll_static(self, shape, shifts, dims):
        class Roll(nn.Module):
            def forward(self, x):
                return torch.ops.aten.roll.default(x, shifts, dims)

        inputs = [torch.randn(shape)]
        self.run_test(Roll(), inputs)

    @parameterized.expand(
        [
            # dim is empty
            ((2,), (3,), (4,), [1], []),
            ((2, 3), (3, 4), (4, 5), [1], []),
            ((2, 3), (3, 4), (4, 5), [2], []),
            ((2, 3), (3, 4), (4, 5), [-15], []),
            ((2, 3, 3), (3, 4, 3), (4, 5, 4), [1], []),
            # dim is not empty
            ((2,), (3,), (4,), [1], [0]),
            ((2, 3), (3, 4), (4, 5), [1], [1]),
            ((2, 3), (3, 4), (4, 5), [2, 0], [0, 1]),
            ((2, 3, 4), (3, 4, 5), (4, 5, 6), [-15, -2, 1], [0, 0, 1]),
            ((2, 3, 3, 5), (3, 4, 3, 5), (4, 5, 4, 6), [11, -23], [0, 1]),
        ]
    )
    def test_roll_dynamic_input_static_shifts(
        self, min_shape, opt_shape, max_shape, shifts, dims
    ):
        class Roll(nn.Module):
            def forward(self, x):
                return torch.ops.aten.roll.default(x, shifts, dims)

        inputs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            )
        ]
        self.run_test_with_dynamic_shape(Roll(), inputs)

    @parameterized.expand(
        [
            ((2, 3), (3, 3), (4, 3)),
            ((2, 3), (3, 4), (4, 5)),
            ((2, 3, 4), (3, 4, 5), (3, 5, 5)),
        ]
    )
    def test_roll_dynamic_input_dynamic_shifts(self, min_shape, opt_shape, max_shape):
        class Roll(nn.Module):
            def forward(self, x):
                dims = [0, 1]
                shifts = [x.shape[d] // 2 for d in dims]
                return torch.ops.aten.roll.default(x, shifts, dims)

        inputs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
            )
        ]
        self.run_test_with_dynamic_shape(Roll(), inputs, use_dynamo_tracer=True)


if __name__ == "__main__":
    run_tests()
