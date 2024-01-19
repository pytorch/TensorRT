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
        ]
    )
    def test_roll(self, shape, shifts, dims):
        class Roll(nn.Module):
            def forward(self, x):
                return torch.ops.aten.roll.default(x, shifts, dims)

        inputs = [torch.randn(shape)]
        self.run_test(Roll(), inputs)


if __name__ == "__main__":
    run_tests()
