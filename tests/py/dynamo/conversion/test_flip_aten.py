import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestFlipConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((3,), [0]),
            ((3,), [-1]),
            ((3, 3), [0, 1]),
            ((3, 3), [-2, 1]),
            ((2, 3, 4), [0]),
            ((3, 3, 3), (0, 1)),
            ((2, 3, 4), [0, 1, 2]),
            ((2, 3, 4), [-3, -2, -1]),
            ((3, 3, 3, 3), [0]),
            ((2, 3, 4, 5), [0, 1, 2, 3]),
            ((2, 3, 4, 5), [-4, 1, -2, 3]),
        ]
    )
    def test_flip(self, shape, dims):
        class Flip(nn.Module):
            def forward(self, x):
                return torch.ops.aten.flip.default(x, dims)

        inputs = [torch.randn(shape)]
        self.run_test(Flip(), inputs)


if __name__ == "__main__":
    run_tests()
