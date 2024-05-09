import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestCdistConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((4, 3, 4), 0),
            ((10, 3, 5, 2, 6), 0.5),
            ((15, 10, 5), 0),
            ((15, 10, 5), 0.5),
            ((15, 10, 5), 1.0),
            ((15, 10, 5), 1.5),
            ((15, 10, 5), 2.0),
            ((15, 10, 5), 2.99),
            ((15, 10, 5), float("inf")),
        ]
    )
    def test_cdist_float(self, shape, p):
        class Cdist(nn.Module):
            def forward(self, x1, x2):
                print("x1 : ", x1)
                print("x2 : ", x2)

                return torch.ops.aten._cdist_forward.default(x1, x2, p, None)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            Cdist(),
            inputs,
        )

    @parameterized.expand(
        [
            ((1, 5), (2, 3, 5), 1),
            ((4, 5), (2, 3, 5), 1),
            ((2, 4, 5), (2, 3, 5), 1),
            ((2, 2, 4, 5), (2, 3, 5), 0),
            ((2, 2, 4, 5), (2, 3, 5), 0.5),
            ((2, 2, 4, 5), (2, 3, 5), 1),
            ((2, 2, 4, 5), (2, 3, 5), 1.5),
            ((2, 2, 4, 5), (2, 3, 5), 2),
            ((2, 2, 4, 5), (2, 3, 5), 2.99),
            ((2, 2, 4, 5), (2, 3, 5), float("inf")),
        ]
    )
    def test_cdist_broadcast_float(self, shape_1, shape_2, p):
        class Cdist(nn.Module):
            def forward(self, x1, x2):
                return torch.ops.aten._cdist_forward.default(x1, x2, p, None)

        inputs = [torch.randn(shape_1), torch.randn(shape_2)]
        self.run_test(
            Cdist(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
