import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestPdistConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((2, 3), 0),
            ((2, 3), 0.4),
            ((2, 3), 1),
            ((2, 3), 1.5),
            ((3, 4), 2),
            ((3, 4), 2.99),
            ((4, 5), 3),
            ((4, 5), 3.3),
            ((5, 6), float("inf")),
        ]
    )
    def test_pdist_float(self, shape, p):
        class Pdist(nn.Module):
            def forward(self, input):
                return torch.ops.aten._pdist_forward.default(input, p)

        inputs = [torch.randn(shape)]
        self.run_test(
            Pdist(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
