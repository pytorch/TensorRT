import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestLogicalAndConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_logical_and(self, _, shape):
        class logical_and(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.logical_and(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            logical_and(),
            inputs,
            expected_ops={torch.ops.aten.logical_and.default},
        )


if __name__ == "__main__":
    run_tests()
