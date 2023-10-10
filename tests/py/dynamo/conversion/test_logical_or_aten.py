import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestLogicalOrConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_logical_or(self, _, shape):
        class logical_or(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.logical_or.default(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            logical_or(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
