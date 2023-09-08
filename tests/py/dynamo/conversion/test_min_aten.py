import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestMinConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_min(self, _, shape):
        class min(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.min(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            min(),
            inputs,
            expected_ops={torch.ops.aten.minimum.default},
        )


if __name__ == "__main__":
    run_tests()
