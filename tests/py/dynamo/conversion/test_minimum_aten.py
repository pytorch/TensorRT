import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestMinimumConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_minimum(self, _, shape):
        class Minimum(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.minimum(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            Minimum(),
            inputs,
            use_dynamo_tracer=True,
        )


if __name__ == "__main__":
    run_tests()
