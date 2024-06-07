import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestBitwiseNotConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (5, 3)),
            ("3d", (5, 3, 2)),
        ]
    )
    def test_bitwise_not_tensor(self, _, shape):
        class bitwise_not(nn.Module):
            def forward(self, val):
                return torch.ops.aten.bitwise_not.default(val)

        inputs = [
            torch.randint(0, 2, shape, dtype=torch.bool),
        ]
        self.run_test(
            bitwise_not(),
            inputs,
            enable_passes=True,
            use_dynamo_tracer=True,
        )


if __name__ == "__main__":
    run_tests()
