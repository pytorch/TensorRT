import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestDropOutConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((10,), 0, False),
            ((1, 3), 0.3, False),
            ((2, 2, 2), 0.5),
            ((2, 2, 2, 2), 1),
        ]
    )
    def test_native_dropout(self, input_shape, p, train=False):
        class NativeDropout(nn.Module):
            def forward(self, input):
                return torch.ops.aten.native_dropout.default(input, p, train)

        inputs = [torch.randn(input_shape)]
        self.run_test(
            NativeDropout(),
            inputs,
        )

    @parameterized.expand(
        [
            (
                torch.randn(
                    10,
                ),
                0,
                False,
            ),
            (torch.randn(1, 3), 0.3, False),
            (torch.randn(2, 2, 2), 0.5),
            (torch.randn(2, 2, 2, 2), 1),
        ]
    )
    def test_native_dropout_pytorch(self, input, p, train=False):
        class NativeDropout(nn.Module):
            def forward(self):
                return torch.ops.aten.native_dropout.default(input, p, train)

        inputs = []
        self.run_test(
            NativeDropout(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
