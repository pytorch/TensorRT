import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestAtanConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((10, 1), torch.float),
            # ((1, 20), torch.float),
            # ((2, 3, 4), torch.float),
            # ((2, 3, 4, 5), torch.float),
        ]
    )
    def test_atan_float(self, input_shape, dtype):
        class atan(nn.Module):
            def forward(self, input):
                return torch.ops.aten.nonzero.default(input)

        inputs = [torch.randn(input_shape, dtype=dtype)]
        self.run_test(
            atan(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
