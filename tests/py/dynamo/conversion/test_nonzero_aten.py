import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestNonZeroConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((10,), torch.int),
            ((1, 20), torch.int32),
            ((2, 3), torch.int64),
            ((2, 3, 4), torch.float),
            ((2, 3, 4, 5), torch.float),
        ]
    )
    def test_non_zero_float(self, input_shape, dtype):
        class NonZero(nn.Module):
            def forward(self, input):
                return torch.ops.aten.nonzero.default(input)

        inputs = [torch.randint(low=0, high=3, size=input_shape, dtype=dtype)]
        self.run_test(
            NonZero(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
