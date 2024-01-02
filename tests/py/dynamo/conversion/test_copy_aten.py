import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestCopyConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((3,), (3,), False),
            ((1, 10), (1, 10), False),
            ((2, 3, 4), (2, 3, 4), True),
            ((2, 3, 4, 5), (2, 3, 4, 5), True),
        ]
    )
    def test_copy_float(self, input_shape, src_shape, non_blocking):
        class Copy(nn.Module):
            def forward(self, input, src):
                return torch.ops.aten.copy.default(input, src, non_blocking)

        inputs = [torch.randn(input_shape), torch.randn(src_shape)]
        self.run_test(
            Copy(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
