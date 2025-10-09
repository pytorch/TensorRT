import unittest

import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.utils import is_thor

from .harness import DispatchTestCase


@unittest.skipIf(
    is_thor(),
    "Skipped on Thor",
)
class TestSymSizeConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((3, 2, 4),),
        ]
    )
    def test_sym_size_batch(self, input_shape):
        class BatchDim(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sym_size.int(x, 0)

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            BatchDim(),
            inputs,
        )

    @parameterized.expand(
        [
            ((3, 2, 4),),
        ]
    )
    def test_sym_size_non_batch(self, input_shape):
        class NonBatchDim(nn.Module):
            def forward(self, x):
                return torch.ops.aten.sym_size.int(x, 1)

        inputs = [torch.randn(*input_shape)]
        self.run_test(
            NonBatchDim(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
