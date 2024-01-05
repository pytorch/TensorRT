import operator

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestGatherConverter(DispatchTestCase):
    def test_gather_zero_two_dim(self):
        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, indices):
                # index0 = torch.randint(0, 1, (1, 1))
                out = torch.ops.aten.gather.default(x, 0, indices)
                return out

        # index0 = torch.randint(0, 1, (1, 1), dtype=torch.int32)
        index0 = torch.randint(0, 1, (1, 1))
        input = [torch.randn(2, 2), index0]
        self.run_test(
            TestModule(),
            input,
        )


if __name__ == "__main__":
    run_tests()
