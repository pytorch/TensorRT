import operator

import torch
import torch.nn as nn
from harness import DispatchTestCase
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input


class TestIndexConverter(DispatchTestCase):
    def test_index_zero_two_dim(self):
        class TestModule(nn.Module):
            def __init__(self):
                self.index0 = torch.randint(0, 1, (1, 1))
                super().__init__()

            def forward(self, x):
                index0 = torch.randint(0, 1, (1, 1))
                indices = [None, self.index0]
                out = torch.ops.aten.gather.default(x, 0, indices)
                return out

        input = [torch.randn(2, 2)]
        self.run_test(
            TestModule(),
            input,
        )


if __name__ == "__main__":
    run_tests()
