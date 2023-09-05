import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestIndexConverter(DispatchTestCase):
    def test_index(self):
        class TestModule(nn.Module):
            def forward(self, x):
                input = torch.randn(2, 1280, 8, 8)
                index0 = torch.randint(0, 16, (1, 16))
                index1 = torch.randint(0, 16, (1, 16))
                out = torch.ops.aten.index(None, None, index0, index1)

        inputs = [torch.randn(1, 10)]
        self.run_test(TestModule(), inputs, expected_ops={torch.ops.aten.index.Tensor})
