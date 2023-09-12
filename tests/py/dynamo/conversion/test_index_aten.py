import operator

import torch
import torch.nn as nn
from harness import DispatchTestCase
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input


class TestIndexConverter(DispatchTestCase):
    def test_index_zero(self):
        class TestModule(nn.Module):
            def forward(self, x):
                index0 = torch.randint(0, 1, (1, 1))
                indices = [None, index0]
                out = torch.ops.aten.index.Tensor(x, indices)
                return out

        input = [torch.randn(2, 2)]
        self.run_test(
            TestModule(),
            input,
            expected_ops={torch.ops.aten.index.Tensor},
        )

    def test_index_zero_index_one(self):
        class TestModule(nn.Module):
            def forward(self, x):
                index0 = torch.randint(0, 1, (1, 1))
                indices = [None, index0, None]
                out = torch.ops.aten.index.Tensor(x, indices)
                return out

        input = [torch.randn(2, 2, 2)]
        self.run_test(
            TestModule(),
            input,
            expected_ops={torch.ops.aten.index.Tensor},
        )

    def test_index_zero_index_one_index_two(self):
        class TestModule(nn.Module):
            def forward(self, x):
                index0 = torch.randint(0, 1, (1, 1))
                index1 = torch.randint(0, 1, (1, 1))
                indices = [None, index0, index1]
                out = torch.ops.aten.index.Tensor(x, indices)
                return out

        input = [torch.randn(2, 2, 2)]
        self.run_test(
            TestModule(),
            input,
            expected_ops={torch.ops.aten.index.Tensor, operator.getitem},
        )

    def test_index_zero_index_one_SD(self):
        class TestModule(nn.Module):
            def forward(self, x):
                index0 = torch.tensor([0, 0, 1, 1])
                index1 = torch.tensor([0, 0, 1, 1])
                indices = [None, index0, index1, None]
                out = torch.ops.aten.index.Tensor(x, indices)
                return out

        input = [torch.randn(2, 4, 4, 2)]
        self.run_test(
            TestModule(),
            input,
            expected_ops={torch.ops.aten.index.Tensor, operator.getitem},
        )

    def test_index_zero_index_one_SD(self):
        class TestModule(nn.Module):
            def forward(self, x):
                index0 = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
                index1 = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])
                indices = [None, index0, index1, None]
                out = torch.ops.aten.index.Tensor(x, indices)
                return out

        input = [torch.randn(2, 1280, 8, 8)]
        self.run_test(
            TestModule(),
            input,
            expected_ops={torch.ops.aten.index.Tensor, operator.getitem},
        )


if __name__ == "__main__":
    run_tests()
