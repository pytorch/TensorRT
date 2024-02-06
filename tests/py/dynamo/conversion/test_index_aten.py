import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase

from .harness import DispatchTestCase


class TestIndexConverter(DispatchTestCase):
    def test_index_zero_two_dim(self):
        class TestModule(nn.Module):
            def __init__(self):
                self.index0 = torch.randint(0, 1, (1, 1))
                super().__init__()

            def forward(self, x):
                indices = [None, self.index0]
                out = torch.ops.aten.index.Tensor(x, indices)
                return out

        input = [torch.randn(2, 2)]
        self.run_test(
            TestModule(),
            input,
        )

    def test_index_zero_two_dim_ITensor(self):
        class TestModule(nn.Module):
            def forward(self, x, index0):
                indices = [None, index0]
                out = torch.ops.aten.index.Tensor(x, indices)
                return out

        input = torch.randn(2, 2)
        index0 = torch.randint(0, 1, (1, 1))
        index0 = index0.to(torch.int32)
        self.run_test(
            TestModule(),
            [input, index0],
        )

    def test_index_zero_index_three_dim(self):
        class TestModule(nn.Module):
            def __init__(self):
                self.index0 = torch.randint(0, 1, (1, 1))
                super().__init__()

            def forward(self, x):
                indices = [None, self.index0, None]
                out = torch.ops.aten.index.Tensor(x, indices)
                return out

        input = [torch.randn(2, 2, 2)]
        self.run_test(
            TestModule(),
            input,
        )

    def test_index_zero_index_three_dim_ITensor(self):
        class TestModule(nn.Module):
            def forward(self, x, index0):
                indices = [None, index0, None]
                out = torch.ops.aten.index.Tensor(x, indices)
                return out

        input = torch.randn(2, 2, 2)
        index0 = torch.randint(0, 1, (1, 1))
        index0 = index0.to(torch.int32)
        self.run_test(TestModule(), [input, index0])

    def test_index_zero_index_one_index_two_three_dim(self):
        class TestModule(nn.Module):
            def __init__(self):
                self.index0 = torch.randint(0, 1, (1, 1))
                self.index1 = torch.randint(0, 1, (1, 1))
                super().__init__()

            def forward(self, x):
                indices = [None, self.index0, self.index1]
                out = torch.ops.aten.index.Tensor(x, indices)
                return out

        input = [torch.randn(2, 2, 2)]
        self.run_test(
            TestModule(),
            input,
        )

    def test_index_zero_index_one_four_dim(self):
        class TestModule(nn.Module):
            def __init__(self):
                self.index0 = torch.tensor([0, 0, 1, 1])
                self.index1 = torch.tensor([0, 0, 1, 1])
                super().__init__()

            def forward(self, x):
                indices = [None, self.index0, self.index1, None]
                out = torch.ops.aten.index.Tensor(x, indices)
                return out

        input = [torch.randn(2, 4, 4, 2)]
        self.run_test(
            TestModule(),
            input,
        )

    def test_index_zero_index_one_four_dim_SD(self):
        class TestModule(nn.Module):
            def __init__(self):
                self.index0 = torch.tensor(
                    [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
                )
                self.index1 = torch.tensor(
                    [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
                )
                super().__init__()

            def forward(self, x):
                indices = [None, self.index0, self.index1, None]
                out = torch.ops.aten.index.Tensor(x, indices)
                return out

        input = [torch.randn(2, 1280, 8, 8)]
        self.run_test(
            TestModule(),
            input,
        )

    def test_index_one_SD_unsqueeze_four_dim(self):
        class TestModule(nn.Module):
            def __init__(self):
                self.index0 = torch.tensor(
                    [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
                )
                self.index1 = self.index0.unsqueeze(0).T.long()
                super().__init__()

            def forward(self, x):
                indices = [None, None, self.index1, self.index1]
                out = torch.ops.aten.index.Tensor(x, indices)
                return out

        input = [torch.randn(2, 1280, 8, 8)]
        self.run_test(
            TestModule(),
            input,
        )

    def test_index_zero_index_one_index_two_SD_unsqueeze_four_dim_broadcast(self):
        class TestModule(nn.Module):
            def __init__(self):
                self.index0 = torch.tensor(
                    [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]
                )
                self.index1 = self.index0.unsqueeze(0).T.long()
                super().__init__()

            def forward(self, x):
                indices = [None, None, self.index0, self.index1]
                out = torch.ops.aten.index.Tensor(x, indices)
                return out

        input = [torch.randn(2, 1280, 8, 8)]
        self.run_test(
            TestModule(),
            input,
        )

    def test_index_zero_index_one_index_four_dim_non_continuous(self):
        class TestModule(nn.Module):
            def __init__(self):
                self.index0 = torch.tensor([0, 0, 1, 1])
                self.index1 = torch.tensor([0, 0, 1, 1])
                super().__init__()

            def forward(self, x):
                indices = [None, self.index0, None, self.index1]
                out = torch.ops.aten.index.Tensor(x, indices)
                return out

        input = [torch.randn(2, 4, 4, 2)]
        self.run_test(
            TestModule(),
            input,
        )


if __name__ == "__main__":
    run_tests()
