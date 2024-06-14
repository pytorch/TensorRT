import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestGatherValueConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (
                "gather_zero_dim_indexOne_constant_value",
                0,
                torch.tensor([[0, 1, 2, 0]]),
            ),
            (
                "gather_zero_dim_indexTwo_constant_value",
                0,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]),
            ),
            (
                "gather_one_dim_indexOne_constant_value",
                1,
                torch.tensor([[0, 1, 2, 0]]),
            ),
            (
                "gather_one_dim_indexTwo_costant_value",
                1,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]),
            ),
        ]
    )
    def test_gather_index_constant(self, _, dim, index):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.gather.default(input, dim, index)

        input = torch.zeros(3, 5, dtype=torch.int32)
        inputs = [input]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            ("gather_zero_dim_indexOne_value", 0, torch.tensor([[0, 1, 2, 0]])),
            (
                "gather_zero_dim_indexTwo_value",
                0,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]),
            ),
            ("gather_one_dim_indexOne_value", 1, torch.tensor([[0, 1, 2, 0]])),
            (
                "gather_one_dim_indexTwo_value",
                1,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]),
            ),
        ]
    )
    def test_gather_index_input(self, _, dim, index):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input, index):
                return torch.ops.aten.gather.default(input, dim, index)

        input = torch.zeros(3, 5, dtype=torch.int32)
        inputs = [input, index]
        self.run_test(TestModule(), inputs)
