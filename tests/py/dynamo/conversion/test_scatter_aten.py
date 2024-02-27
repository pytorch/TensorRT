import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestScatterValueConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("scatter_zero_dim_indexOne_value", 0, [[0, 1, 2, 0]], 1),
            ("scatter_zero_dim_indexTwo_value", 0, [[0, 1, 2, 0], [1, 2, 1, 1]], 1),
            ("scatter_one_dim_indexOne_value", 1, [[0, 1, 2, 0]], 1),
            ("scatter_one_dim_indexTwo_value", 1, [[0, 1, 2, 0], [1, 2, 1, 1]], 1),
        ]
    )
    def test_scatter(self, _, dim, index, value):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input, src):
                return torch.ops.aten.scatter.value(input, dim, index, value)

        input = [torch.zeros(3, 5, dtype=torch.int32)]
        self.run_test(
            TestModule(),
            input,
        )


class TestScatterSrcConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("scatter_zero_dim_indexOne", 0, [[0, 1, 2, 0]]),
            ("scatter_zero_dim_indexTwo", 0, [[0, 1, 2, 0], [1, 2, 1, 1]]),
            ("scatter_one_dim_indexOne", 1, [[0, 1, 2, 0]]),
            ("scatter_one_dim_indexTwo", 1, [[0, 1, 2, 0], [1, 2, 1, 1]]),
        ]
    )
    def test_scatter(self, _, dim, index):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input, src):
                return torch.ops.aten.scatter.src(input, dim, index, src)

        src = [torch.arange(1, 11).reshape((2, 5))]
        input = torch.zeros(3, 5, dtype=src.dtype)
        inputs = [input, src]
        self.run_test(
            TestModule(),
            inputs,
        )
