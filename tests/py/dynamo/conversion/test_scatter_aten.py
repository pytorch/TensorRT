import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestScatterValueConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (
                "scatter_zero_dim_indexOne_constant_value",
                0,
                torch.tensor([[0, 1, 2, 0]]),
                1,
            ),
            (
                "scatter_zero_dim_indexTwo_constant_value",
                0,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]),
                1,
            ),
            (
                "scatter_one_dim_indexOne_constant_value",
                1,
                torch.tensor([[0, 1, 2, 0]]),
                1,
            ),
            (
                "scatter_one_dim_indexTwo_costant_value",
                1,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]),
                1,
            ),
        ]
    )
    def test_scatter_index_constant(self, _, dim, index, value):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.scatter.value(input, dim, index, value)

        input = torch.zeros(3, 5, dtype=torch.int32)
        inputs = [input]
        self.run_test(TestModule(), inputs, int32_reqd=True)

    @parameterized.expand(
        [
            ("scatter_zero_dim_indexOne_value", 0, torch.tensor([[0, 1, 2, 0]]), 1),
            (
                "scatter_zero_dim_indexTwo_value",
                0,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]),
                1,
            ),
            ("scatter_one_dim_indexOne_value", 1, torch.tensor([[0, 1, 2, 0]]), 1),
            (
                "scatter_one_dim_indexTwo_value",
                1,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]),
                1,
            ),
        ]
    )
    def test_scatter_index_input(self, _, dim, index, value):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input, index):
                return torch.ops.aten.scatter.value(input, dim, index, value)

        input = torch.zeros(3, 5, dtype=torch.int32)
        inputs = [input, index]
        self.run_test(TestModule(), inputs, int32_reqd=True)


class TestScatterSrcConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (
                "scatter_zero_dim_indexOne_src",
                0,
                torch.tensor([[0, 1, 2, 0]]),
                torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
            ),
            (
                "scatter_zero_dim_indexTwo_src",
                0,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]),
                torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32),
            ),
            (
                "scatter_one_dim_indexOne_src",
                1,
                torch.tensor([[0, 1, 2, 0]]),
                torch.tensor([[1, 2, 3, 1]], dtype=torch.int32),
            ),
            (
                "scatter_one_dim_indexTwo_src",
                1,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]),
                torch.tensor([[1, 2, 3, 1], [5, 6, 5, 5]], dtype=torch.int32),
            ),
            (
                "scatter_one_dim_indexOne_constant_src",
                1,
                torch.tensor([[0, 1, 2, 0]]),
                torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
            ),
            (
                "scatter_one_dim_indexTwo_constant_src",
                1,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]),
                torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32),
            ),
        ]
    )
    def test_scatter_index_constant(self, _, dim, index, src):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.scatter.src(input, dim, index, src)

        input = torch.zeros(3, 5, dtype=torch.int32)
        inputs = [input]
        scatter = TestModule()
        self.run_test(TestModule(), inputs, int32_reqd=True)

    @parameterized.expand(
        [
            (
                "scatter_zero_dim_indexOne_constant_src",
                0,
                torch.tensor([[0, 1, 2, 0]]),
                torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
            ),
            (
                "scatter_zero_dim_indexTwo_constant_src",
                0,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]),
                torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32),
            ),
            (
                "scatter_one_dim_indexOne_constant_src",
                1,
                torch.tensor([[0, 1, 2, 0]]),
                torch.tensor([[1, 2, 3, 1]], dtype=torch.int32),
            ),
            (
                "scatter_one_dim_indexTwo_constant_src",
                1,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]),
                torch.tensor([[1, 2, 3, 1], [5, 6, 5, 5]], dtype=torch.int32),
            ),
        ]
    )
    def test_scatter_index_input(self, _, dim, index, src):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input, index):
                return torch.ops.aten.scatter.src(input, dim, index, src)

        input = torch.zeros(3, 5, dtype=torch.int32)
        inputs = [input, index]
        self.run_test(TestModule(), inputs, int32_reqd=True)


class TestScatterInt64IndexConverter(DispatchTestCase):
    """Test cases for scatter with int64 indices.
    TensorRT now supports int64 indices for scatter operations.
    """

    @parameterized.expand(
        [
            (
                "scatter_zero_dim_int64_index_value",
                0,
                torch.tensor([[0, 1, 2, 0]], dtype=torch.int64),
                1,
            ),
            (
                "scatter_one_dim_int64_index_value",
                1,
                torch.tensor([[0, 1, 2, 0]], dtype=torch.int64),
                1,
            ),
            (
                "scatter_zero_dim_int64_indexTwo_value",
                0,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]], dtype=torch.int64),
                1,
            ),
        ]
    )
    def test_scatter_int64_index_constant(self, _, dim, index, value):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.scatter.value(input, dim, index, value)

        input = torch.zeros(3, 5, dtype=torch.int32)
        inputs = [input]
        self.run_test(TestModule(), inputs, int32_reqd=True)

    @parameterized.expand(
        [
            (
                "scatter_zero_dim_int64_index_input",
                0,
                torch.tensor([[0, 1, 2, 0]], dtype=torch.int64),
                1,
            ),
            (
                "scatter_one_dim_int64_index_input",
                1,
                torch.tensor([[0, 1, 2, 0]], dtype=torch.int64),
                1,
            ),
        ]
    )
    def test_scatter_int64_index_input(self, _, dim, index, value):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input, index):
                return torch.ops.aten.scatter.value(input, dim, index, value)

        input = torch.zeros(3, 5, dtype=torch.int32)
        inputs = [input, index]
        self.run_test(TestModule(), inputs, int32_reqd=True)

    @parameterized.expand(
        [
            (
                "scatter_src_zero_dim_int64_index",
                0,
                torch.tensor([[0, 1, 2, 0]], dtype=torch.int64),
                torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
            ),
            (
                "scatter_src_one_dim_int64_index",
                1,
                torch.tensor([[0, 1, 2, 0]], dtype=torch.int64),
                torch.tensor([[1, 2, 3, 1]], dtype=torch.int32),
            ),
        ]
    )
    def test_scatter_src_int64_index(self, _, dim, index, src):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input, index):
                return torch.ops.aten.scatter.src(input, dim, index, src)

        input = torch.zeros(3, 5, dtype=torch.int32)
        inputs = [input, index]
        self.run_test(TestModule(), inputs, int32_reqd=True)


if __name__ == "__main__":
    run_tests()
