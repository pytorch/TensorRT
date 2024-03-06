import torch
from harness import DispatchTestCase
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input


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
        self.run_test(
            TestModule(),
            inputs,
        )

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
        self.run_test(
            TestModule(),
            inputs,
        )


class TestScatterSrcConverter(DispatchTestCase):
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
            # These are special cases where in the harness.py code might need to be changed to input cuda_inputs
            # In that case below two test cases would also require index and src to be on cuda
            # ("scatter_one_dim_indexOne_constant_src", 1, torch.tensor([[0, 1, 2, 0]]), torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)),
            # ("scatter_one_dim_indexTwo_constant_src", 1, torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]]), torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32)),
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
        self.run_test(
            TestModule(),
            inputs,
        )

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
        self.run_test(
            TestModule(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
