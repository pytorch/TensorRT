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


class TestGatherInt64IndexConverter(DispatchTestCase):
    """Test cases for gather with int64 indices.
    TensorRT now supports int64 indices for gather operations.
    https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/classnvinfer1_1_1_i_gather_layer.html
    """

    @parameterized.expand(
        [
            (
                "gather_zero_dim_indexOne_int64",
                0,
                torch.tensor([[0, 1, 2, 0]], dtype=torch.int64),
            ),
            (
                "gather_zero_dim_indexTwo_int64",
                0,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]], dtype=torch.int64),
            ),
            (
                "gather_one_dim_indexOne_int64",
                1,
                torch.tensor([[0, 1, 2, 0]], dtype=torch.int64),
            ),
            (
                "gather_one_dim_indexTwo_int64",
                1,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]], dtype=torch.int64),
            ),
        ]
    )
    def test_gather_index_int64_constant(self, _, dim, index):
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
            (
                "gather_zero_dim_indexOne_int64_input",
                0,
                torch.tensor([[0, 1, 2, 0]], dtype=torch.int64),
            ),
            (
                "gather_zero_dim_indexTwo_int64_input",
                0,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]], dtype=torch.int64),
            ),
            (
                "gather_one_dim_indexOne_int64_input",
                1,
                torch.tensor([[0, 1, 2, 0]], dtype=torch.int64),
            ),
            (
                "gather_one_dim_indexTwo_int64_input",
                1,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]], dtype=torch.int64),
            ),
        ]
    )
    def test_gather_index_int64_input(self, _, dim, index):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input, index):
                return torch.ops.aten.gather.default(input, dim, index)

        input = torch.zeros(3, 5, dtype=torch.int32)
        inputs = [input, index]
        self.run_test(TestModule(), inputs)

    @parameterized.expand(
        [
            (
                "gather_float_input_int64_index",
                0,
                torch.tensor([[0, 1, 2, 0]], dtype=torch.int64),
            ),
            (
                "gather_float_input_int64_index_dim1",
                1,
                torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]], dtype=torch.int64),
            ),
        ]
    )
    def test_gather_float_input_int64_index(self, _, dim, index):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input, index):
                return torch.ops.aten.gather.default(input, dim, index)

        input = torch.randn(3, 5, dtype=torch.float32)
        inputs = [input, index]
        self.run_test(TestModule(), inputs)


if __name__ == "__main__":
    run_tests()
