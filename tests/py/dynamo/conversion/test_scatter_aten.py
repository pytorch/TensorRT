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


class TestScatterValueDynamicShapeConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("dim0_float", 0, 1, torch.float),
            ("dim1_float", 1, 2.5, torch.float),
            ("dim0_half", 0, 1, torch.half),
            ("dim1_half", 1, 2.5, torch.half),
        ]
    )
    def test_scatter_value_dynamic_input(self, _, dim, value, dtype):
        """Test scatter.value with dynamic input shape but no dynamic src or index"""

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.index = torch.tensor([[0, 1, 2, 0], [1, 2, 1, 1]])

            def forward(self, input):
                return torch.ops.aten.scatter.value(input, dim, self.index, value)

        input_specs = [
            Input(
                min_shape=(2, 5),
                opt_shape=(3, 5),
                max_shape=(4, 5),
                dtype=dtype,
            ),
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)

    @parameterized.expand(
        [
            # (name, dim, value, dtype, input_dynamic)
            ("dim0_float_static_input", 0, 1, torch.float, False),
            ("dim1_float_static_input", 1, 3.0, torch.float, False),
            ("dim0_half_static_input", 0, 1, torch.half, False),
            ("dim1_half_static_input", 1, 3.0, torch.half, False),
            ("dim1_float_dynamic_input", 1, 5.0, torch.float, True),
            ("dim1_half_dynamic_input", 1, 5.0, torch.half, True),
        ]
    )
    def test_scatter_value_dynamic_index(self, _, dim, value, dtype, input_dynamic):
        """Test scatter.value with dynamic index shape (and optionally dynamic input)"""

        class TestModule(torch.nn.Module):
            def forward(self, input, index):
                return torch.ops.aten.scatter.value(input, dim, index, value)

        if input_dynamic:
            input_specs = [
                Input(
                    min_shape=(2, 5),
                    opt_shape=(3, 5),
                    max_shape=(4, 5),
                    dtype=dtype,
                ),
                Input(
                    min_shape=(2, 3),
                    opt_shape=(3, 3),
                    max_shape=(4, 3),
                    dtype=torch.int64,
                ),
            ]
        else:
            input_specs = [
                Input(
                    min_shape=(3, 5),
                    opt_shape=(3, 5),
                    max_shape=(3, 5),
                    dtype=dtype,
                ),
                Input(
                    min_shape=(1, 4),
                    opt_shape=(2, 4),
                    max_shape=(3, 4),
                    dtype=torch.int64,
                ),
            ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)


class TestScatterSrcDynamicShapeConverter(DispatchTestCase):
    @parameterized.expand(
        [
            # (name, dim, dtype, input_dynamic)
            ("dim1_float_static_input", 1, torch.float, False),
            ("dim1_half_static_input", 1, torch.half, False),
            ("dim1_float_dynamic_input", 1, torch.float, True),
            ("dim1_half_dynamic_input", 1, torch.half, True),
        ]
    )
    def test_scatter_src_dynamic_index_and_src(self, _, dim, dtype, input_dynamic):
        """Test scatter.src with dynamic index and src shapes (and optionally dynamic input)"""

        class TestModule(torch.nn.Module):
            def forward(self, input, index, src):
                return torch.ops.aten.scatter.src(input, dim, index, src)

        if input_dynamic:
            input_specs = [
                Input(
                    min_shape=(2, 5),
                    opt_shape=(3, 5),
                    max_shape=(4, 5),
                    dtype=dtype,
                ),
                Input(
                    min_shape=(2, 4),
                    opt_shape=(3, 4),
                    max_shape=(4, 4),
                    dtype=torch.int64,
                ),
                Input(
                    min_shape=(2, 4),
                    opt_shape=(3, 4),
                    max_shape=(4, 4),
                    dtype=dtype,
                ),
            ]
        else:
            input_specs = [
                Input(
                    min_shape=(3, 5),
                    opt_shape=(3, 5),
                    max_shape=(3, 5),
                    dtype=dtype,
                ),
                Input(
                    min_shape=(1, 4),
                    opt_shape=(2, 4),
                    max_shape=(3, 4),
                    dtype=torch.int64,
                ),
                Input(
                    min_shape=(1, 4),
                    opt_shape=(2, 4),
                    max_shape=(3, 4),
                    dtype=dtype,
                ),
            ]
        self.run_test_with_dynamic_shape(TestModule(), input_specs)


if __name__ == "__main__":
    run_tests()
