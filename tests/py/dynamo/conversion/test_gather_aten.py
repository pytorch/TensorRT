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

    @parameterized.expand(
        [
            ("positive_dim", 1),
            ("negative_dim", -1),
        ]
    )
    def test_gather_dynamic_shape(self, _, dim):
        """The registry only consults supports_dynamic_shapes when a node carries symbolic
        shape metadata, which the legacy tracer does not produce, so this needs the dynamo
        tracer or it passes either way."""

        class TestModule(torch.nn.Module):
            def forward(self, input, index):
                return torch.ops.aten.gather.default(input, dim, index)

        input_specs = [
            Input(
                min_shape=(1, 5),
                opt_shape=(3, 5),
                max_shape=(6, 5),
                dtype=torch.float32,
            ),
            Input(
                min_shape=(1, 4),
                opt_shape=(3, 4),
                max_shape=(6, 4),
                dtype=torch.int64,
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(),
            input_specs,
            use_dynamo_tracer=True,
        )

    def test_gather_dynamic_gathered_axis(self):
        """The gathered axis itself is dynamic here, so index validity depends on the shape
        the engine is given at run time rather than on the shape it was built at."""

        class TestModule(torch.nn.Module):
            def forward(self, input, index):
                return torch.ops.aten.gather.default(input, 1, index)

        input_specs = [
            Input(
                min_shape=(3, 1),
                opt_shape=(3, 4),
                max_shape=(3, 6),
                dtype=torch.float32,
            ),
            Input(
                min_shape=(3, 1),
                opt_shape=(3, 4),
                max_shape=(3, 6),
                dtype=torch.int64,
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(),
            input_specs,
            use_dynamo_tracer=True,
        )


if __name__ == "__main__":
    run_tests()
