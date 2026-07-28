import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo.conversion import UnsupportedOperatorException

from .harness import DispatchTestCase


# FIXME: check about implicit and explicit batch
class TestSplitConverterNoDim(DispatchTestCase):
    @parameterized.expand(
        [
            ("split_size_or_sections_no_dim", 2),
        ]
    )
    def test_split(self, _, split_size_or_tensor):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                out = torch.ops.aten.split.Tensor(input, split_size_or_tensor)
                return out

        input = [torch.randn(10).reshape(5, 2)]
        self.run_test(
            TestModule(),
            input,
        )

    @parameterized.expand(
        [
            ("split_size_or_sections_list_no_dim_list", [1, 4]),
        ]
    )
    def test_split_list(self, _, split_size_or_tensor):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                out = torch.ops.aten.split_with_sizes.default(
                    input, split_size_or_tensor
                )
                return out

        input = [torch.randn(10).reshape(5, 2)]
        self.run_test(
            TestModule(),
            input,
        )

    @parameterized.expand(
        [
            ("split_size_or_sections_dims", 2, 1),
        ]
    )
    def test_split(self, _, split_size_or_tensor, dim):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                out = torch.ops.aten.split.Tensor(input, split_size_or_tensor, dim)
                return out

        input = [torch.randn(10).reshape(5, 2)]
        self.run_test(
            TestModule(),
            input,
        )

    @parameterized.expand(
        [
            ("split_size_or_sections_list_dims", [1, 1], 1),
        ]
    )
    def test_split_dim_list(self, _, split_size_or_tensor, dim):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                out = torch.ops.aten.split_with_sizes.default(
                    input, split_size_or_tensor, dim
                )
                return out

        input = [torch.randn(10).reshape(5, 2)]
        self.run_test(
            TestModule(),
            input,
        )

    @parameterized.expand(
        [
            ("split_size_or_sections_list_dims_not_full_list", [1, 1], 1),
        ]
    )
    def test_split_dim_list(self, _, split_size_or_tensor, dim):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                out = torch.ops.aten.split_with_sizes.default(
                    input, split_size_or_tensor, dim
                )
                return out

        input = [torch.randn(15).reshape(5, 3)]
        with self.assertRaises(RuntimeError):
            self.run_test(
                TestModule(),
                input,
            )

    @parameterized.expand(
        [
            ("select_split_size_or_sections_dim_dynamic_shape", 2, 1),
            ("select_split_size_or_sections_non_divisible_dim_dynamic_shape", 3, 1),
        ]
    )
    def test_split_dynamic(self, _, split_size_or_tensor, dim):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                out = torch.ops.aten.split.Tensor(input, split_size_or_tensor, dim)
                return out

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=[1, 10, 1],
                opt_shape=[1, 10, 10],
                max_shape=[1, 10, 10],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(),
            input_specs,
        )

    @parameterized.expand(
        [
            ("select_split_size_or_sections_dim_dynamic_shape_on_first_axis", 2, 1),
        ]
    )
    def test_split_dynamic_first_axis_dynamic(self, _, split_size_or_tensor, dim):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                out = torch.ops.aten.split.Tensor(input, split_size_or_tensor, dim)
                return out

        input_specs = [
            Input(
                dtype=torch.float32,
                min_shape=[1, 10, 10],
                opt_shape=[3, 10, 10],
                max_shape=[5, 10, 10],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(),
            input_specs,
        )

    @parameterized.expand(
        [
            ("select_chunk_dim", 6, 0),
        ]
    )
    def test_split_dynamic(self, _, chunk, dim):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                out = torch.ops.aten.chunk(input, chunk, dim)
                return out

        input = [torch.randn(11)]
        with self.assertRaises(UnsupportedOperatorException):
            self.run_test(
                TestModule(),
                input,
            )


if __name__ == "__main__":
    run_tests()
