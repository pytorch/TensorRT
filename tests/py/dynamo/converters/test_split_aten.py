import torch
from harness import DispatchTestCase
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo.conversion import UnsupportedOperatorException


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
                out = torch.split(input, split_size_or_tensor)
                return out

        input = [torch.randn(10).reshape(5, 2)]
        self.run_test(
            TestModule(),
            input,
            expected_ops={torch.ops.aten.split.Tensor},
            disable_passes=True,
        )

    @parameterized.expand(
        [
            ("split_size_or_sections_list_no_dim_list", [1, 4]),
            ("split_size_or_sections_list_no_dim_not_full_list", [1, 3]),
        ]
    )
    def test_split_list(self, _, split_size_or_tensor):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                out = torch.split(input, split_size_or_tensor)
                return out

        input = [torch.randn(10).reshape(5, 2)]
        self.run_test(
            TestModule(),
            input,
            expected_ops={torch.ops.aten.split_with_sizes.default},
            disable_passes=True,
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
                out = torch.split(input, split_size_or_tensor, dim)
                return out

        input = [torch.randn(10).reshape(5, 2)]
        self.run_test(
            TestModule(),
            input,
            expected_ops={torch.ops.aten.split.Tensor},
            disable_passes=True,
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
                out = torch.split(input, split_size_or_tensor, dim)
                return out

        input = [torch.randn(10).reshape(5, 2)]
        self.run_test(
            TestModule(),
            input,
            expected_ops={torch.ops.aten.split_with_sizes.default},
            disable_passes=True,
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
                out = torch.split(input, split_size_or_tensor, dim)
                return out

        input = [torch.randn(15).reshape(5, 3)]
        with self.assertRaises(RuntimeError):
            self.run_test(
                TestModule(),
                input,
                expected_ops={torch.ops.aten.split_with_sizes.default},
                disable_passes=True,
            )

    @parameterized.expand(
        [
            ("select_split_size_or_sections_dim_dynamic_shape", 2, 1),
        ]
    )
    def test_split_dynamic(self, _, split_size_or_tensor, dim):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                out = torch.split(input, split_size_or_tensor, dim)
                return out

        input_specs = [
            Input(
                shape=(1, 10, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 10, 1), (1, 10, 10), (1, 10, 10))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(),
            input_specs,
            expected_ops={torch.ops.aten.split.Tensor},
            disable_passes=True,
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
                expected_ops={torch.ops.aten.split.Tensor},
                disable_passes=True,
            )


if __name__ == "__main__":
    run_tests()
