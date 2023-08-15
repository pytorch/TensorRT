import torch
from harness import DispatchTestCase
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input


# FIXME: check about implicit and explicit batch
class TestSplitConverterNoDim(DispatchTestCase):
    @parameterized.expand(
        [
            ("split_size_or_sections_no_dim", 2),
            ("split_size_or_sections_list_no_dim", [1, 4]),
            ("split_size_or_sections_list_no_dim_not_full_split", [1, 3]),
        ]
    )
    def test_split(self, _, split_size_or_tensor):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                out = torch.split(input, split_size_or_tensor)
                return out

        input = torch.arange(10).reshape(5, 2)
        self.run_test(
            TestModule(),
            input,
            expected_ops={torch.ops.aten.split.default},
        )


class TestSplitConverterWithDim(DispatchTestCase):
    @parameterized.expand(
        [
            ("split_size_or_sections_dim", 2, 1),
            ("split_size_or_sections_list_dim", [1, 4], 1),
            ("split_size_or_sections_list_dim_not_full_split", [1, 3], 1),
        ]
    )
    def test_split(self, _, split_size_or_tensor, dim):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                out = torch.split(split_size_or_tensor, dim)
                return out

        input = torch.arange(10).reshape(2, 5)
        self.run_test(
            TestModule(),
            input,
            expected_ops={torch.ops.aten.split.default},
        )


class TestSplitConverterDynamicShape(DispatchTestCase):
    @parameterized.expand(
        [
            ("select_split_size_or_sections_dim", 2, 1),
            ("select_split_size_or_sections_list_dim", [1, 4], 1),
        ]
    )
    def test_split(self, _, split_size_or_tensor, dim):
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
            expected_ops={torch.ops.aten.split.default},
        )


class TestSplitSymIntConverterImplicitBatch(DispatchTestCase):
    @parameterized.expand(
        [
            ("select_chunk_dim", 6, 0),
        ]
    )
    def test_chunk(self, _, chunk, dim):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                out = torch.ops.aten.chunk(input, chunk, dim)
                return out

        input = [torch.randn(11)]
        self.run_test(
            TestModule(),
            input,
            expected_ops={torch.ops.aten.split.default},
        )


if __name__ == "__main__":
    run_tests()
