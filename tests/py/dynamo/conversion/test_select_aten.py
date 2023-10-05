import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestSelectConverterOne(DispatchTestCase):
    @parameterized.expand(
        [
            ("select_dim_index", 1, 0),
        ]
    )
    def test_select(self, _, dim, index):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.select.int(input, dim, index)

        input = [torch.randn(1, 2)]
        self.run_test(
            TestModule(),
            input,
        )


class TestSelectConverterTwo(DispatchTestCase):
    @parameterized.expand(
        [
            ("select_dim_index", 1, 0),
        ]
    )
    def test_select(self, _, dim, index):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.select.int(input, dim, index)

        input = [torch.randn(4, 4, 4, 4)]
        self.run_test(
            TestModule(),
            input,
        )


class TestSelectConverterWithDynamicShape(DispatchTestCase):
    @parameterized.expand(
        [
            ("select_dim_index", 1, 0),
        ]
    )
    def test_select_with_dynamic_shape(self, _, dim, index):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.ops.aten.select.int(input, dim, index)

        input_spec = [
            Input(
                shape=(-1, 3, 3),
                dtype=torch.float32,
                shape_ranges=[((1, 3, 3), (3, 3, 3), (3, 3, 3))],
            ),
        ]
        self.run_test_with_dynamic_shape(TestModule(), input_spec)


if __name__ == "__main__":
    run_tests()
