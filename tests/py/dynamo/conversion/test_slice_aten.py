import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestSelectConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("select_dim_start_stop_step", 0, 0, 7, 2),
            ("select_dim_start_stop_step_offset", 1, 0, 7, 2),
            ("select_dim_start_stop_step_exact", 1, 0, 10, 2),
            ("select_dim_start_stop_step_negatives", -3, -2, -1, 1),
            ("select_dim_start_stop_step_max_int", 2, 0, 2**63 - 1, 1),
        ]
    )
    def test_slice(self, _, dim, start, stop, step):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                out = torch.ops.aten.slice.Tensor(input, dim, start, stop, step)
                return out

        input = [torch.randn(10, 10, 3, 1)]
        self.run_test(
            TestModule(),
            input,
        )


class TestSelectConverterDynamicShape(DispatchTestCase):
    @parameterized.expand(
        [
            ("select_dim_start_stop_step", 1, 0, 7, 2),
            ("select_dim_start_stop_step", 1, 0, 10, 2),
        ]
    )
    def test_slice(self, _, dim, start, stop, step):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                out = torch.ops.aten.slice.Tensor(input, dim, start, stop, step)
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
        )


if __name__ == "__main__":
    run_tests()
