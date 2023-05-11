import unittest

import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import DispatchTestCase, InputTensorSpec


class TestSelectConverterImplicitBatch(DispatchTestCase):
    @parameterized.expand(
        [
            ("select_dim_start_stop_step", 0, 0, 7, 2),
        ]
    )
    def test_slice(self, _, dim, start, stop, step):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                out = torch.ops.aten.slice.Tensor(input, dim, start, stop, step)
                return out

        input = [torch.randn(10, 2, 3, 1)]
        self.run_test(
            TestModule(),
            input,
            expected_ops={torch.ops.aten.slice.Tensor},
        )


class TestSelectConverterExplicitBatch(DispatchTestCase):
    @parameterized.expand(
        [
            ("select_dim_start_stop_step", 1, 0, 7, 2),
            ("select_dim_start_stop_step_exact", 1, 0, 10, 2),
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
            expected_ops={torch.ops.aten.slice.Tensor},
            test_explicit_precision=True,
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
            InputTensorSpec(
                shape=(1, 10, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 10, 1), (1, 10, 10), (1, 10, 10))],
            ),
        ]
        self.run_test_with_dynamic_shape(
            TestModule(),
            input_specs,
            expected_ops={torch.ops.aten.slice.Tensor},
        )


if __name__ == "__main__":
    run_tests()
