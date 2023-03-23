import unittest

import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import param, parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import DispatchTestCase, InputTensorSpec


class TestSelectConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("select_dim_index", 2, 1),
        ]
    )
    def test_select(self, _, dim, index):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return torch.select(input, dim, index)

        input = [torch.randn(1, 3, 32)]
        self.run_test(
            TestModule(),
            input,
            expected_ops={torch.ops.aten.select.int},
            test_explicit_precision=True,
        )

    # def test_select_with_dynamic_shape(self, _, dim_test, index_test):
    #     class TestModule(torch.nn.Module):
    #         def __init__(self, dim, index):
    #             super().__init__()
    #             self.dim = dim
    #             self.index = index
    #         def forward(self, input):
    #             return torch.select(input, self.dim, self.index)

    #     input_spec = [
    #         InputTensorSpec(
    #             shape=(-1, 3, 32),
    #             dtype=torch.float32,
    #             shape_ranges=[((1, 3, 3), (3, 3, 3), (32, 32, 32))],
    #         ),
    #     ]
    #     self.run_test_with_dynamic_shape(
    #         TestModule(dim_test, index_test), input_spec, expected_ops={torch.ops.aten.select}
    #     )


if __name__ == "__main__":
    run_tests()
