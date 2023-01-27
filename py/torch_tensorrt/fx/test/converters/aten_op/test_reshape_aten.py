import unittest

import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import DispatchTestCase, InputTensorSpec


class TestReshapeConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((1, 20),),
            ((1, 10, -1),),
        ]
    )
    @unittest.skip("Need support")
    def test_reshape(self, target_shape):
        class TestModule(torch.nn.Module):
            def __init__(self, target_shape):
                super().__init__()
                self.target_shape = target_shape

            def forward(self, x):
                return torch.reshape(x, self.target_shape)

        inputs = [torch.randn(1, 2, 10)]
        self.run_test(
            TestModule(target_shape),
            inputs,
            expected_ops={torch.ops.aten.view.default},
        )

    ## TODO: proxytensor tracer does not support output size containing -1. If dim=0 is set to -1 for dynamic batch,
    ## then it is becomes fixed acoording to the input. For ex. input (-1, 2, 3), output size (-1, 6), then
    ## proxytensor tracer output is (32, 6) if sample input is (32, 2, 3). But fx tracer could keep the output size as (-1, 6)
    # @parameterized.expand(
    #     [
    #         ((-1, 2),),
    #         ((1, 2, -1),),
    #     ]
    # )
    # def test_reshape_with_dynamic_shape(self, target_shape):
    #     class TestModule(torch.nn.Module):
    #         def __init__(self, target_shape):
    #             super().__init__()
    #             self.target_shape = target_shape

    #         def forward(self, x):
    #             return torch.reshape(x, self.target_shape)

    #     input_specs = [
    #         InputTensorSpec(
    #             shape=(-1, -1, -1),
    #             dtype=torch.float32,
    #             shape_ranges=[((1, 1, 1), (1, 2, 3), (3, 3, 3))],
    #         ),
    #     ]
    #     self.run_test_with_dynamic_shape(
    #         TestModule(target_shape), input_specs, expected_ops={torch.ops.aten._reshape_alias.default}
    #     )

    # def test_reshape_with_dynamic_shape_size(self):
    #     class TestModule(torch.nn.Module):
    #         def forward(self, x, y):
    #             shape_y = y.shape
    #             t = shape_y[1]
    #             return torch.reshape(x, [-1, t, 3])

    #     input_specs = [
    #         InputTensorSpec(
    #             shape=(-1, 5, 6),
    #             dtype=torch.float32,
    #             shape_ranges=[((1, 5, 6), (2, 5, 6), (3, 5, 6))],
    #         ),
    #         InputTensorSpec(
    #             shape=(-1, 5),
    #             dtype=torch.float32,
    #             shape_ranges=[((1, 5), (1, 5), (3, 5))],
    #         ),
    #     ]

    #     self.run_test_with_dynamic_shape(
    #         TestModule(), input_specs, expected_ops={acc_ops.reshape}
    #     )


if __name__ == "__main__":
    run_tests()
