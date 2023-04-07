import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import DispatchTestCase, InputTensorSpec


class TestWhereConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d_condition_xshape_yshape", (x < 0), (2, 2), (2, 2)),
            ("2d_broadcast_condition_xshape_yshape", (x < 0), (2, 2), (2, 1)),
            ("3d_condition_xshape_yshape", (x > 0), (2, 2, 1), (2, 2, 1)),
            ("2d_3d_condition_xshape_yshape", (x < 0), (2, 2), (2, 2, 1)),
        ]
    )
    def test_(self, _, condition, x_size, y_size):
        class Where(nn.Module):
            def forward(self, x):
                return torch.where(x, dim)

        inputX = [torch.randn(*x_size)]
        inputOther = [torch.randn(*y_size)]
        expected_op = {}
        self.run_test(
            Where(),
            inputs,
            expected_ops=torch.ops.aten.where.self,
        )


# class TestWhereConverter(DispatchTestCase):
#     @parameterized.expand(
#         [
#             ("2d_dim", (1), (-1, 1), [((1, 1), (1, 1), (3, 1))]),
#             ("3d_one_dim", (1), (-1, 2, 1), [((1, 2, 1), (1, 2, 1), (3, 2, 1))]),
#             #("3d_two_dim", (0, 1), (-1, -1, 1), [((1, 3, 1, 1), (1, 3, 1, 1))]),
#         ]
#     )
#     def test_where(self, _, dim, init_size, shape_range):
#         class Squeeze(nn.Module):
#             def forward(self, x):
#                 return torch.squeeze(x, dim)

#         input_specs = [
#             InputTensorSpec(
#                 shape=init_size,
#                 dtype=torch.float32,
#                 shape_ranges=shape_range,
#             ),
#         ]
#         self.run_test_with_dynamic_shape(
#             Squeeze(),
#             input_specs,
#             expected_ops=torch.ops.aten.where.self,
#         )
