import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import DispatchTestCase, InputTensorSpec


class TestWhereConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d_condition_xshape_yshape", (2, 2), (2, 2)),
            ("2d_broadcast_condition_xshape_yshape", (2, 2), (2, 1)),
            ("3d_condition_xshape_yshape", (2, 2, 1), (2, 2, 1)),
            ("2d_3d_condition_xshape_yshape", (2, 2), (1, 2, 2)),
        ]
    )
    def test_(self, _, x_size, y_size):
        class Where(nn.Module):
            def forward(self, condition, x, y):
                return torch.where(condition, x, y)

        inputX = torch.randn(*x_size)
        inputOther = torch.randn(*y_size)
        condition = inputX < 0
        self.run_test(
            Where(),
            (condition, inputX, inputOther),
            expected_ops={torch.ops.aten.where.self},
        )


# FIXME: How to specify condition for dynamic shape
# InputTensorSpec like case below where one input is dynamic another is not
# class TestWhereConverter(DispatchTestCase):
#     @parameterized.expand(
#         [
#             ("2d_dim", (-1, 2), [((1, 2), (2, 2), (2, 2))], (2,2))
#             #("3d_one_dim", (1), (-1, 2, 1), [((1, 2, 1), (1, 2, 1), (3, 2, 1))]),
#             #("3d_two_dim", (0, 1), (-1, -1, 1), [((1, 3, 1, 1), (1, 3, 1, 1))]),
#         ]
#     )
#     def test_where(self, _, x_size, x_size_range, y_size):
#         class Where(nn.Module):
#             def forward(self, condition, x, y):
#                 return torch.where(condition, x, y)
#         inputX = InputTensorSpec(
#                 shape=x_size,
#                 dtype=torch.float32,
#                 shape_ranges=x_size_range,
#                 )
#         inputOther = torch.randn(*y_size)
#         condition = (inputOther < 0)
#         input_specs = [
#             inputX, inputOther, condition
#         ]
#         self.run_test_with_dynamic_shape(
#             Where(),
#             input_specs,
#             expected_ops=torch.ops.aten.where.self,
#         )

# if __name__ == "__main__":
#     run_tests()
