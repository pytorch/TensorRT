# import torch
# import torch.nn as nn
# from parameterized import parameterized
# from torch.testing._internal.common_utils import run_tests

# from .harness import DispatchTestCase


# class TestMaxConverter(DispatchTestCase):
#     @parameterized.expand(
#         [
#             ((3, 2, 4),),
#             ((2, 3, 4, 5),),
#             ((2, 3, 4, 5),),
#             ((6, 7, 5, 4, 5),),
#         ]
#     )
#     def test_max_dim_int_default(self, input_shape):
#         class Max(nn.Module):
#             def forward(self, x):
#                 return torch.max(x)

#         inputs = [torch.randn(*input_shape)]
#         self.run_test(
#             Max(),
#             inputs,
#             expected_ops={torch.ops.aten.max.default},
#         )

#     @parameterized.expand(
#         [
#             ((3, 2, 4), 1, True),
#             ((2, 3, 4, 5), 3, True),
#             ((6, 7, 5, 4, 5), 4, False),
#             ((1, 5, 2, 1), -3, False),
#             ((1, 5, 2, 3), -2, True),
#         ]
#     )
#     def test_max_dim_int(self, input_shape, dim, keep_dims):
#         class Max(nn.Module):
#             def forward(self, x):
#                 return torch.max(x, dim=dim, keepdim=keep_dims)

#         inputs = [torch.randn(*input_shape)]
#         self.run_test(
#             Max(),
#             inputs,
#             expected_ops={torch.ops.aten.max.dim},
#         )


#     @parameterized.expand(
#         [
#             ((3, 2, 4), 1, True, torch.int, 0, 5),
#             ((2, 3, 4, 5), 2, False, torch.int32, -5, 0),
#             ((6, 7, 5, 4, 5), 4, False, torch.int32, -5, 5),
#         ]
#     )
#     def test_max_dim_int_int(self, input_shape, dim, keep_dims, dtype, low, high):
#         class Max(nn.Module):
#             def forward(self, x):
#                 return torch.max(x, dim=dim, keepdim=keep_dims)

#         inputs = [torch.randint(low, high, input_shape, dtype=dtype)]
#         self.run_test(
#             Max(),
#             inputs,
#             expected_ops={torch.ops.aten.max.dim},
#             check_dtype=False,
#         )


# if __name__ == "__main__":
#     run_tests()
