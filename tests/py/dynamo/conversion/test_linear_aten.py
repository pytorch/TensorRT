import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestLinearConverter(DispatchTestCase):
    @parameterized.expand(
        [
            (
                "default",
                [1, 512],
                True,
            ),
            (
                "matrix",
                [5, 512],
                True,
            ),
            (
                "no_bias",
                [1, 512],
                False,
            ),
            (
                "multi_dim_matrix",
                [4, 5, 512],
                True,
            ),
            (
                "multi_dim_matrix",
                [4, 5, 512],
                False,
            ),
        ]
    )
    def test_linear(self, test_name, shape, bias):
        class linear(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.randn((256, 512))
                if bias:
                    self.bias = torch.randn((256))
                else:
                    self.bias = None

            def forward(self, x):
                return torch.ops.aten.linear.default(x, self.weight, self.bias)

        inputs = [torch.randn(shape)]
        self.run_test(linear(), inputs)

        # linear will be decomposed to P531484488 and view(reshape) can not handle reshape pattern
        # like (2, 3, n)->(6, n) in implicit mode which is similar to dynamic shape test below.

    # # Input is transposed through view [3,3,512]->[9,512]. Converter does not know dim=0 is dynamic now.
    @parameterized.expand(
        [
            (
                "2d_dim",
                (1, 512),
                (2, 512),
                (3, 512),
                torch.float32,
                (256, 512),
                None,
            ),
            (
                "3d_one_dynamic_dim",
                (1, 1, 512),
                (2, 2, 512),
                (3, 3, 512),
                torch.float32,
                (256, 512),
                (256,),
            ),
            (
                "3d_two_dynamic_dim_bias",
                (1, 1, 512),
                (2, 2, 512),
                (3, 3, 512),
                torch.float32,
                (256, 512),
                (256,),
            ),
            (
                "3d_two_dynamic_dim_no_bias",
                (1, 1, 512),
                (2, 2, 512),
                (3, 3, 512),
                torch.float32,
                (256, 512),
                None,
            ),
        ]
    )
    def test_linear_with_dynamic_shape(
        self, _, min_shape, opt_shape, max_shape, type, weight_shape, bias_shape
    ):
        class linear(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.rand(weight_shape)

                if bias_shape:
                    self.bias = torch.randn(bias_shape)
                else:
                    self.bias = None

            def forward(self, x):
                return torch.ops.aten.linear.default(x, self.weight, self.bias)

        input_specs = [
            Input(
                min_shape=min_shape,
                opt_shape=opt_shape,
                max_shape=max_shape,
                dtype=type,
            ),
        ]
        self.run_test_with_dynamic_shape(
            linear(),
            input_specs,
        )


if __name__ == "__main__":
    run_tests()
