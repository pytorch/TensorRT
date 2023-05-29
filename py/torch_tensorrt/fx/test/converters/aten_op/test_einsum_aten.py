import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import DispatchTestCase, InputTensorSpec


class TestConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d_dim", "ij,jk->ik", (2, 3), (3, 4)),
            ("2d_dim_ext", "ij,kj->ik", (2, 3), (4, 3)),
            ("3d_dim", "cxd,cyd->cxy", (3, 4, 5), (3, 6, 5)),
            ("4d_dim", "bcwd,bcdh->bcwh", (2, 3, 4, 5), (2, 3, 5, 6)),
            ("4d_dim_ext", "bcxd,bcyd->bcxy", (2, 3, 4, 5), (2, 3, 6, 5)),
            # TRT does not support ellipsis or diagonal operations
        ]
    )
    def test_einsum(self, _, equation, x_size, y_size):
        class Einsum(nn.Module):
            def forward(self, x, y):
                return torch.einsum(equation, x, y)

        inputs = [torch.randn(*x_size), torch.randn(*y_size)]
        self.run_test(
            Einsum(),
            inputs,
            expected_ops={torch.ops.aten.einsum},
            # test_implicit_batch_dim=False,
        )

    @parameterized.expand(
        [
            ("4d_dim", "bcwd,bcdh->bcwh", (2, 3, 4, 5), (2, 3, 5, 6)),
            ("4d_dim_ext", "bcxd,bcyd->bcxy", (2, 3, 4, 5), (2, 3, 6, 5)),
            # TRT does not support ellipsis or diagonal operations
        ]
    )
    def test_einsum_with_dynamic_shape_four_dimensions(
        self, _, equation, x_size, y_size
    ):
        class Einsum(nn.Module):
            def forward(self, x, y):
                return torch.einsum(equation, x, y)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 3, 3), (1, 2, 3, 3), (3, 3, 3, 3))],
            ),
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 3, 3), (1, 2, 3, 3), (3, 3, 3, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            Einsum(), input_specs, expected_ops={torch.ops.aten.einsum}
        )


if __name__ == "__main__":
    run_tests()
