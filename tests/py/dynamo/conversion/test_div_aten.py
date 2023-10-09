import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestDivConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_div_tensor(self, _, shape):
        class div(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.div.Tensor(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            div(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), None),
            ("3d", (2, 1, 2), "trunc"),
            ("3d", (2, 3, 2), "floor"),
        ]
    )
    def test_div_tensor_rounding_mode(self, _, shape, rounding_mode):
        class div(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.div.Tensor_mode(
                    lhs_val, rhs_val, rounding_mode=rounding_mode
                )

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            div(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), -1),
            ("3d", (2, 1, 2), 2.0),
        ]
    )
    def test_div_tensor(self, _, shape, scalar):
        class div(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.div.Tensor(lhs_val, scalar)

        inputs = [torch.randn(shape)]
        self.run_test(
            div(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1.0, None),
            ("3d", (2, 1, 2), 2.0, "trunc"),
            ("3d", (2, 3, 2), -3.0, "floor"),
        ]
    )
    def test_div_tensor_rounding_mode(self, _, shape, scalar, rounding_mode):
        class div(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.div.Tensor_mode(
                    lhs_val, scalar, rounding_mode=rounding_mode
                )

        inputs = [torch.randn(shape)]
        self.run_test(
            div(),
            inputs,
        )

    @parameterized.expand(
        [
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_prims_div_tensor(self, _, shape):
        class div(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.prims.div.default(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            div(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
