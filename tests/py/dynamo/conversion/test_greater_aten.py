import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

from .harness import DispatchTestCase


class TestGreaterConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d", (2, 1)),
            ("3d", (2, 1, 2)),
        ]
    )
    def test_greater_tensor(self, _, shape):
        class greater(nn.Module):
            def forward(self, lhs_val, rhs_val):
                return torch.ops.aten.gt.Tensor(lhs_val, rhs_val)

        inputs = [torch.randn(shape), torch.randn(shape)]
        self.run_test(
            greater(),
            inputs,
            output_dtypes=[torch.bool],
        )

    @parameterized.expand(
        [
            ("2d", (2, 1), 1),
            ("3d", (2, 1, 2), 2.0),
        ]
    )
    def test_greater_tensor_scalar(self, _, shape, scalar):
        class greater(nn.Module):
            def forward(self, lhs_val):
                return torch.ops.aten.gt.Tensor(lhs_val, torch.tensor(scalar))

        inputs = [torch.randn(shape)]
        self.run_test(
            greater(),
            inputs,
            output_dtypes=[torch.bool],
        )


if __name__ == "__main__":
    run_tests()
