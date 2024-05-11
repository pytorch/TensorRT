import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from .harness import DispatchTestCase


class TestAtan2OutConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ((10,), (5,), torch.float),
            ((10,), (10,), torch.float),
        ]
    )
    def test_atan2_float(self, input_shape, out_shape, dtype):
        class atan2_out(nn.Module):
            def forward(self, lhs_val, rhs_val, out):
                return torch.ops.aten.atan2.out(lhs_val, rhs_val, out=out)

        out = torch.empty(out_shape)

        inputs = [
            torch.randn(input_shape, dtype=dtype),
            torch.randn(input_shape, dtype=dtype),
            out,
        ]

        self.run_test(
            atan2_out(),
            inputs,
        )


if __name__ == "__main__":
    run_tests()
