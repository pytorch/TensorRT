import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import DispatchTestCase, InputTensorSpec


class TestRSubConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d_dim_alpha", (2, 1), 2),
            ("3d_dim_alpha", (2, 1, 2), 2),
        ]
    )
    def test_rsqrt(self, _, x, alpha):
        class rsqrt(nn.Module):
            def forward(self, input):
                return torch.rsqrt(input, input, alpha)

        inputs = [torch.randn(x) + 1]
        self.run_test(
            rsqrt(),
            inputs,
            expected_ops=torch.ops.aten.rsqrt,
        )


if __name__ == "__main__":
    run_tests()
