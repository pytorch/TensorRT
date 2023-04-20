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
    def test_rsub(self, _, x, alpha):
        class rsub(nn.Module):
            def forward(self, input):
                return torch.rsub(input, input, alpha=alpha)

        inputs = [torch.randn(x)]
        self.run_test(
            rsub(),
            inputs,
            expected_ops={torch.ops.aten.rsub.Tensor},
        )


if __name__ == "__main__":
    run_tests()
