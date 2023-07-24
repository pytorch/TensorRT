import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.dynamo.test_utils import DispatchTestCase
from torch_tensorrt import Input


class TestRSqrtConverter(DispatchTestCase):
    @parameterized.expand(
        [
            ("2d_dim_alpha", (2, 1), 2),
            ("3d_dim_alpha", (2, 1, 2), 2),
        ]
    )
    def test_sqrt(self, _, x, alpha):
        class sqrt(nn.Module):
            def forward(self, input):
                return torch.sqrt(input)

        inputs = [torch.randn(x)]
        self.run_test(
            sqrt(),
            inputs,
            expected_ops={torch.ops.aten.sqrt.default},
        )


if __name__ == "__main__":
    run_tests()