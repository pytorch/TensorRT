import torch
import torch.nn as nn
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_fx2trt import AccTestCase
from torch.testing._internal.common_utils import run_tests


class TestConverter(AccTestCase):
    @parameterized.expand(
        [
            ("2d_dim_v1", (5, 5), (2, 3), (1, 2), 0),
            ("2d_dim_v2", (5, 5), (2, 3), (2, 2), 1),
            ("3d_dim_v1", (20, 20), (2, 3, 2), (2, 2, 2), 0),
            # take long time on large dimensions, we do not have better implementation yet
            # ("4d_dim_v1", (200, 200, 200, 200), (9, 9, 3, 2), (2, 2, 2, 3), 0),
            # ("4d_dim_v2", (200, 200, 200, 200), (1, 15, 512, 1), (4096, 256, 1, 1), 0),
        ]
    )
    def test_as_strided(self, _, x_size, size, stride, offset):
        class Stride(nn.Module):
            def forward(self, x):
                return torch.as_strided(x, size, stride, offset)

        inputs = [torch.randn(*x_size)]
        self.run_test(
            Stride(),
            inputs,
            expected_ops={acc_ops.as_strided},
            test_implicit_batch_dim=False,
        )


if __name__ == "__main__":
    run_tests()
