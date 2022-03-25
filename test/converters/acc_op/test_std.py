# Owner(s): ["oncall: aiacc"]

import torch
import fx2trt_oss.tracer.acc_tracer.acc_ops as acc_ops
from torch.testing._internal.common_fx2trt import AccTestCase
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests


class TestMinConverter(AccTestCase):
    @parameterized.expand(
        [
            ("norm_1d", (-1), ),
            ("norm_2d", (2, 3), ),
        ]
    )
    def test_std(self, _, dim):
        class Std(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.std(x, unbiased=True, dim=dim, keepdim=True)

        inputs = [torch.randn(2, 3, 4, 5)]
        self.run_test(
            Std(),
            inputs,
            expected_ops={acc_ops.mean, acc_ops.sub, acc_ops.pow, acc_ops.sqrt},
        )

    @parameterized.expand(
        [
            ("norm_1d", (-1), ),
            ("norm_2d", (2, 3), ),
        ]
    )
    def test_std_method(self, _, dim):
        class Std(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.std(unbiased=True, dim=dim, keepdim=True)

        inputs = [torch.randn(2, 3, 4, 5)]
        self.run_test(
            Std(),
            inputs,
            expected_ops={acc_ops.mean, acc_ops.sub, acc_ops.pow, acc_ops.sqrt},
        )
if __name__ == '__main__':
    run_tests()
