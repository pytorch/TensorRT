import fx2trt_oss.tracer.acc_tracer.acc_ops as acc_ops
import torch
from parameterized import parameterized
from torch.testing._internal.common_fx2trt import AccTestCase
from torch.testing._internal.common_utils import run_tests


class TestMinConverter(AccTestCase):
    @parameterized.expand(
        [
            ("norm_1d", (-1), False),
            ("norm_1d", (-1), True),
            ("norm_2d", (2, 3), False),
            ("norm_2d", (2, 3), True),
        ]
    )
    def test_std(self, _, dim, unbiased):
        class Std(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.std(x, dim, unbiased=unbiased, keepdim=True)

        inputs = [torch.randn(2, 3, 4, 5)]
        self.run_test(
            Std(),
            inputs,
            expected_ops={acc_ops.mean, acc_ops.sub, acc_ops.pow, acc_ops.sqrt},
        )

    @parameterized.expand(
        [
            ("norm_1d", (-1), True),
            ("norm_1d", (-1), False),
            ("norm_2d", (2, 3), True),
            ("norm_2d", (2, 3), False),
        ]
    )
    def test_std_method(self, _, dim, unbiased):
        class Std(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.std(dim, unbiased=unbiased, keepdim=True)

        inputs = [torch.randn(2, 3, 4, 5)]
        self.run_test(
            Std(),
            inputs,
            expected_ops={acc_ops.mean, acc_ops.sub, acc_ops.pow, acc_ops.sqrt},
        )


if __name__ == "__main__":
    run_tests()
