from torch.testing._internal.common_fx2trt import AccTestCase
from parameterized import parameterized
from torch import nn
import torch
from fx2trt_oss.tracer.acc_tracer import acc_ops



class TestBmmConverters(AccTestCase):
    @parameterized.expand(
        [
            ("1", (10, 2, 13), (10, 13, 2)),
        ]
    )
    def test_bmm(self, _, input_shape, other_shape, alpha=1, beta=1):
        class Bmm(nn.Module):
            def forward(self, input, other):
                return torch.bmm(input, other)

        inputs = [torch.randn(*input_shape), torch.randn(*other_shape)]

        self.run_test(
            Bmm(),
            inputs,
            expected_ops={acc_ops.matmul}
        )

    @parameterized.expand(
        [
            ("default", (2, 3), (2, 3), (3, 3)),
            ("broadcast", (1, 1), (10, 7), (7, 13)),
        ]
    )
    def test_addmm(self, _, input_shape, m1_shape, m2_shape, alpha=1, beta=1):
        class Addmm(nn.Module):
            def forward(self, input, m1, m2):
                return torch.addmm(input, m1, m2, alpha=alpha, beta=beta)

        inputs = [torch.randn(input_shape), torch.randn(*m1_shape), torch.randn(*m2_shape)]
        test_implicit_batch_dim = len(input_shape) > 2

        self.run_test(
            Addmm(),
            inputs,
            expected_ops={acc_ops.matmul, acc_ops.add},
            test_implicit_batch_dim = test_implicit_batch_dim,
        )

    @parameterized.expand(
        [
            ("default", (10, 2, 2), (10, 2, 3), (10, 3, 2)),
            ("broadcast", (10, 2, 1), (10, 2, 3), (10, 3, 2)),
        ]
    )
    def test_baddbmm(self, _, input_shape, m1_shape, m2_shape, alpha=1, beta=1):
        class Baddbmm(nn.Module):
            def forward(self, input, m1, m2):
                return torch.baddbmm(input, m1, m2, alpha=alpha, beta=beta)

        inputs = [torch.randn(*input_shape), torch.randn(*m1_shape), torch.randn(*m2_shape)]

        self.run_test(
            Baddbmm(),
            inputs,
            expected_ops={acc_ops.matmul, acc_ops.add}
        )
