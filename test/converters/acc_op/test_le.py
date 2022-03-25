import torch
import fx2trt_oss.tracer.acc_tracer.acc_ops as acc_ops
from torch.testing._internal.common_fx2trt import AccTestCase
from torch.testing._internal.common_utils import run_tests
from parameterized import parameterized

class TestLeConverter(AccTestCase):
    @parameterized.expand(
        [
            ("rand_2d", torch.randn(3,4), torch.randn(3,4)),
            ("rand_3d", torch.randn(3,4,5), torch.randn(3,4,5)),
            ("rand_4d", torch.randn(3,4,5,6), torch.randn(3,4,5,6)),
        ]
    )
    def test_le(self, _, input, other):
        class Le(torch.nn.Module):
            def forward(self, x, y):
                mask = torch.le(x, y)
                return x.masked_fill(mask, 5)

        inputs = [
            input,
            other,
        ]
        self.run_test(Le(), inputs, expected_ops={acc_ops.le}, test_implicit_batch_dim = False)


class TestLeMethodConverter(AccTestCase):
    @parameterized.expand(
        [
            ("rand_2d", torch.randn(3,4), torch.randn(3,4)),
            ("rand_3d", torch.randn(3,4,5), torch.randn(3,4,5)),
            ("rand_4d", torch.randn(3,4,5,6), torch.randn(3,4,5,6)),
        ]
    )
    def test_le(self, _, input, other):
        class Le(torch.nn.Module):
            def forward(self, x, y):
                mask = x.le(y)
                return x.masked_fill(mask, 5)

        inputs = [
            input,
            other,
        ]
        self.run_test(Le(), inputs, expected_ops={acc_ops.le}, test_implicit_batch_dim = False)

if __name__ == '__main__':
    run_tests()
