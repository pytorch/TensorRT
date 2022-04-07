import torch
import fx2trt_oss.tracer.acc_tracer.acc_ops as acc_ops
from torch.testing._internal.common_fx2trt import AccTestCase
from torch.testing._internal.common_utils import run_tests
from parameterized import parameterized

class TestNeFunctionConverter(AccTestCase):
    @parameterized.expand(
        [
            ("rand_2d_float_bool", torch.randn(3,4), torch.randn(3,4).to(torch.bool)),
            ("rand_2d_int_bool", torch.randn(3,4).to(torch.int), torch.randn(3,4).to(torch.bool)),
            ("rand_2d_bool_bool", torch.randn(3,4).to(torch.bool), torch.randn(3,4).to(torch.bool)),
            ("rand_2d_float_int", torch.randn(3,4).to(torch.float), torch.randn(3,4).to(torch.int)),
            ("rand_2d_float_single_bool", torch.randn(3,4), torch.tensor(0).to(torch.bool)),
            ("rand_2d_int_single_bool", torch.randn(3,4).to(torch.int), torch.tensor(0).to(torch.bool)),
            ("rand_2d_bool_single_bool", torch.randn(3,4).to(torch.bool), torch.tensor(0).to(torch.bool)),
        ]
    )
    def test_ne(self, _, input, other):
        class Ne(torch.nn.Module):
            def forward(self, x, y):
                return torch.ne(x, y)

        inputs = [
            input,
            other,
        ]
        self.run_test(Ne(), inputs, expected_ops={acc_ops.ne}, test_implicit_batch_dim = False)


class TestNeMethodConverter(AccTestCase):
    @parameterized.expand(
        [
            ("rand_2d_float_bool", torch.randn(3,4), torch.randn(3,4).to(torch.bool)),
            ("rand_2d_int_bool", torch.randn(3,4).to(torch.int), torch.randn(3,4).to(torch.bool)),
            ("rand_2d_bool_bool", torch.randn(3,4).to(torch.bool), torch.randn(3,4).to(torch.bool)),
            ("rand_2d_float_int", torch.randn(3,4).to(torch.float), torch.randn(3,4).to(torch.int)),
            ("rand_2d_float_single_bool", torch.randn(3,4), torch.tensor(0).to(torch.bool)),
            ("rand_2d_int_single_bool", torch.randn(3,4).to(torch.int), torch.tensor(0).to(torch.bool)),
            ("rand_2d_bool_single_bool", torch.randn(3,4).to(torch.bool), torch.tensor(0).to(torch.bool)),
        ]
    )
    def test_ne(self, _, input, other):
        class Ne(torch.nn.Module):
            def forward(self, x, y):
                return x.ne(y)

        inputs = [
            input,
            other,
        ]
        self.run_test(Ne(), inputs, expected_ops={acc_ops.ne}, test_implicit_batch_dim = False)


class TestNeOperatorConverter(AccTestCase):
    @parameterized.expand(
        [
            ("rand_2d_float_bool", torch.randn(3,4), torch.randn(3,4).to(torch.bool)),
            ("rand_2d_int_bool", torch.randn(3,4).to(torch.int), torch.randn(3,4).to(torch.bool)),
            ("rand_2d_bool_bool", torch.randn(3,4).to(torch.bool), torch.randn(3,4).to(torch.bool)),
            ("rand_2d_float_int", torch.randn(3,4).to(torch.float), torch.randn(3,4).to(torch.int)),
            ("rand_2d_float_single_bool", torch.randn(3,4), torch.tensor(0).to(torch.bool)),
            ("rand_2d_int_single_bool", torch.randn(3,4).to(torch.int), torch.tensor(0).to(torch.bool)),
            ("rand_2d_bool_single_bool", torch.randn(3,4).to(torch.bool), torch.tensor(0).to(torch.bool)),
        ]
    )
    def test_ne(self, _, input, other):
        class Ne(torch.nn.Module):
            def forward(self, x, y):
                return x != y

        inputs = [
            input,
            other,
        ]
        self.run_test(Ne(), inputs, expected_ops={acc_ops.ne}, test_implicit_batch_dim = False)


class TestNeOperatorConstantConverter(AccTestCase):
    @parameterized.expand(
        [
            ("rand_2d_float_bool", torch.randn(3,4), torch.randn(3,4).to(torch.bool)),
            ("rand_2d_int_bool", torch.randn(3,4).to(torch.int), torch.randn(3,4).to(torch.bool)),
            ("rand_2d_bool_bool", torch.randn(3,4).to(torch.bool), torch.randn(3,4).to(torch.bool)),
            ("rand_2d_float_int", torch.randn(3,4).to(torch.float), torch.randn(3,4).to(torch.int)),
            ("rand_2d_float_single_bool", torch.randn(3,4), False),
            ("rand_2d_int_single_bool", torch.randn(3,4).to(torch.int), False),
            ("rand_2d_bool_single_bool", torch.randn(3,4).to(torch.bool), False),
        ]
    )
    def test_ne(self, _, input, other):
        class Ne(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.other = other

            def forward(self, x):
                return  x != self.other

        inputs = [
            input,

        ]
        self.run_test(Ne(), inputs, expected_ops={acc_ops.ne}, test_implicit_batch_dim = False)


class TestConstInputConverter(AccTestCase):
    def test_ne(self):
        class Ne(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return  x.shape[0] != 4

        input = torch.randn(3,4)
        inputs = [
            input,
        ]
        self.run_test(Ne(), inputs, expected_ops={acc_ops.ne}, test_implicit_batch_dim = False)


if __name__ == '__main__':
    run_tests()
