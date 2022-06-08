import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_fx2trt import AccTestCase
from torch.testing._internal.common_utils import run_tests


class TestAndMethodSimpleConverter(AccTestCase):
    @parameterized.expand(
        [
            ("rand_2d_float_bool", torch.randn(3, 4), torch.randn(3, 4).to(torch.bool)),
            (
                "rand_2d_int_bool",
                torch.randn(3, 4).to(torch.int),
                torch.randn(3, 4).to(torch.bool),
            ),
            (
                "rand_2d_bool_bool",
                torch.randn(3, 4).to(torch.bool),
                torch.randn(3, 4).to(torch.bool),
            ),
            (
                "rand_2d_float_int",
                torch.randn(3, 4).to(torch.float),
                torch.randn(3, 4).to(torch.int),
            ),
            (
                "rand_2d_float_single_bool",
                torch.randn(3, 4),
                torch.tensor(0).to(torch.bool),
            ),
            (
                "rand_2d_int_single_bool",
                torch.randn(3, 4).to(torch.int),
                torch.tensor(0).to(torch.bool),
            ),
            (
                "rand_2d_bool_single_bool",
                torch.randn(3, 4).to(torch.bool),
                torch.tensor(0).to(torch.bool),
            ),
        ]
    )
    def test_and(self, _, input, other):
        class And(torch.nn.Module):
            def forward(self, x, y):
                return x.logical_and(y)

        inputs = [
            input,
            other,
        ]
        self.run_test(
            And(),
            inputs,
            expected_ops={acc_ops.logical_and},
            test_implicit_batch_dim=False,
        )


class TestAndFunctionSimpleConverter(AccTestCase):
    @parameterized.expand(
        [
            ("rand_2d_float_bool", torch.randn(3, 4), torch.randn(3, 4).to(torch.bool)),
            (
                "rand_2d_int_bool",
                torch.randn(3, 4).to(torch.int),
                torch.randn(3, 4).to(torch.bool),
            ),
            (
                "rand_2d_bool_bool",
                torch.randn(3, 4).to(torch.bool),
                torch.randn(3, 4).to(torch.bool),
            ),
            (
                "rand_2d_float_int",
                torch.randn(3, 4).to(torch.float),
                torch.randn(3, 4).to(torch.int),
            ),
            (
                "rand_2d_float_single_bool",
                torch.randn(3, 4),
                torch.tensor(0).to(torch.bool),
            ),
            (
                "rand_2d_int_single_bool",
                torch.randn(3, 4).to(torch.int),
                torch.tensor(0).to(torch.bool),
            ),
            (
                "rand_2d_bool_single_bool",
                torch.randn(3, 4).to(torch.bool),
                torch.tensor(0).to(torch.bool),
            ),
        ]
    )
    def test_and(self, _, input, other):
        class And(torch.nn.Module):
            def forward(self, x, y):
                return torch.logical_and(x, y)

        inputs = [
            input,
            other,
        ]
        self.run_test(
            And(),
            inputs,
            expected_ops={acc_ops.logical_and},
            test_implicit_batch_dim=False,
        )


class TestAndOperatorSimpleConverter(AccTestCase):
    @parameterized.expand(
        [
            (
                "rand_2d_bool_bool",
                torch.randn(3, 4).to(torch.bool),
                torch.randn(3, 4).to(torch.bool),
            ),
            (
                "rand_2d_bool_single_bool",
                torch.randn(3, 4).to(torch.bool),
                torch.tensor(0).to(torch.bool),
            ),
        ]
    )
    def test_and(self, _, input, other):
        class And(torch.nn.Module):
            def forward(self, x, y):
                return x & y

        inputs = [
            input,
            other,
        ]
        self.run_test(
            And(),
            inputs,
            expected_ops={acc_ops.bitwise_and},
            test_implicit_batch_dim=False,
        )


class TestAndOperatorConstantConverter(AccTestCase):
    @parameterized.expand(
        [
            (
                "rand_2d_bool_bool",
                torch.randn(3, 4).to(torch.bool),
                torch.randn(3, 4).to(torch.bool),
            ),
            (
                "rand_2d_bool_single_bool",
                torch.randn(3, 4).to(torch.bool),
                torch.tensor(0).to(torch.bool),
            ),
        ]
    )
    def test_and(self, _, input, other):
        class And(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.other = other

            def forward(self, x):
                return x & self.other

        inputs = [
            input,
        ]
        self.run_test(
            And(),
            inputs,
            expected_ops={acc_ops.bitwise_and},
            test_implicit_batch_dim=False,
        )


if __name__ == "__main__":
    run_tests()
