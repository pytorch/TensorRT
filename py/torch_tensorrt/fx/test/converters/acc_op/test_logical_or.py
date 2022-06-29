import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase


class TestLogicalOrMethodSimpleConverter(AccTestCase):
    @parameterized.expand(
        [
            ("rand_2d_bool_bool", torch.randn(3, 4) > 0, torch.randn(3, 4) > 0),
            ("rand_3d_bool_bool", torch.randn(3, 4, 5) > 0, torch.randn(3, 4, 5) > 0),
            (
                "rand_4d_bool_bool",
                torch.randn(3, 4, 5, 6) > 0,
                torch.randn(3, 4, 5, 6) > 0,
            ),
            ("rand_2d_bool_single_bool", torch.randn(3, 4) > 0, torch.tensor(0) > 0),
            (
                "rand_2d_int_bool",
                torch.randn(3, 4).to(torch.int),
                torch.randn(3, 4) > 0,
            ),
            (
                "rand_2d_int_single_bool",
                torch.randn(3, 4).to(torch.int),
                torch.tensor(0) > 0,
            ),
        ]
    )
    def test_logical_or(self, _, input, other):
        class LogicalOr(torch.nn.Module):
            def forward(self, x, y):
                return x.logical_or(y)

        inputs = [
            input,
            other,
        ]
        self.run_test(
            LogicalOr(),
            inputs,
            expected_ops={acc_ops.logical_or},
            test_implicit_batch_dim=False,
        )


class TestLogicalOrFunctionSimpleConverter(AccTestCase):
    @parameterized.expand(
        [
            ("rand_2d_bool_bool", torch.randn(3, 4) > 0, torch.randn(3, 4) > 0),
            ("rand_3d_bool_bool", torch.randn(3, 4, 5) > 0, torch.randn(3, 4, 5) > 0),
            (
                "rand_4d_bool_bool",
                torch.randn(3, 4, 5, 6) > 0,
                torch.randn(3, 4, 5, 6) > 0,
            ),
            ("rand_2d_bool_single_bool", torch.randn(3, 4) > 0, torch.tensor(0) > 0),
            (
                "rand_2d_int_bool",
                torch.randn(3, 4).to(torch.int),
                torch.randn(3, 4) > 0,
            ),
            (
                "rand_2d_int_single_bool",
                torch.randn(3, 4).to(torch.int),
                torch.tensor(0) > 0,
            ),
        ]
    )
    def test_logical_or(self, _, input, other):
        class LogicalOr(torch.nn.Module):
            def forward(self, x, y):
                return torch.logical_or(x, y)

        inputs = [
            input,
            other,
        ]
        self.run_test(
            LogicalOr(),
            inputs,
            expected_ops={acc_ops.logical_or},
            test_implicit_batch_dim=False,
        )


class TestLogicalOrOperatorSimpleConverter(AccTestCase):
    @parameterized.expand(
        [
            ("rand_2d_bool_bool", torch.randn(3, 4) > 0, torch.randn(3, 4) > 0),
            ("rand_3d_bool_bool", torch.randn(3, 4, 5) > 0, torch.randn(3, 4, 5) > 0),
            (
                "rand_4d_bool_bool",
                torch.randn(3, 4, 5, 6) > 0,
                torch.randn(3, 4, 5, 6) > 0,
            ),
            ("rand_2d_bool_single_bool", torch.randn(3, 4) > 0, torch.tensor(0) > 0),
            (
                "rand_2d_int_bool",
                torch.randn(3, 4).to(torch.int),
                torch.randn(3, 4) > 0,
            ),
            (
                "rand_2d_int_single_bool",
                torch.randn(3, 4).to(torch.int),
                torch.tensor(0) > 0,
            ),
        ]
    )
    def test_logical_or(self, _, input, other):
        class LogicalOr(torch.nn.Module):
            def forward(self, x, y):
                return x | y

        inputs = [
            input,
            other,
        ]
        self.run_test(
            LogicalOr(),
            inputs,
            expected_ops={acc_ops.logical_or},
            test_implicit_batch_dim=False,
        )


if __name__ == "__main__":
    run_tests()
