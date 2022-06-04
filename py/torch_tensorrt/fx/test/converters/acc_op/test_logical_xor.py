import fx2trt_oss.tracer.acc_tracer.acc_ops as acc_ops
import torch
from parameterized import parameterized
from torch.testing._internal.common_fx2trt import AccTestCase
from torch.testing._internal.common_utils import run_tests


class TestLogicalXorMethodSimpleConverter(AccTestCase):
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
    def test_logical_xor(self, _, input, other):
        class LogicalXor(torch.nn.Module):
            def forward(self, x, y):
                return x.logical_xor(y)

        inputs = [
            input,
            other,
        ]
        self.run_test(
            LogicalXor(),
            inputs,
            expected_ops={acc_ops.logical_xor},
            test_implicit_batch_dim=False,
        )


class TestLogicalXorFunctionSimpleConverter(AccTestCase):
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
    def test_logical_xor(self, _, input, other):
        class LogicalXor(torch.nn.Module):
            def forward(self, x, y):
                return torch.logical_xor(x, y)

        inputs = [
            input,
            other,
        ]
        self.run_test(
            LogicalXor(),
            inputs,
            expected_ops={acc_ops.logical_xor},
            test_implicit_batch_dim=False,
        )


class TestLogicalXorOperatorSimpleConverter(AccTestCase):
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
    def test_logical_xor(self, _, input, other):
        class LogicalXor(torch.nn.Module):
            def forward(self, x, y):
                return x ^ y

        inputs = [
            input,
            other,
        ]
        self.run_test(
            LogicalXor(),
            inputs,
            expected_ops={acc_ops.logical_xor},
            test_implicit_batch_dim=False,
        )


if __name__ == "__main__":
    run_tests()
