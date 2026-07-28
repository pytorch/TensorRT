import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


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


class TestLogicalXorMethodSimpleConverterWithDynamicShape(AccTestCase):
    def test_logical_xor(self):
        class LogicalXor(torch.nn.Module):
            def forward(self, x, y):
                return x.logical_xor(y)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (2, 3, 4, 5), (2, 3, 10, 10))],
            ),
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (2, 3, 4, 5), (2, 3, 10, 10))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            LogicalXor(), input_specs, expected_ops={acc_ops.logical_xor}
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


class TestLogicalXorFunctionSimpleConverterWithDynamicShape(AccTestCase):
    def test_logical_xor(self):
        class LogicalXor(torch.nn.Module):
            def forward(self, x, y):
                return torch.logical_xor(x, y)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (2, 3, 4, 5), (2, 3, 10, 10))],
            ),
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (2, 3, 4, 5), (2, 3, 10, 10))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            LogicalXor(), input_specs, expected_ops={acc_ops.logical_xor}
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


class TestLogicalXorOperatorSimpleConverterWithDynamicShape(AccTestCase):
    def test_logical_xor(self):
        class LogicalXor(torch.nn.Module):
            def forward(self, x, y):
                return x ^ y

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.bool,
                shape_ranges=[((1, 1, 5, 5), (2, 3, 5, 5), (2, 3, 5, 5))],
            ),
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.bool,
                shape_ranges=[((1, 1, 5, 5), (2, 3, 5, 5), (2, 3, 5, 5))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            LogicalXor(), input_specs, expected_ops={acc_ops.logical_xor}
        )


if __name__ == "__main__":
    run_tests()
