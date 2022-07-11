import torch
import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec


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


class TestLogicalOrMethodSimpleConverterWithDynamicShape(AccTestCase):
    def test_logical_or(self):
        class LogicalOr(torch.nn.Module):
            def forward(self, x, y):
                return x.logical_or(y)

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
            LogicalOr(), input_specs, expected_ops={acc_ops.logical_or}
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


class TestLogicalOrFunctionSimpleConverterWithDynamicShape(AccTestCase):
    def test_logical_or(self):
        class LogicalOr(torch.nn.Module):
            def forward(self, x, y):
                return torch.logical_or(x, y)

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
            LogicalOr(), input_specs, expected_ops={acc_ops.logical_or}
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


class TestLogicalOrOperatorSimpleConverterWithDynamicShape(AccTestCase):
    def test_logical_or(self):
        class LogicalOr(torch.nn.Module):
            def forward(self, x, y):
                return x | y

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
            LogicalOr(), input_specs, expected_ops={acc_ops.logical_or}
        )


if __name__ == "__main__":
    run_tests()
