from typing import Callable

import torch
import torch.nn as nn

import torch_tensorrt.fx.tracer.acc_tracer.acc_ops as acc_ops
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import AccTestCase, InputTensorSpec

unary_ops = [
    (torch.sin, acc_ops.sin),
    (torch.cos, acc_ops.cos),
    (torch.tan, acc_ops.tan),
    (torch.sinh, acc_ops.sinh),
    (torch.cosh, acc_ops.cosh),
    (torch.asin, acc_ops.asin),
    (torch.acos, acc_ops.acos),
    (torch.atan, acc_ops.atan),
    (torch.abs, acc_ops.abs),
    (torch.neg, acc_ops.neg),
    (torch.reciprocal, acc_ops.reciprocal),
    (torch.sqrt, acc_ops.sqrt),
    (torch.log, acc_ops.log),
    (torch.exp, acc_ops.exp),
    (torch.floor, acc_ops.floor),
    (torch.ceil, acc_ops.ceil),
    (torch.sign, acc_ops.sign),
]


class TestUnaryOpConverters(AccTestCase):
    @parameterized.expand([(op[1].__name__, op[0], op[1]) for op in unary_ops])
    def test_unary_ops(self, name, orig_op: Callable, expected_op):
        class TestModule(nn.Module):
            def __init__(self, orig_op):
                super().__init__()
                self.orig_op = orig_op

            def forward(self, x):
                return self.orig_op(x)

        m = TestModule(orig_op)
        inputs = [torch.randn(2, 2, 3)]
        self.run_test(m, inputs, expected_ops={expected_op})


class TestUnaryOpConvertersWithDynamicShapeFourDimensions(AccTestCase):
    @parameterized.expand([(op[1].__name__, op[0], op[1]) for op in unary_ops])
    def test_unary_ops(self, name, orig_op: Callable, expected_op):
        class TestModule(nn.Module):
            def __init__(self, orig_op):
                super().__init__()
                self.orig_op = orig_op

            def forward(self, x):
                return self.orig_op(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (3, 3, 3, 3), (3, 3, 3, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(orig_op), input_specs, expected_ops={expected_op}
        )


class TestUnaryOpNotConverters(AccTestCase):
    @parameterized.expand(
        [
            ("not_bool", torch.logical_not, acc_ops.logical_not, torch.bool),
            ("not_float", torch.logical_not, acc_ops.logical_not, torch.float),
            ("not_int", torch.logical_not, acc_ops.logical_not, torch.int),
        ]
    )
    def test_unary_ops(self, name, orig_op: Callable, expected_op, input_dtype):
        class TestModule(nn.Module):
            def __init__(self, orig_op):
                super().__init__()
                self.orig_op = orig_op

            def forward(self, x):
                x = self.orig_op(x)
                return self.orig_op(x)

        m = TestModule(orig_op)
        inputs = [torch.randn(2, 2, 3).to(input_dtype)]
        self.run_test(
            m, inputs, expected_ops={expected_op}, test_implicit_batch_dim=False
        )


class TestUnaryOpNotConvertersWithDynamicShapeFourDimensions(AccTestCase):
    @parameterized.expand(
        [
            ("not_bool", torch.logical_not, acc_ops.logical_not, torch.bool),
            ("not_float", torch.logical_not, acc_ops.logical_not, torch.float),
            ("not_int", torch.logical_not, acc_ops.logical_not, torch.int),
        ]
    )
    def test_unary_ops(self, name, orig_op: Callable, expected_op, input_dtype):
        class TestModule(nn.Module):
            def __init__(self, orig_op):
                super().__init__()
                self.orig_op = orig_op

            def forward(self, x):
                x = self.orig_op(x)
                return self.orig_op(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (3, 3, 3, 3), (3, 3, 3, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(orig_op), input_specs, expected_ops={expected_op}
        )


class TestUnaryRSQRTConverters(AccTestCase):
    def test_unary_ops(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.rsqrt(x)

        m = TestModule()
        inputs = [torch.randn(2, 2, 3)]
        self.run_test(m, inputs, expected_ops={acc_ops.sqrt, acc_ops.reciprocal})


class TestUnaryRSQRTConvertersWithDynamicShapeFourDimensions(AccTestCase):
    def test_unary_ops(self):
        class TestModule(nn.Module):
            def forward(self, x):
                return torch.rsqrt(x)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (3, 3, 3, 3), (3, 3, 3, 3))],
            ),
        ]

        self.run_test_with_dynamic_shape(
            TestModule(), input_specs, expected_ops={acc_ops.sqrt, acc_ops.reciprocal}
        )


if __name__ == "__main__":
    run_tests()
