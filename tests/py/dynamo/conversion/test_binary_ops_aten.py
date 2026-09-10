import operator
import unittest
from typing import Callable

import tensorrt as trt
import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input
from torch_tensorrt.dynamo.conversion.impl.elementwise import base as elementwise_base

from .harness import DispatchTestCase

NEED_TEST_BOTH_CONSTANTS_CASE = True

elementwise_ops = [
    ((lambda x, y: torch.ops.aten.add.Tensor(x, y)), NEED_TEST_BOTH_CONSTANTS_CASE),
    ((lambda x, y: torch.ops.aten.sub.Tensor(x, y)), NEED_TEST_BOTH_CONSTANTS_CASE),
    ((lambda x, y: torch.ops.aten.div.Tensor(x, y)), NEED_TEST_BOTH_CONSTANTS_CASE),
    (
        (lambda x, y: torch.ops.aten.floor_divide.default(x, y)),
        NEED_TEST_BOTH_CONSTANTS_CASE,
    ),
    (
        (lambda x, y: torch.ops.aten.div.Tensor_mode(x, y, rounding_mode="trunc")),
        not NEED_TEST_BOTH_CONSTANTS_CASE,
    ),
    (
        (lambda x, y: torch.ops.aten.div.Tensor_mode(x, y, rounding_mode="floor")),
        NEED_TEST_BOTH_CONSTANTS_CASE,
    ),
    (
        torch.ops.aten.fmod.Tensor,
        not NEED_TEST_BOTH_CONSTANTS_CASE,
    ),
    ## torch.floor_divide rounds result toward zero, rather than -Inf.
    ## https://github.com/pytorch/pytorch/issues/43874
    (
        (lambda x, y: torch.ops.aten.floor_divide.default(x, y)),
        not NEED_TEST_BOTH_CONSTANTS_CASE,
    ),
    ((lambda x, y: torch.ops.aten.mul.Tensor(x, y)), NEED_TEST_BOTH_CONSTANTS_CASE),
    (torch.ops.aten.pow.Tensor_Tensor, not NEED_TEST_BOTH_CONSTANTS_CASE),
]


class TestBinaryOpConverters(DispatchTestCase):
    @parameterized.expand([(op[0].__name__, op[0]) for op in elementwise_ops])
    def test_elementwise_ops(self, name, orig_op: Callable):
        class TestModule(nn.Module):
            def __init__(self, orig_op):
                super().__init__()
                self.orig_op = orig_op

            def forward(self, x):
                return self.orig_op(x, x)

        m = TestModule(orig_op)
        # Avoid dividing by 0.
        inputs = [torch.rand(1, 1) + 1]
        self.run_test(m, inputs)

    @parameterized.expand([(op[0].__name__, op[0]) for op in elementwise_ops])
    def test_elementwise_ops_mismatched_dtypes(self, name, orig_op: Callable):
        class TestModule(nn.Module):
            def __init__(self, orig_op):
                super().__init__()
                self.orig_op = orig_op

            def forward(self, x, y):
                return self.orig_op(x, y)

        m = TestModule(orig_op)
        # Avoid dividing by 0.
        inputs = [
            2 * torch.rand(1, 1, dtype=torch.float) + 1,
            torch.randint(1, 3, (1, 1), dtype=torch.int),
        ]
        self.run_test(m, inputs)

    @parameterized.expand(
        [
            (op[0].__name__, op[0])
            for op in elementwise_ops
            if op[0].__name__ not in ["pow.Tensor_Tensor", "fmod.Tensor"]
        ]
    )
    def test_elementwise_ops_with_one_constant(self, name, orig_op: Callable):
        class TestModule(nn.Module):
            def __init__(self, orig_op):
                super().__init__()
                self.constant = torch.randn(1)
                self.orig_op = orig_op

            def forward(self, x):
                x = self.orig_op(x, self.constant)
                return self.orig_op(x, -2)

        m = TestModule(orig_op)
        inputs = [torch.randn(2, 2)]
        self.run_test(m, inputs)

    @parameterized.expand([(op[0].__name__, op[0]) for op in elementwise_ops if op[1]])
    def test_elementwise_op_with_both_constants(self, name, orig_op: Callable):
        class TestModule(nn.Module):
            def __init__(self, orig_op):
                super().__init__()
                self.constant0 = torch.nn.Parameter(torch.randn(1))
                self.constant1 = torch.nn.Parameter(torch.randn(1))
                self.orig_op = orig_op

            def forward(self, x):
                const = self.orig_op(self.constant0, self.constant1)
                return self.orig_op(x, const)

        m = TestModule(orig_op)
        inputs = [torch.randn(2, 2)]
        self.run_test(m, inputs)

    @parameterized.expand(
        [
            ("maximum", torch.ops.aten.maximum.default),
            ("minimum", torch.ops.aten.minimum.default),
        ]
    )
    def test_elementwise_op_with_both_constants_multi_element(
        self, name, orig_op: Callable
    ):
        """A constant of more than one element. Folding two of those in Python compares
        the whole operands rather than their elements, so max returns one operand."""

        class TestModule(nn.Module):
            def __init__(self, orig_op):
                super().__init__()
                self.constant0 = torch.nn.Parameter(torch.randn(3))
                self.constant1 = torch.nn.Parameter(torch.randn(3))
                self.orig_op = orig_op

            def forward(self, x):
                const = self.orig_op(self.constant0, self.constant1)
                return self.orig_op(x, const)

        m = TestModule(orig_op)
        inputs = [torch.randn(3)]
        self.run_test(m, inputs)

    @parameterized.expand(
        [
            ("logical_and", torch.ops.aten.logical_and.default),
            ("logical_or", torch.ops.aten.logical_or.default),
            ("logical_xor", torch.ops.aten.logical_xor.default),
        ]
    )
    def test_logical_op_with_both_constants_multi_element(
        self, name, orig_op: Callable
    ):
        """Folding two bool constants in Python calls __bool__ on an operand, which raises
        for more than one element. The result also has to stay bool: uint8 is rejected when
        the constant is built."""

        class TestModule(nn.Module):
            def __init__(self, orig_op):
                super().__init__()
                # Parameters keep this operation in the graph during legacy tracing.
                self.constant0 = nn.Parameter(
                    torch.tensor([True, False, True]), requires_grad=False
                )
                self.constant1 = nn.Parameter(
                    torch.tensor([True, True, False]), requires_grad=False
                )
                self.orig_op = orig_op

            def forward(self, x):
                const = self.orig_op(self.constant0, self.constant1)
                return self.orig_op(torch.ops.aten.gt.Scalar(x, 0), const)

        m = TestModule(orig_op)
        inputs = [torch.randn(3)]
        self.run_test(m, inputs)

    def test_constant_fold_preserves_float16_overflow(self):
        class FoldHalf(nn.Module):
            def __init__(self):
                super().__init__()
                self.constant = nn.Parameter(torch.tensor([300.0], dtype=torch.float16))

            def forward(self, x):
                value = torch.ops.aten.mul.Scalar(self.constant, 1000.0)
                value = torch.ops.aten.div.Scalar(value, 1000.0)
                return torch.ops.aten.add.Tensor(x, value)

        self.run_test(FoldHalf(), [torch.zeros(2)])

    def test_constant_fold_preserves_bool_scalar_add(self):
        class FoldBool(nn.Module):
            def __init__(self):
                super().__init__()
                self.constant = nn.Parameter(
                    torch.tensor([True, False]), requires_grad=False
                )

            def forward(self, x):
                value = torch.ops.aten.add.Scalar(self.constant, True)
                return torch.ops.aten.add.Tensor(x, value)

        self.run_test(FoldBool(), [torch.zeros(2)])

    @parameterized.expand([(lambda x, y: torch.ops.aten.div.Tensor(x, y),)])
    def test_elementwise_op_div_with_two_ints(self, orig_op: Callable):
        class TestModule(nn.Module):
            def __init__(self, orig_op):
                super().__init__()
                self.orig_op = orig_op

            def forward(self, x):
                return self.orig_op(x, x + 1)

        m = TestModule(orig_op)
        inputs = [torch.randint(1, 10, (5,), dtype=torch.int32)]
        self.run_test(m, inputs)

    @parameterized.expand([(lambda x, y: torch.ops.aten.div.Tensor(x, y),)])
    def test_elementwise_op_div_with_one_int_one_constant(self, orig_op: Callable):
        class TestModule(nn.Module):
            def __init__(self, orig_op):
                super().__init__()
                self.constant1 = torch.nn.Parameter(
                    torch.randn(
                        5,
                    )
                )
                self.orig_op = orig_op

            def forward(self, x):
                return self.orig_op(x, self.constant1)

        m = TestModule(orig_op)
        inputs = [torch.randint(1, 10, (5,), dtype=torch.int32)]
        self.run_test(m, inputs)

    # Dynamic shape test
    @parameterized.expand(
        [
            (
                f"no_broadcast_{op[0].__name__}",
                (-1, -1),
                ((1, 1), (2, 2), (3, 3)),
                (-1, -1),
                ((1, 1), (2, 2), (3, 3)),
                op[0],
            )
            for op in elementwise_ops
        ]
        + [
            (
                f"broadcast_{op[0].__name__}",
                (-1, -1, -1),
                ((1, 1, 1), (2, 2, 2), (3, 3, 3)),
                (-1, -1),
                ((1, 1), (2, 2), (3, 3)),
                op[0],
            )
            for op in elementwise_ops
        ]
    )
    def test_elementwise_op_with_dynamic_shape(
        self, _, x_shape, x_shape_ranges, y_shape, y_shape_ranges, orig_op
    ):
        class Op(nn.Module):
            def forward(self, x, y):
                return orig_op(x, y)

        input_specs = [
            Input(
                shape=x_shape,
                dtype=torch.float32,
                shape_ranges=[x_shape_ranges],
            ),
            Input(
                shape=y_shape,
                dtype=torch.float32,
                shape_ranges=[y_shape_ranges],
            ),
        ]
        self.run_test_with_dynamic_shape(Op(), input_specs)

    @parameterized.expand(
        [
            (
                f"no_broadcast_{op[0].__name__}",
                op[0],
            )
            for op in elementwise_ops
        ]
        + [
            (
                f"broadcast_{op[0].__name__}",
                op[0],
            )
            for op in elementwise_ops
        ]
    )
    def test_elementwise_op_with_dynamic_shape_four_dimensions(self, _, orig_op):
        class Op(nn.Module):
            def forward(self, x, y):
                return orig_op(x, y)

        input_specs = [
            Input(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (3, 3, 3, 3), (5, 5, 5, 5))],
            ),
            Input(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (3, 3, 3, 3), (5, 5, 5, 5))],
            ),
        ]
        self.run_test_with_dynamic_shape(Op(), input_specs)

    @parameterized.expand(
        [
            (f"bf16_{op[0].__name__}_one_constant", op[0])
            for op in elementwise_ops
            if op[0].__name__ not in ["pow.Tensor_Tensor", "fmod.Tensor"]
        ]
    )
    def test_elementwise_ops_bf16(self, _, orig_op):
        class TestModule(nn.Module):
            def __init__(self, orig_op):
                super().__init__()
                self.constant = torch.randn(1)
                self.orig_op = orig_op

            def forward(self, x):
                x = self.orig_op(x, self.constant)
                return self.orig_op(x, -2)

        m = TestModule(orig_op)
        inputs = [torch.randn(2, 2, dtype=torch.bfloat16)]
        self.run_test(m, inputs)

    def test_elementwise_mul_zerodim_fp32_against_fp16_tensor(self):
        # Regression for pytorch/TensorRT#4265: a 0-dim fp32 scalar (e.g.
        # an nn.Parameter holding `torch.tensor(1.0)`) multiplied with an
        # Nd fp16 tensor must produce fp16 (PyTorch's weak-ZeroDim rule),
        # not fp32 from torch.promote_types. The previous strong-promotion
        # path forced fp32 through downstream ops and broke the next
        # type-strict layer (here MatMul) with `A=Float, B=Half`.
        class ZeroDimMulThenMatmul(nn.Module):
            def forward(self, alpha_0d_fp32, x_fp16, weight_fp16):
                scaled = alpha_0d_fp32 * x_fp16
                return torch.matmul(scaled, weight_fp16)

        inputs = [
            torch.tensor(1.0, dtype=torch.float32),
            torch.randn(2, 4, 8, dtype=torch.float16),
            torch.randn(8, 8, dtype=torch.float16),
        ]
        self.run_test(ZeroDimMulThenMatmul(), inputs, use_dynamo_tracer=True)


class TestConstantFolding(unittest.TestCase):
    @parameterized.expand(
        [
            ("add", trt.ElementWiseOperation.SUM, operator.add),
            ("sub", trt.ElementWiseOperation.SUB, operator.sub),
            ("mul", trt.ElementWiseOperation.PROD, operator.mul),
            ("div", trt.ElementWiseOperation.DIV, operator.truediv),
            ("floor_div", trt.ElementWiseOperation.FLOOR_DIV, operator.floordiv),
            ("pow", trt.ElementWiseOperation.POW, operator.pow),
            ("eq", trt.ElementWiseOperation.EQUAL, operator.eq),
            ("gt", trt.ElementWiseOperation.GREATER, operator.gt),
            ("lt", trt.ElementWiseOperation.LESS, operator.lt),
            ("max", trt.ElementWiseOperation.MAX, max),
            ("min", trt.ElementWiseOperation.MIN, min),
        ]
    )
    def test_python_scalars(self, _, operation, eager):
        for lhs, rhs in ((2, 3), (2147483647, 1), (2147483648, 2), (1.0000000001, 1.0)):
            with self.subTest(lhs=lhs, rhs=rhs):
                expected = eager(lhs, rhs)
                result = elementwise_base._fold_constants(operation, lhs, rhs)
                self.assertIs(type(result), type(expected))
                self.assertEqual(result, expected)

    @parameterized.expand(
        [
            ("add", trt.ElementWiseOperation.SUM, operator.add),
            ("sub", trt.ElementWiseOperation.SUB, operator.sub),
            ("mul", trt.ElementWiseOperation.PROD, operator.mul),
            ("div", trt.ElementWiseOperation.DIV, operator.truediv),
            ("floor_div", trt.ElementWiseOperation.FLOOR_DIV, operator.floordiv),
            ("pow", trt.ElementWiseOperation.POW, operator.pow),
            ("eq", trt.ElementWiseOperation.EQUAL, operator.eq),
            ("gt", trt.ElementWiseOperation.GREATER, operator.gt),
            ("lt", trt.ElementWiseOperation.LESS, operator.lt),
        ]
    )
    def test_mixed_scalar_promotion(self, _, operation, eager):
        for dtype in (torch.float16, torch.bfloat16, torch.int32):
            for shape in ((), (2,)):
                for scalar in (2, 1.0001):
                    for scalar_first in (False, True):
                        with self.subTest(
                            dtype=dtype,
                            shape=shape,
                            scalar=scalar,
                            scalar_first=scalar_first,
                        ):
                            tensor = torch.full(shape, 3, dtype=dtype)
                            lhs, rhs = (
                                (scalar, tensor) if scalar_first else (tensor, scalar)
                            )
                            torch.testing.assert_close(
                                elementwise_base._fold_constants(operation, lhs, rhs),
                                eager(lhs, rhs),
                                rtol=0,
                                atol=0,
                            )

    def test_bool_scalar_add(self):
        tensor = torch.tensor([True, False])
        for lhs, rhs in ((tensor, True), (True, tensor)):
            torch.testing.assert_close(
                elementwise_base._fold_constants(
                    trt.ElementWiseOperation.SUM, lhs, rhs
                ),
                lhs + rhs,
            )

    def test_float16_overflow(self):
        tensor = torch.tensor([300.0], dtype=torch.float16)
        for lhs, rhs in ((tensor, 1000.0), (1000.0, tensor)):
            torch.testing.assert_close(
                elementwise_base._fold_constants(
                    trt.ElementWiseOperation.PROD, lhs, rhs
                ),
                lhs * rhs,
            )

    @parameterized.expand(
        [
            ("max", trt.ElementWiseOperation.MAX, torch.maximum),
            ("min", trt.ElementWiseOperation.MIN, torch.minimum),
            ("and", trt.ElementWiseOperation.AND, torch.logical_and),
            ("or", trt.ElementWiseOperation.OR, torch.logical_or),
            ("xor", trt.ElementWiseOperation.XOR, torch.logical_xor),
        ]
    )
    def test_tensor_operations(self, _, operation, eager):
        lhs = torch.tensor([[0.0], [3.0]])
        rhs = torch.tensor([2.0, 0.0, 4.0])
        torch.testing.assert_close(
            elementwise_base._fold_constants(operation, lhs, rhs), eager(lhs, rhs)
        )
        torch.testing.assert_close(
            elementwise_base._fold_constants(operation, lhs.numpy(), rhs.numpy()),
            eager(lhs, rhs),
        )
        scalar_tensor = torch.tensor(2.0, dtype=torch.float16)
        for left, right in ((scalar_tensor, 3.0), (3.0, scalar_tensor)):
            torch.testing.assert_close(
                elementwise_base._fold_constants(operation, left, right),
                eager(
                    torch.as_tensor(left, dtype=torch.float16),
                    torch.as_tensor(right, dtype=torch.float16),
                ),
            )


if __name__ == "__main__":
    run_tests()
