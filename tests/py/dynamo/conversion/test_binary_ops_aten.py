from typing import Callable

import torch
import torch.nn as nn
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import Input

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


if __name__ == "__main__":
    run_tests()
