from typing import Callable

import torch
import pytest
import torch.nn as nn

from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.fx.tools.common_fx2trt import DispatchTestCase, InputTensorSpec

NEED_TEST_BOTH_CONSTANTS_CASE = True

elementwise_ops = [
    ((lambda x, y: x + y), torch.ops.aten.add.Tensor, NEED_TEST_BOTH_CONSTANTS_CASE),
    (
        (lambda x, y: torch.add(x, y)),
        torch.ops.aten.add.Tensor,
        NEED_TEST_BOTH_CONSTANTS_CASE,
    ),
    ((lambda x, y: x.add(y)), torch.ops.aten.add.Tensor, NEED_TEST_BOTH_CONSTANTS_CASE),
    ((lambda x, y: x - y), torch.ops.aten.sub.Tensor, NEED_TEST_BOTH_CONSTANTS_CASE),
    ((lambda x, y: torch.sub(x, y)), torch.ops.aten.sub.Tensor, False),
    ((lambda x, y: x.sub(y)), torch.ops.aten.sub.Tensor, False),
    ((lambda x, y: x / y), torch.ops.aten.div.Tensor, NEED_TEST_BOTH_CONSTANTS_CASE),
    (
        (lambda x, y: x // y),
        torch.ops.aten.floor_divide.default,
        NEED_TEST_BOTH_CONSTANTS_CASE,
    ),
    (
        (lambda x, y: torch.div(x, y, rounding_mode="trunc")),
        torch.ops.aten.div.Tensor_mode,
        not NEED_TEST_BOTH_CONSTANTS_CASE,
    ),
    (
        (lambda x, y: torch.div(x, y, rounding_mode="floor")),
        torch.ops.aten.div.Tensor_mode,
        NEED_TEST_BOTH_CONSTANTS_CASE,
    ),
    (
        (lambda x, y: torch.div(x, y)),
        torch.ops.aten.div.Tensor,
        NEED_TEST_BOTH_CONSTANTS_CASE,
    ),
    (
        (lambda x, y: torch.fmod(x, y)),
        torch.ops.aten.fmod.Tensor,
        not NEED_TEST_BOTH_CONSTANTS_CASE,
    ),
    ## torch.floor_divide rounds result toward zero, rather than -Inf.
    ## https://github.com/pytorch/pytorch/issues/43874
    (
        (lambda x, y: torch.floor_divide(x, y)),
        torch.ops.aten.floor_divide.default,
        not NEED_TEST_BOTH_CONSTANTS_CASE,
    ),
    ((lambda x, y: x * y), torch.ops.aten.mul.Tensor, NEED_TEST_BOTH_CONSTANTS_CASE),
    (torch.pow, torch.ops.aten.pow.Tensor_Tensor, not NEED_TEST_BOTH_CONSTANTS_CASE),
]


@pytest.mark.xfail(reason="Nondeterministic Dynamo errors", raises=AssertionError)
class TestBinaryOpConverters(DispatchTestCase):
    @parameterized.expand([(op[1].__name__, op[0], op[1]) for op in elementwise_ops])
    def test_elementwise_ops(self, name, orig_op: Callable, expected_op):
        class TestModule(nn.Module):
            def __init__(self, orig_op):
                super().__init__()
                self.orig_op = orig_op

            def forward(self, x):
                return self.orig_op(x, x)

        m = TestModule(orig_op)
        # Avoid dividing by 0.
        inputs = [torch.rand(1, 1) + 1]
        self.run_test(m, inputs, expected_ops={expected_op})

    @parameterized.expand([(op[1].__name__, op[0], op[1]) for op in elementwise_ops])
    def test_elementwise_ops_with_one_constant(
        self, name, orig_op: Callable, expected_op
    ):
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
        self.run_test(m, inputs, expected_ops={expected_op})

    @parameterized.expand(
        [(op[1].__name__, op[0], op[1]) for op in elementwise_ops if op[2]]
    )
    def test_elementwise_op_with_both_constants(
        self, name, orig_op: Callable, expected_op
    ):
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
        self.run_test(m, inputs, expected_ops={expected_op})

    # Dynamic shape test
    @parameterized.expand(
        [
            (
                f"no_broadcast_{op[1].__name__}",
                (-1, -1),
                ((1, 1), (2, 2), (3, 3)),
                (-1, -1),
                ((1, 1), (2, 2), (3, 3)),
                op[0],
                op[1],
            )
            for op in elementwise_ops
        ]
        + [
            (
                f"broadcast_{op[1].__name__}",
                (-1, -1, -1),
                ((1, 1, 1), (2, 2, 2), (3, 3, 3)),
                (-1, -1),
                ((1, 1), (2, 2), (3, 3)),
                op[0],
                op[1],
            )
            for op in elementwise_ops
        ]
    )
    def test_elementwise_op_with_dynamic_shape(
        self, _, x_shape, x_shape_ranges, y_shape, y_shape_ranges, orig_op, expected_op
    ):
        class Op(nn.Module):
            def forward(self, x, y):
                return orig_op(x, y)

        input_specs = [
            InputTensorSpec(
                shape=x_shape,
                dtype=torch.float32,
                shape_ranges=[x_shape_ranges],
            ),
            InputTensorSpec(
                shape=y_shape,
                dtype=torch.float32,
                shape_ranges=[y_shape_ranges],
            ),
        ]
        self.run_test_with_dynamic_shape(Op(), input_specs, expected_ops={expected_op})

    @parameterized.expand(
        [
            (
                f"no_broadcast_{op[1].__name__}",
                op[0],
                op[1],
            )
            for op in elementwise_ops
        ]
        + [
            (
                f"broadcast_{op[1].__name__}",
                op[0],
                op[1],
            )
            for op in elementwise_ops
        ]
    )
    def test_elementwise_op_with_dynamic_shape_four_dimensions(
        self, _, orig_op, expected_op
    ):
        class Op(nn.Module):
            def forward(self, x, y):
                return orig_op(x, y)

        input_specs = [
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (3, 3, 3, 3), (5, 5, 5, 5))],
            ),
            InputTensorSpec(
                shape=(-1, -1, -1, -1),
                dtype=torch.float32,
                shape_ranges=[((1, 1, 1, 1), (3, 3, 3, 3), (5, 5, 5, 5))],
            ),
        ]
        self.run_test_with_dynamic_shape(Op(), input_specs, expected_ops={expected_op})


if __name__ == "__main__":
    run_tests()
