import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
    bitwise_type_validator,
)


def _make_bitwise_node(target, lhs_shape, rhs_shape):
    def make_operand(shape):
        operand = MagicMock()
        operand.meta = {
            "tensor_meta": SimpleNamespace(dtype=torch.bool, shape=torch.Size(shape))
        }
        return operand

    node = MagicMock()
    node.target = target
    node.args = (make_operand(lhs_shape), make_operand(rhs_shape))
    return node


def _make_bitwise_scalar_node(target, tensor_shape, scalar, scalar_first=False):
    operand = MagicMock()
    operand.meta = {
        "tensor_meta": SimpleNamespace(dtype=torch.bool, shape=torch.Size(tensor_shape))
    }

    node = MagicMock()
    node.target = target
    node.args = (scalar, operand) if scalar_first else (operand, scalar)
    return node


class TestBitwiseValidator(unittest.TestCase):
    def test_bitwise_and_scalar_tensor_other_falls_back(self):
        node = _make_bitwise_node(torch.ops.aten.bitwise_and.Tensor, (2, 3), ())
        self.assertFalse(bitwise_type_validator(node))

    def test_bitwise_or_scalar_tensor_other_falls_back(self):
        node = _make_bitwise_node(torch.ops.aten.bitwise_or.Tensor, (2, 3), ())
        self.assertFalse(bitwise_type_validator(node))

    def test_bitwise_and_non_scalar_tensor_other_is_supported(self):
        node = _make_bitwise_node(torch.ops.aten.bitwise_and.Tensor, (2, 3), (2, 3))
        self.assertTrue(bitwise_type_validator(node))

    def test_bitwise_xor_scalar_tensor_other_is_supported(self):
        node = _make_bitwise_node(torch.ops.aten.bitwise_xor.Tensor, (2, 3), ())
        self.assertTrue(bitwise_type_validator(node))

    # A scalar operand only breaks when its value fixes the result of the op:
    # False for AND, True for OR. The other value, and XOR either way, stay in
    # TensorRT.
    def test_bitwise_and_false_scalar_falls_back(self):
        for target, scalar_first in (
            (torch.ops.aten.bitwise_and.Scalar, False),
            (torch.ops.aten.bitwise_and.Scalar_Tensor, True),
        ):
            with self.subTest(target=target):
                node = _make_bitwise_scalar_node(target, (2, 3), False, scalar_first)
                self.assertFalse(bitwise_type_validator(node))

    def test_bitwise_and_true_scalar_is_supported(self):
        for target, scalar_first in (
            (torch.ops.aten.bitwise_and.Scalar, False),
            (torch.ops.aten.bitwise_and.Scalar_Tensor, True),
        ):
            with self.subTest(target=target):
                node = _make_bitwise_scalar_node(target, (2, 3), True, scalar_first)
                self.assertTrue(bitwise_type_validator(node))

    def test_bitwise_or_true_scalar_falls_back(self):
        for target, scalar_first in (
            (torch.ops.aten.bitwise_or.Scalar, False),
            (torch.ops.aten.bitwise_or.Scalar_Tensor, True),
        ):
            with self.subTest(target=target):
                node = _make_bitwise_scalar_node(target, (2, 3), True, scalar_first)
                self.assertFalse(bitwise_type_validator(node))

    def test_bitwise_or_false_scalar_is_supported(self):
        for target, scalar_first in (
            (torch.ops.aten.bitwise_or.Scalar, False),
            (torch.ops.aten.bitwise_or.Scalar_Tensor, True),
        ):
            with self.subTest(target=target):
                node = _make_bitwise_scalar_node(target, (2, 3), False, scalar_first)
                self.assertTrue(bitwise_type_validator(node))

    def test_bitwise_xor_scalar_is_supported(self):
        for target, scalar_first in (
            (torch.ops.aten.bitwise_xor.Scalar, False),
            (torch.ops.aten.bitwise_xor.Scalar_Tensor, True),
        ):
            for scalar in (True, False):
                with self.subTest(target=target, scalar=scalar):
                    node = _make_bitwise_scalar_node(
                        target, (2, 3), scalar, scalar_first
                    )
                    self.assertTrue(bitwise_type_validator(node))


if __name__ == "__main__":
    run_tests()
