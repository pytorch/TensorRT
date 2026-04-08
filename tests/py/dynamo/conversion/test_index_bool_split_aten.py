"""Tests for the bool/int index.Tensor converter split (commit b91168b).

Verifies that:
1. `index_has_bool_indices` validator correctly distinguishes bool vs int indices.
2. Integer-indexed `aten.index.Tensor` routes to the converter WITHOUT output allocator.
3. Boolean-indexed `aten.index.Tensor` routes to the converter WITH output allocator.
4. Both paths produce correct results.
"""

import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn
import torch_tensorrt
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt import ENABLED_FEATURES

from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
    index_has_bool_indices,
    index_nonbool_validator,
)

from .harness import DispatchTestCase


def _make_index_node(indices):
    """Create a mock FX Node whose args[1] contains index tensors with proper metadata."""
    mock_indices = []
    for idx in indices:
        if idx is None:
            mock_indices.append(None)
        else:
            mock_ind = MagicMock()
            mock_ind.meta = {"val": idx}
            mock_indices.append(mock_ind)
    node = MagicMock()
    node.args = (MagicMock(), mock_indices)
    return node


class TestIndexHasBoolIndicesValidator(unittest.TestCase):
    """Unit tests for the index_has_bool_indices validator function."""

    def test_int_indices_returns_false(self):
        node = _make_index_node([torch.tensor([0, 1, 2])])
        self.assertFalse(index_has_bool_indices(node))

    def test_bool_indices_returns_true(self):
        node = _make_index_node([torch.tensor([True, False, True])])
        self.assertTrue(index_has_bool_indices(node))

    def test_none_with_int_indices_returns_false(self):
        node = _make_index_node([None, torch.tensor([0, 1])])
        self.assertFalse(index_has_bool_indices(node))

    def test_none_with_bool_indices_returns_true(self):
        node = _make_index_node([None, torch.tensor([True, False])])
        self.assertTrue(index_has_bool_indices(node))

    def test_mixed_int_and_bool_returns_true(self):
        """If any index is bool, the function should return True."""
        node = _make_index_node([torch.tensor([0, 1]), torch.tensor([True, False])])
        self.assertTrue(index_has_bool_indices(node))

    def test_all_none_returns_false(self):
        node = _make_index_node([None, None])
        self.assertFalse(index_has_bool_indices(node))

    def test_empty_indices_returns_false(self):
        node = _make_index_node([])
        self.assertFalse(index_has_bool_indices(node))


class TestIndexNonboolValidatorConsistency(unittest.TestCase):
    """Verify index_nonbool_validator and index_has_bool_indices interact correctly."""

    def test_int_index_nonbool_true_has_bool_false(self):
        node = _make_index_node([torch.tensor([0, 1])])
        self.assertTrue(index_nonbool_validator(node))
        self.assertFalse(index_has_bool_indices(node))

    def test_bool_index_has_bool_true(self):
        node = _make_index_node([torch.tensor([True, False])])
        self.assertTrue(index_has_bool_indices(node))

    @unittest.skipUnless(
        ENABLED_FEATURES.tensorrt_rtx,
        "index_nonbool_validator only rejects bool on tensorrt_rtx",
    )
    def test_bool_index_nonbool_false_on_rtx(self):
        node = _make_index_node([torch.tensor([True, False])])
        self.assertFalse(index_nonbool_validator(node))

    @unittest.skipIf(
        ENABLED_FEATURES.tensorrt_rtx,
        "On non-RTX, index_nonbool_validator always passes",
    )
    def test_bool_index_nonbool_true_on_non_rtx(self):
        """On non-RTX, nonbool_validator passes even for bool indices;
        the bool/int split is handled by index_has_bool_indices instead."""
        node = _make_index_node([torch.tensor([True, False])])
        self.assertTrue(index_nonbool_validator(node))
        self.assertTrue(index_has_bool_indices(node))


class TestIndexIntConverterNoOutputAllocator(DispatchTestCase):
    """Integer indexing should work correctly (routed to non-output-allocator converter)."""

    @parameterized.expand(
        [
            ("int_1d_index", [torch.tensor([0, 1])], torch.randn(3, 4)),
            (
                "int_2d_with_none",
                [None, torch.tensor([0, 1])],
                torch.randn(2, 3),
            ),
            (
                "int_multi_index",
                [torch.tensor([0, 1]), torch.tensor([1, 0])],
                torch.randn(3, 3),
            ),
        ]
    )
    def test_int_index(self, _, index, input_tensor):
        class IndexModule(nn.Module):
            def forward(self, x):
                return torch.ops.aten.index.Tensor(x, index)

        self.run_test(IndexModule(), [input_tensor])


@unittest.skipIf(
    ENABLED_FEATURES.tensorrt_rtx,
    "Skipped on tensorrt_rtx due to nonzero not supported",
)
class TestIndexBoolConverterWithOutputAllocator(DispatchTestCase):
    """Boolean indexing should work correctly (routed to output-allocator converter)."""

    @parameterized.expand(
        [
            (
                "bool_1d_mask",
                [torch.tensor([True, False, True])],
                torch.randn(3, 4),
            ),
            (
                "bool_mask_with_none",
                [None, torch.tensor([True, False])],
                torch.randn(2, 2),
            ),
        ]
    )
    def test_bool_index(self, _, index, input_tensor):
        class BoolIndexModule(nn.Module):
            def forward(self, x):
                return torch.ops.aten.index.Tensor(x, index)

        self.run_test(BoolIndexModule(), [input_tensor])


if __name__ == "__main__":
    run_tests()
