import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from torch.testing._internal.common_utils import run_tests
from torch_tensorrt.dynamo.conversion.aten_ops_converters import cat_validator


def _make_cat_node(shapes, dim=0):
    """A cat node whose operands carry only the metadata the validator reads."""
    operands = []
    for shape in shapes:
        operand = MagicMock()
        operand.meta = {
            "tensor_meta": SimpleNamespace(shape=torch.Size(shape), dtype=torch.float32)
        }
        operands.append(operand)
    node = MagicMock()
    node.args = (operands, dim)
    node.kwargs = {}
    return node


class TestCatValidator(unittest.TestCase):
    """Metadata-only checks need no GPU and live outside the converter harness."""

    def test_rank1_empty_with_agreeing_survivors_is_accepted(self):
        """The DynamicCache shape. TensorRT cannot take mixed ranks, but the empty
        operand holds nothing, so dropping it leaves operands that agree."""
        self.assertTrue(cat_validator(_make_cat_node([(0,), (1, 2, 8, 4)])))
        self.assertTrue(cat_validator(_make_cat_node([(0,), (2, 3), (5, 3)])))

    def test_disagreeing_survivors_are_refused(self):
        """Dropping the empty operand must not paper over a real mismatch."""
        self.assertFalse(cat_validator(_make_cat_node([(0,), (2, 3), (2, 3, 4)])))

    def test_uniform_ranks_are_accepted_unchanged(self):
        """Nothing is dropped here, so the answer must not depend on this change."""
        self.assertTrue(cat_validator(_make_cat_node([(2, 3), (5, 3)])))
        self.assertTrue(cat_validator(_make_cat_node([(0,), (3,)])))

    def test_single_operand_is_accepted(self):
        self.assertTrue(cat_validator(_make_cat_node([(0,)])))

    def test_unbacked_rank1_operands_do_not_guard(self):
        """Data-dependent lengths must not force a guard during validation."""

        class M(torch.nn.Module):
            def forward(self, x):
                a = torch.nonzero(x).flatten().float()
                return torch.cat([a, a * 2])

        ep = torch.export.export(M(), (torch.tensor([0, 1, 1]),), strict=False)
        cat = next(n for n in ep.graph.nodes if n.target is torch.ops.aten.cat.default)
        for operand in cat.args[0]:
            self.assertIsInstance(operand.meta["tensor_meta"].shape[0], torch.SymInt)
        self.assertTrue(cat_validator(cat))


if __name__ == "__main__":
    run_tests()
