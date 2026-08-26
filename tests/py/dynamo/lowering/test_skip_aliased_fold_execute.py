import unittest

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.constant_folding import (
    _MAX_CONSTANT_FOLD_BYTES,
    _TorchTensorRTConstantFolder,
    constant_fold,
    skip_large_aliased_view_fold,
)


def _exported_gm(model: torch.nn.Module, example: torch.Tensor) -> torch.fx.GraphModule:
    return torch.export.export(model, (example,)).module()


def _permute_nodes(gm: torch.fx.GraphModule) -> list[torch.fx.Node]:
    return [
        node for node in gm.graph.nodes if node.target is torch.ops.aten.permute.default
    ]


def _call_targets(gm: torch.fx.GraphModule) -> set[object]:
    return {node.target for node in gm.graph.nodes if node.op == "call_function"}


class _WeightPermute(torch.nn.Module):
    def __init__(self, rows: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(rows, rows))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.weight.permute(1, 0)


class _WeightAdd(torch.nn.Module):
    def __init__(self, rows: int = 32) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(rows, rows))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + (self.weight + self.weight)


class _WeightContiguous(torch.nn.Module):
    def __init__(self, rows: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(rows, rows))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # Tensor.contiguous() is DCE'd by export when the param is already
        # contiguous; the ATen op stays in the graph.
        return value + torch.ops.aten.contiguous.default(self.weight)


class _CountingFolder(_TorchTensorRTConstantFolder):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.executed_targets: list[object] = []

    def run_node(self, node: torch.fx.Node) -> object:
        skip_fn = getattr(self, "skip_folding_node_fn", None)
        skipping = skip_fn is not None and node.op == "call_function" and skip_fn(node)
        if node.op == "call_function" and not skipping:
            self.executed_targets.append(node.target)
        return super().run_node(node)


class TestSkipAliasedFoldExecute(unittest.TestCase):
    def test_skip_fn_true_for_large_permute(self) -> None:
        rows = 1024
        self.assertGreater(rows * rows * 4, _MAX_CONSTANT_FOLD_BYTES)
        gm = _exported_gm(_WeightPermute(rows), torch.ones(rows, rows))
        permute_nodes = _permute_nodes(gm)
        self.assertEqual(len(permute_nodes), 1)
        self.assertTrue(skip_large_aliased_view_fold(gm, permute_nodes[0]))

    def test_skip_fn_false_for_small_permute(self) -> None:
        gm = _exported_gm(_WeightPermute(8), torch.ones(8, 8))
        permute_nodes = _permute_nodes(gm)
        self.assertEqual(len(permute_nodes), 1)
        self.assertFalse(skip_large_aliased_view_fold(gm, permute_nodes[0]))

    def test_skip_fn_false_for_large_contiguous(self) -> None:
        rows = 1024
        gm = _exported_gm(_WeightContiguous(rows), torch.ones(rows, rows))
        contiguous_nodes = [
            node
            for node in gm.graph.nodes
            if node.target is torch.ops.aten.contiguous.default
        ]
        self.assertEqual(len(contiguous_nodes), 1)
        self.assertFalse(skip_large_aliased_view_fold(gm, contiguous_nodes[0]))

    def test_large_permute_is_not_executed(self) -> None:
        gm = _exported_gm(_WeightPermute(1024), torch.ones(1024, 1024))
        permute_node = _permute_nodes(gm)[0]
        folder = _CountingFolder(
            gm,
            skip_constructors=False,
            skip_folding_node_fn=lambda node: skip_large_aliased_view_fold(gm, node),
        )
        folder.run()
        self.assertNotIn(torch.ops.aten.permute.default, folder.executed_targets)
        self.assertNotIn(permute_node, folder.node_replacements)

    def test_small_permute_is_executed(self) -> None:
        gm = _exported_gm(_WeightPermute(8), torch.ones(8, 8))
        permute_node = _permute_nodes(gm)[0]
        folder = _CountingFolder(
            gm,
            skip_constructors=False,
            skip_folding_node_fn=lambda node: skip_large_aliased_view_fold(gm, node),
        )
        folder.run()
        self.assertIn(torch.ops.aten.permute.default, folder.executed_targets)
        self.assertIn(permute_node, folder.node_replacements)

    def test_constant_fold_keeps_large_permute_in_graph(self) -> None:
        gm = _exported_gm(_WeightPermute(1024), torch.ones(1024, 1024))
        folded = constant_fold(gm, CompilationSettings())
        self.assertIn(torch.ops.aten.permute.default, _call_targets(folded))

    def test_constant_fold_still_folds_small_permute(self) -> None:
        model = _WeightPermute(8)
        example = torch.ones(8, 8)
        gm = _exported_gm(model, example)
        folded = constant_fold(gm, CompilationSettings())
        self.assertNotIn(torch.ops.aten.permute.default, _call_targets(folded))
        torch.testing.assert_close(folded(example), model(example))

    def test_materialized_weight_add_still_folds(self) -> None:
        model = _WeightAdd()
        example = torch.ones(32, 32)
        gm = _exported_gm(model, example)
        adds_before = sum(
            1 for node in gm.graph.nodes if node.target is torch.ops.aten.add.Tensor
        )
        folded = constant_fold(gm, CompilationSettings())
        adds_after = sum(
            1 for node in folded.graph.nodes if node.target is torch.ops.aten.add.Tensor
        )
        self.assertEqual(adds_before, 2)
        self.assertEqual(adds_after, 1)
        torch.testing.assert_close(folded(example), model(example))


if __name__ == "__main__":
    unittest.main()
