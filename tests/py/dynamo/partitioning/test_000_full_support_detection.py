import copy

import torch
from parameterized import parameterized
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo import partitioning
from torch_tensorrt.dynamo.lowering import (
    get_decompositions,
    post_lowering,
    pre_export_lowering,
)

PARTITIONERS = [
    ("fast", partitioning.fast_partition),
    ("global", partitioning.global_partition),
    ("hierarchical", partitioning.hierarchical_adjacency_partition),
]


class TestFullSupportDetection(TestCase):
    """A refused operator must never read as fully supported.

    An operator with side effects, such as a random or in-place op, that has no converter
    is kept in PyTorch. unsupported_operators does not record it, since that dictionary
    excludes impure nodes on purpose. fallback_operators records it instead, so
    require_full_compilation must consult both when it decides whether a model is fully
    supported. Otherwise a model that must run a random op in PyTorch would compile under
    require_full_compilation=True.
    """

    @staticmethod
    def _lower(module, args):
        exported = torch.export.export(module.eval().cuda(), args)
        lowered = exported.run_decompositions(get_decompositions(False))
        return post_lowering(pre_export_lowering(lowered).module())

    @staticmethod
    def _six_linear_layers():
        return torch.nn.ModuleList([torch.nn.Linear(64, 64) for _ in range(6)])

    @classmethod
    def _impure_refusal_module(cls):
        class WithImpureRefusal(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = cls._six_linear_layers()

            def forward(self, x):
                out = x
                for index, layer in enumerate(self.layers):
                    out = torch.relu(layer(out))
                    if index == 2:
                        # No converter, and impure, so it is kept in PyTorch.
                        out = out + torch.normal(
                            0.0,
                            1.0,
                            size=out.shape,
                            device=out.device,
                            dtype=out.dtype,
                        )
                return out

        return WithImpureRefusal()

    @classmethod
    def _fully_supported_module(cls):
        class FullySupported(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = cls._six_linear_layers()

            def forward(self, x):
                out = x
                for layer in self.layers:
                    out = torch.relu(layer(out))
                return out

        return FullySupported()

    @staticmethod
    def _partition(partition_fn, graph_module, **kwargs):
        if partition_fn is partitioning.hierarchical_adjacency_partition:
            kwargs["backend_priority"] = ["tensorrt"]
        # Both partitioners mutate the module they are given, so hand each a copy.
        return partition_fn(copy.deepcopy(graph_module), **kwargs)

    @parameterized.expand(PARTITIONERS)
    def test_impure_refusal_is_not_fully_supported(self, _, partition_fn):
        graph_module = self._lower(
            self._impure_refusal_module(), (torch.randn(8, 64, device="cuda"),)
        )
        with self.assertRaisesRegex(AssertionError, "not fully supported"):
            self._partition(
                partition_fn,
                graph_module,
                min_block_size=1,
                require_full_compilation=True,
            )

    @parameterized.expand(PARTITIONERS)
    def test_impure_refusal_is_recorded_as_fallback(self, _, partition_fn):
        """The refused impure operator is recorded in fallback_operators, not in
        unsupported_operators. unsupported_operators keeps excluding impure nodes, which is
        the contract the fallback reporting relies on."""
        graph_module = self._lower(
            self._impure_refusal_module(), (torch.randn(8, 64, device="cuda"),)
        )
        _, support = self._partition(partition_fn, graph_module, min_block_size=1)
        self.assertTrue(
            support.fallback_operators,
            "the refused impure operator was not recorded in fallback_operators, so "
            "require_full_compilation cannot see it",
        )

    @parameterized.expand(PARTITIONERS)
    def test_pure_refusal_is_not_fully_supported(self, _, partition_fn):
        """A refused pure operator was already caught. Keep it that way."""
        graph_module = self._lower(
            self._fully_supported_module(), (torch.randn(8, 64, device="cuda"),)
        )
        with self.assertRaisesRegex(AssertionError, "not fully supported"):
            self._partition(
                partition_fn,
                graph_module,
                min_block_size=1,
                require_full_compilation=True,
                torch_executed_ops={"torch.ops.aten.relu.default"},
            )

    @parameterized.expand(PARTITIONERS)
    def test_fully_supported_module_is_accepted(self, name, partition_fn):
        """Guards against over correction. This passes before the change too, so it does
        not prove the fix; it proves the fix did not start rejecting good graphs."""
        graph_module = self._lower(
            self._fully_supported_module(), (torch.randn(8, 64, device="cuda"),)
        )
        partitioned, support = self._partition(
            partition_fn,
            graph_module,
            min_block_size=1,
            require_full_compilation=True,
        )
        self.assertFalse(support.unsupported_operators)
        self.assertFalse(support.fallback_operators)
        blocks = [child for child, _ in partitioned.named_children()]
        self.assertTrue(
            any("_run_on_acc" in block for block in blocks),
            f"expected an accelerated block, got {blocks}",
        )
        # The global partitioner leaves a refused node inline in the parent rather than
        # naming a torch block, so assert on what is left outside the block instead.
        remaining = [
            node.name
            for node in partitioned.graph.nodes
            if node.op == "call_function" and "_run_on_acc" not in node.name
        ]
        self.assertEqual(
            remaining,
            [],
            f"expected no operator left outside the engine, got {remaining}",
        )


if __name__ == "__main__":
    run_tests()
