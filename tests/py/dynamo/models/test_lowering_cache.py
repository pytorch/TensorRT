import os
import tempfile
import unittest

import torch
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo._lowering_cache import (
    DiskLoweringCache,
    LoweringCacheEntry,
)
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering import post_lowering, pre_export_lowering


class _Linear(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 4))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value @ self.weight


class TestDiskLoweringCache(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = (torch.randn(2, 4),)
        self.input_specs = (Input(self.inputs[0].shape, dtype=torch.float32),)
        self.settings = CompilationSettings(
            require_full_compilation=True,
            use_fast_partitioner=True,
        )

    def test_key_is_stable_for_same_exported_program(self) -> None:
        exported_program = torch.export.export(_Linear().eval(), self.inputs)

        first = DiskLoweringCache.get_hash(
            exported_program, self.input_specs, {}, self.settings
        )
        second = DiskLoweringCache.get_hash(
            exported_program, self.input_specs, {}, self.settings
        )

        self.assertEqual(first, second)

    def test_key_is_stable_for_scalar_buffers(self) -> None:
        # 0-dim tensors cannot be viewed as uint8; hashing must not fall back to
        # pickling the tensor, whose bytes differ between equal tensors.
        class _ScalarBuffer(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("scale", torch.tensor(67.2000045776367))

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return value * self.scale

        first = DiskLoweringCache.get_hash(
            torch.export.export(_ScalarBuffer().eval(), self.inputs),
            self.input_specs,
            {},
            self.settings,
        )
        second = DiskLoweringCache.get_hash(
            torch.export.export(_ScalarBuffer().eval(), self.inputs),
            self.input_specs,
            {},
            self.settings,
        )

        self.assertEqual(first, second)

    def test_key_changes_with_weights(self) -> None:
        first_program = torch.export.export(_Linear().eval(), self.inputs)
        second_model = _Linear().eval()
        with torch.no_grad():
            second_model.weight.add_(1)
        second_program = torch.export.export(second_model, self.inputs)

        first = DiskLoweringCache.get_hash(
            first_program, self.input_specs, {}, self.settings
        )
        second = DiskLoweringCache.get_hash(
            second_program, self.input_specs, {}, self.settings
        )

        self.assertNotEqual(first, second)

    def test_round_trip(self) -> None:
        exported_program = torch.export.export(_Linear().eval(), self.inputs)
        exported_program = pre_export_lowering(exported_program, self.settings)
        graph_module = post_lowering(
            exported_program.run_decompositions({}).module(), self.settings
        )
        entry = LoweringCacheEntry(graph_module, ())

        with tempfile.TemporaryDirectory() as directory:
            cache = DiskLoweringCache(directory)
            restored_module = cache.save("a" * 64, entry)
            loaded = cache.load("a" * 64)

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertIsInstance(loaded.lowered_module, torch.fx.GraphModule)
        torch.testing.assert_close(
            loaded.lowered_module(*self.inputs), restored_module(*self.inputs)
        )
        self.assertEqual(
            [node.name for node in loaded.lowered_module.graph.nodes],
            [node.name for node in restored_module.graph.nodes],
        )
        placeholder = next(
            node
            for node in loaded.lowered_module.graph.nodes
            if node.op == "placeholder"
        )
        self.assertIn("val", placeholder.meta)

    def test_artifact_is_torch_saved_graph_module_not_exported_program(self) -> None:
        exported_program = torch.export.export(_Linear().eval(), self.inputs)
        exported_program = pre_export_lowering(exported_program, self.settings)
        graph_module = post_lowering(
            exported_program.run_decompositions({}).module(), self.settings
        )

        with tempfile.TemporaryDirectory() as directory:
            cache = DiskLoweringCache(directory)
            cache.save("b" * 64, LoweringCacheEntry(graph_module, ()))
            artifact = os.path.join(directory, "b" * 2, "b" * 64, "lowered.pt")
            self.assertTrue(os.path.exists(artifact))
            self.assertLess(os.path.getsize(artifact), 1_000_000)
            self.assertFalse(
                os.path.exists(os.path.join(directory, "b" * 2, "b" * 64, "lowered.ep"))
            )

    def test_round_trip_uint8_and_scalar_buffers(self) -> None:
        class _Packed(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "packed", torch.randint(0, 255, (8, 8), dtype=torch.uint8)
                )
                self.register_buffer("scale", torch.tensor(0.025))

            def forward(self, value: torch.Tensor) -> torch.Tensor:
                return (
                    value
                    + self.packed[: value.shape[0], : value.shape[1]].float()
                    * self.scale
                )

        inputs = (torch.randn(2, 4),)
        exported_program = torch.export.export(_Packed().eval(), inputs)
        exported_program = pre_export_lowering(exported_program, self.settings)
        graph_module = post_lowering(
            exported_program.run_decompositions({}).module(), self.settings
        )

        with tempfile.TemporaryDirectory() as directory:
            cache = DiskLoweringCache(directory)
            restored = cache.save("c" * 64, LoweringCacheEntry(graph_module, ()))
            loaded = cache.load("c" * 64)

        self.assertIsNotNone(loaded)
        assert loaded is not None
        torch.testing.assert_close(loaded.lowered_module(*inputs), restored(*inputs))

    def test_bypasses_non_full_compilation(self) -> None:
        self.assertFalse(DiskLoweringCache.can_cache(CompilationSettings()))


if __name__ == "__main__":
    unittest.main()
