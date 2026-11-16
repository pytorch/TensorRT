import unittest
from unittest.mock import patch

import torch
from torch_tensorrt.dynamo._tracer import trace
from torch_tensorrt.dynamo.lowering._export_with_decomps import (
    decomp_fingerprint,
    export_with_tensorrt_decomps,
    matching_decomp_stamp,
    maybe_run_decompositions,
)


class _InPlaceAdd(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + 1
        value = torch.ops.aten.add_.Tensor(value, value)
        return value


class TestOneShotDecomp(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = (torch.randn(4, 4),)
        self.model = _InPlaceAdd().eval()

    def test_export_stamps_matching_fingerprint(self) -> None:
        exported_program = export_with_tensorrt_decomps(
            self.model, args=self.inputs, strict=False
        )
        fingerprint = decomp_fingerprint()
        self.assertTrue(matching_decomp_stamp(exported_program, fingerprint))

    def test_maybe_run_decompositions_skips_when_stamped(self) -> None:
        exported_program = export_with_tensorrt_decomps(
            self.model, args=self.inputs, strict=False
        )
        with patch.object(
            type(exported_program),
            "run_decompositions",
            wraps=exported_program.run_decompositions,
        ) as mocked:
            maybe_run_decompositions(exported_program)
            mocked.assert_not_called()

    def test_stock_export_still_runs_decompositions(self) -> None:
        exported_program = torch.export.export(self.model, self.inputs, strict=False)
        with patch.object(
            type(exported_program),
            "run_decompositions",
            wraps=exported_program.run_decompositions,
        ) as mocked:
            maybe_run_decompositions(exported_program)
            mocked.assert_called_once()

    def test_settings_mismatch_runs_decompositions(self) -> None:
        exported_program = export_with_tensorrt_decomps(
            self.model,
            args=self.inputs,
            strict=False,
            decompose_attention=False,
        )
        with patch.object(
            type(exported_program),
            "run_decompositions",
            wraps=exported_program.run_decompositions,
        ) as mocked:
            maybe_run_decompositions(exported_program, decompose_attention=True)
            mocked.assert_called_once()

    def test_inplace_add_is_decomposed_during_export(self) -> None:
        exported_program = export_with_tensorrt_decomps(
            self.model, args=self.inputs, strict=False
        )
        targets = {
            node.target
            for node in exported_program.graph.nodes
            if node.op == "call_function"
        }
        self.assertNotIn(torch.ops.aten.add_.Tensor, targets)
        self.assertIn(torch.ops.aten.add.Tensor, targets)

    def test_trace_stamps_default_table(self) -> None:
        exported_program = trace(self.model, self.inputs)
        self.assertTrue(matching_decomp_stamp(exported_program, decomp_fingerprint()))

    def test_maybe_run_decompositions_runs_when_autocast_enabled(self) -> None:
        exported_program = export_with_tensorrt_decomps(
            self.model, args=self.inputs, strict=False
        )
        with patch.object(
            type(exported_program),
            "run_decompositions",
            wraps=exported_program.run_decompositions,
        ) as mocked:
            maybe_run_decompositions(exported_program, enable_autocast=True)
            mocked.assert_called_once()

    def test_trace_does_not_stamp_when_autocast_enabled(self) -> None:
        exported_program = trace(self.model, self.inputs, enable_autocast=True)
        self.assertFalse(matching_decomp_stamp(exported_program, decomp_fingerprint()))


class _ScatterAdd(torch.nn.Module):
    def forward(
        self, base: torch.Tensor, index: torch.Tensor, src: torch.Tensor
    ) -> torch.Tensor:
        return torch.ops.aten.scatter_add.default(base, 0, index, src)


class TestOneShotDecompShapeDependent(unittest.TestCase):
    """scatter_add decomposes only when the index extent is static, which export
    cannot know before tracing, so export defers it to compile."""

    def setUp(self) -> None:
        self.model = _ScatterAdd().eval()
        self.inputs = (
            torch.zeros((16, 8)),
            torch.zeros((4, 8), dtype=torch.int64),
            torch.ones((16, 8)),
        )

    def _targets(self, exported_program: torch.export.ExportedProgram) -> set:
        return {
            node.target
            for node in exported_program.graph.nodes
            if node.op == "call_function"
        }

    def test_symbolic_extent_exports_and_keeps_scatter_add(self) -> None:
        index_rows = torch.export.Dim("index_rows", min=1, max=16)
        exported_program = export_with_tensorrt_decomps(
            self.model,
            args=self.inputs,
            dynamic_shapes={"base": {}, "index": {0: index_rows}, "src": {}},
        )
        self.assertTrue(matching_decomp_stamp(exported_program, decomp_fingerprint()))
        with patch.object(
            type(exported_program),
            "run_decompositions",
            wraps=exported_program.run_decompositions,
        ) as mocked:
            lowered = maybe_run_decompositions(exported_program)
            mocked.assert_not_called()
        self.assertIn(torch.ops.aten.scatter_add.default, self._targets(lowered))

    def test_static_extent_decomposes_scatter_add_in_compile(self) -> None:
        exported_program = export_with_tensorrt_decomps(self.model, args=self.inputs)
        self.assertIn(
            torch.ops.aten.scatter_add.default, self._targets(exported_program)
        )
        with patch.object(
            type(exported_program),
            "run_decompositions",
            wraps=exported_program.run_decompositions,
        ) as mocked:
            lowered = maybe_run_decompositions(exported_program)
            mocked.assert_called_once()
        self.assertNotIn(torch.ops.aten.scatter_add.default, self._targets(lowered))
        self.assertTrue(matching_decomp_stamp(lowered, decomp_fingerprint()))


if __name__ == "__main__":
    unittest.main()
