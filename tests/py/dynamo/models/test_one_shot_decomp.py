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


if __name__ == "__main__":
    unittest.main()
