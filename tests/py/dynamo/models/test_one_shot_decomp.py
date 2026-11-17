import unittest
from unittest.mock import patch

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo._tracer import trace
from torch_tensorrt.dynamo.lowering._export_with_decomps import (
    decomp_fingerprint,
    export_with_tensorrt_decomps,
    matching_decomp_stamp,
    maybe_run_decompositions,
)
from torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass import (
    ATEN_PRE_LOWERING_PASSES,
    pre_export_lowering,
    pre_export_lowering_undecomposed_ops,
)
from torch_tensorrt.dynamo.lowering.passes._undecomposed_ops import (
    ALL_OPS,
    undecomposed_ops_for,
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
            maybe_run_decompositions(exported_program, CompilationSettings())
            mocked.assert_not_called()

    def test_stock_export_still_runs_decompositions(self) -> None:
        exported_program = torch.export.export(self.model, self.inputs, strict=False)
        with patch.object(
            type(exported_program),
            "run_decompositions",
            wraps=exported_program.run_decompositions,
        ) as mocked:
            maybe_run_decompositions(exported_program, CompilationSettings())
            mocked.assert_called_once()

    def test_settings_mismatch_runs_decompositions(self) -> None:
        exported_program = export_with_tensorrt_decomps(
            self.model,
            args=self.inputs,
            strict=False,
        )
        with patch.object(
            type(exported_program),
            "run_decompositions",
            wraps=exported_program.run_decompositions,
        ) as mocked:
            maybe_run_decompositions(
                exported_program, CompilationSettings(decompose_attention=True)
            )
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
            maybe_run_decompositions(
                exported_program, CompilationSettings(enable_autocast=True)
            )
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
            lowered = maybe_run_decompositions(exported_program, CompilationSettings())
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
            lowered = maybe_run_decompositions(exported_program, CompilationSettings())
            mocked.assert_called_once()
        self.assertNotIn(torch.ops.aten.scatter_add.default, self._targets(lowered))
        self.assertTrue(matching_decomp_stamp(lowered, decomp_fingerprint()))


def _call_targets(exported_program: torch.export.ExportedProgram) -> set:
    return {
        node.target
        for module in exported_program.graph_module.modules()
        if isinstance(module, torch.fx.GraphModule)
        for node in module.graph.nodes
        if node.op == "call_function"
    }


class _Interleave(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.stack([a, b], dim=2).flatten(1, 2)


class _Attention(torch.nn.Module):
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


class TestPreExportPassDeclarations(unittest.TestCase):
    """Pre-export lowering passes run after dynamo.trace has decomposed the graph, so
    each declares the ops it must see undecomposed and trace leaves those for compile.
    """

    def test_every_pre_export_pass_declares_its_undecomposed_ops(self) -> None:
        # A new pass in pre_lowering_pass_list must declare, with
        # needs_undecomposed_ops or needs_no_undecomposed_ops, which ops it matches
        # that the Torch-TRT decomposition table rewrites. Without one, dynamo.trace
        # falls back to exporting without decompositions.
        undeclared = [
            lowering_pass.__name__
            for lowering_pass in ATEN_PRE_LOWERING_PASSES.passes
            if not hasattr(lowering_pass, "_torch_tensorrt_undecomposed_ops")
        ]
        self.assertEqual(undeclared, [])

    def test_undeclared_pass_falls_back_to_all_ops(self) -> None:
        def undeclared_pass(gm, settings):
            return gm

        with self.assertLogs(
            "torch_tensorrt.dynamo.lowering.passes._undecomposed_ops", "WARNING"
        ):
            self.assertEqual(
                undecomposed_ops_for([undeclared_pass], CompilationSettings()),
                ALL_OPS,
            )

    def test_default_settings_defer_only_shape_dependent_ops(self) -> None:
        self.assertEqual(
            pre_export_lowering_undecomposed_ops(CompilationSettings()), frozenset()
        )

    def test_interleave_rewrite_sees_stack_after_trace(self) -> None:
        inputs = (torch.randn(2, 4, 8), torch.randn(2, 4, 8))
        settings = CompilationSettings(replace_interleave_stack_with_cat=True)
        exported_program = trace(
            _Interleave(), inputs, replace_interleave_stack_with_cat=True
        )
        self.assertTrue(
            matching_decomp_stamp(exported_program, decomp_fingerprint(settings))
        )
        self.assertIn(torch.ops.aten.stack.default, _call_targets(exported_program))

        lowered = pre_export_lowering(exported_program, settings)
        self.assertNotIn(torch.ops.aten.stack.default, _call_targets(lowered))
        decomposed = maybe_run_decompositions(lowered, settings)
        torch.testing.assert_close(decomposed.module()(*inputs), _Interleave()(*inputs))

    def test_interleave_stack_decomposes_without_the_rewrite(self) -> None:
        exported_program = trace(_Interleave(), (torch.randn(2, 4, 8),) * 2)
        self.assertNotIn(torch.ops.aten.stack.default, _call_targets(exported_program))

    def test_decompose_attention_keeps_sdpa_for_layout_pass(self) -> None:
        q = torch.randn(1, 2, 4, 8)
        settings = CompilationSettings(decompose_attention=True)
        exported_program = trace(_Attention(), (q, q, q), decompose_attention=True)
        self.assertIn(
            torch.ops.aten.scaled_dot_product_attention.default,
            _call_targets(exported_program),
        )
        decomposed = maybe_run_decompositions(
            pre_export_lowering(exported_program, settings), settings
        )
        self.assertNotIn(
            torch.ops.aten.scaled_dot_product_attention.default,
            _call_targets(decomposed),
        )

    def test_trace_settings_that_decompose_needed_ops_warn(self) -> None:
        # Traced without the rewrite, so stack is already decomposed when compile
        # asks for it.
        exported_program = trace(_Interleave(), (torch.randn(2, 4, 8),) * 2)
        with self.assertLogs(
            "torch_tensorrt.dynamo.lowering._export_with_decomps", "WARNING"
        ) as logs:
            maybe_run_decompositions(
                exported_program,
                CompilationSettings(replace_interleave_stack_with_cat=True),
            )
        self.assertIn("aten.stack.default", "\n".join(logs.output))

    def test_autocast_exports_without_decompositions(self) -> None:
        exported_program = trace(
            _Interleave(), (torch.randn(2, 4, 8),) * 2, enable_autocast=True
        )
        self.assertIn(torch.ops.aten.stack.default, _call_targets(exported_program))
        self.assertFalse(
            matching_decomp_stamp(
                exported_program,
                decomp_fingerprint(CompilationSettings(enable_autocast=True)),
            )
        )


if __name__ == "__main__":
    unittest.main()
