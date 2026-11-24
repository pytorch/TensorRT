import unittest
from contextlib import contextmanager
from typing import Generator
from unittest.mock import MagicMock, patch

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo._tracer import trace
from torch_tensorrt.dynamo.lowering._export_with_decomps import (
    export_with_tensorrt_decomps,
    maybe_run_decompositions,
)
from torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass import (
    ATEN_PRE_LOWERING_PASSES,
    pre_export_lowering,
    pre_export_lowering_needs_undecomposed_graph,
)
from torch_tensorrt.dynamo.lowering.passes._undecomposed_graph import (
    any_needs_undecomposed_graph,
)


@contextmanager
def _watch_run_decompositions(
    exported_program: torch.export.ExportedProgram,
) -> Generator[MagicMock, None, None]:
    with patch.object(
        type(exported_program),
        "run_decompositions",
        autospec=True,
        side_effect=type(exported_program).run_decompositions,
    ) as mocked:
        yield mocked


def _call_targets(exported_program: torch.export.ExportedProgram) -> set:
    return {
        node.target
        for module in exported_program.graph_module.modules()
        if isinstance(module, torch.fx.GraphModule)
        for node in module.graph.nodes
        if node.op == "call_function"
    }


class _InPlaceAdd(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + 1
        value = torch.ops.aten.add_.Tensor(value, value)
        return value


class _InPlaceOutsideTable(torch.nn.Module):
    # mul_ and sigmoid_ are not in the Torch-TRT decomp table, but stock export
    # keeps them in training IR and only run_decompositions functionalizes them.
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.clone()
        value.mul_(2)
        value.sigmoid_()
        return value


class _Attention(torch.nn.Module):
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


class TestOneShotDecomp(unittest.TestCase):
    def setUp(self) -> None:
        self.inputs = (torch.randn(4, 4),)
        self.model = _InPlaceAdd().eval()

    def test_inplace_add_is_decomposed_during_export(self) -> None:
        exported_program = export_with_tensorrt_decomps(
            self.model, args=self.inputs, strict=False
        )
        targets = _call_targets(exported_program)
        self.assertNotIn(torch.ops.aten.add_.Tensor, targets)
        self.assertIn(torch.ops.aten.add.Tensor, targets)

    def test_skips_run_decompositions_after_one_shot_export(self) -> None:
        exported_program = trace(self.model, self.inputs)
        with _watch_run_decompositions(exported_program) as mocked:
            maybe_run_decompositions(exported_program, CompilationSettings())
            mocked.assert_not_called()

    def test_stock_export_still_runs_decompositions(self) -> None:
        exported_program = torch.export.export(self.model, self.inputs, strict=False)
        with _watch_run_decompositions(exported_program) as mocked:
            maybe_run_decompositions(exported_program, CompilationSettings())
            mocked.assert_called_once()

    def test_training_ir_runs_even_without_table_ops(self) -> None:
        model, inputs = _InPlaceOutsideTable().eval(), (torch.randn(4, 4),)
        exported_program = torch.export.export(model, inputs, strict=False)
        with _watch_run_decompositions(exported_program) as mocked:
            decomposed = maybe_run_decompositions(
                exported_program, CompilationSettings()
            )
            mocked.assert_called_once()
        self.assertNotIn(torch.ops.aten.mul_.Tensor, _call_targets(decomposed))
        torch.testing.assert_close(decomposed.module()(*inputs), model(*inputs))

    def test_decomposed_program_is_not_decomposed_again(self) -> None:
        exported_program = torch.export.export(self.model, self.inputs, strict=False)
        decomposed = maybe_run_decompositions(exported_program, CompilationSettings())
        with _watch_run_decompositions(decomposed) as mocked:
            maybe_run_decompositions(decomposed, CompilationSettings())
            mocked.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_op_the_table_cannot_decompose_does_not_rerun(self) -> None:
        # On CUDA, _fused_rms_norm has a real kernel, so its table entry, a
        # composite-implicit decomposition, never applies and the op stays.
        model = torch.nn.RMSNorm(8).cuda().eval()
        exported_program = trace(model, (torch.randn(2, 8, device="cuda"),))
        self.assertIn(
            torch.ops.aten._fused_rms_norm.default, _call_targets(exported_program)
        )
        with _watch_run_decompositions(exported_program) as mocked:
            maybe_run_decompositions(exported_program, CompilationSettings())
            mocked.assert_not_called()

    def test_table_op_added_after_trace_reruns(self) -> None:
        # silu is in the table but absent from this graph when trace applied it, so
        # a pass that adds one afterwards leaves work for run_decompositions.
        exported_program = trace(self.model, self.inputs)
        graph = exported_program.graph_module.graph
        output = next(node for node in graph.nodes if node.op == "output")
        with graph.inserting_before(output):
            silu = graph.call_function(
                torch.ops.aten.silu.default, (output.args[0][0],)
            )
        output.args = ((silu,),)
        exported_program.graph_module.recompile()
        self.assertIn(torch.ops.aten.silu.default, _call_targets(exported_program))
        with _watch_run_decompositions(exported_program) as mocked:
            maybe_run_decompositions(exported_program, CompilationSettings())
            mocked.assert_called_once()

    def test_compile_settings_that_decompose_more_rerun(self) -> None:
        # Traced with the default table, which keeps SDPA; compile asks for it to
        # be decomposed, and the SDPA node left in the graph says so.
        q = torch.randn(1, 2, 4, 8)
        exported_program = trace(_Attention(), (q, q, q))
        settings = CompilationSettings(decompose_attention=True)
        with _watch_run_decompositions(exported_program) as mocked:
            decomposed = maybe_run_decompositions(exported_program, settings)
            mocked.assert_called_once()
        self.assertNotIn(
            torch.ops.aten.scaled_dot_product_attention.default,
            _call_targets(decomposed),
        )


class _ScatterAdd(torch.nn.Module):
    def forward(
        self, base: torch.Tensor, index: torch.Tensor, src: torch.Tensor
    ) -> torch.Tensor:
        return torch.ops.aten.scatter_add.default(base, 0, index, src)


class TestOneShotDecompShapeDependent(unittest.TestCase):
    """scatter_add decomposes only when the index extent is static. Its
    decomposition declines on a symbolic extent, so the one-shot export applies it
    where it can and compile never retraces for it."""

    def setUp(self) -> None:
        self.model = _ScatterAdd().eval()
        self.inputs = (
            torch.zeros((16, 8)),
            torch.zeros((4, 8), dtype=torch.int64),
            torch.ones((16, 8)),
        )

    def test_symbolic_extent_exports_and_keeps_scatter_add(self) -> None:
        index_rows = torch.export.Dim("index_rows", min=1, max=16)
        exported_program = export_with_tensorrt_decomps(
            self.model,
            args=self.inputs,
            dynamic_shapes={"base": {}, "index": {0: index_rows}, "src": {}},
        )
        self.assertIn(
            torch.ops.aten.scatter_add.default, _call_targets(exported_program)
        )
        with _watch_run_decompositions(exported_program) as mocked:
            lowered = maybe_run_decompositions(exported_program, CompilationSettings())
            mocked.assert_not_called()
        self.assertIn(torch.ops.aten.scatter_add.default, _call_targets(lowered))

    def test_static_extent_decomposes_scatter_add_during_export(self) -> None:
        exported_program = export_with_tensorrt_decomps(self.model, args=self.inputs)
        self.assertNotIn(
            torch.ops.aten.scatter_add.default, _call_targets(exported_program)
        )
        with _watch_run_decompositions(exported_program) as mocked:
            maybe_run_decompositions(exported_program, CompilationSettings())
            mocked.assert_not_called()
        torch.testing.assert_close(
            exported_program.module()(*self.inputs), self.model(*self.inputs)
        )


class _Interleave(torch.nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.stack([a, b], dim=2).flatten(1, 2)


class TestPreExportPassDeclarations(unittest.TestCase):
    """Pre-export lowering passes run after dynamo.trace, so each declares whether
    it needs the graph before Torch-TRT decompositions, and trace falls back to a
    stock export when one does."""

    def test_every_pre_export_pass_declares_whether_it_needs_undecomposed_graph(
        self,
    ) -> None:
        # A new pass in pre_lowering_pass_list must be decorated with
        # needs_undecomposed_graph or runs_on_decomposed_graph. Without one,
        # dynamo.trace falls back to exporting without decompositions.
        undeclared = [
            lowering_pass.__name__
            for lowering_pass in ATEN_PRE_LOWERING_PASSES.passes
            if not hasattr(lowering_pass, "_torch_tensorrt_needs_undecomposed_graph")
        ]
        self.assertEqual(undeclared, [])

    def test_guard_fails_for_a_new_undeclared_pass(self) -> None:
        def new_undeclared_pass(gm, settings):
            return gm

        passes = [*ATEN_PRE_LOWERING_PASSES.passes, new_undeclared_pass]
        with patch.object(ATEN_PRE_LOWERING_PASSES, "passes", passes):
            with self.assertRaisesRegex(AssertionError, "new_undeclared_pass"):
                self.test_every_pre_export_pass_declares_whether_it_needs_undecomposed_graph()

    def test_undeclared_pass_warns_and_needs_undecomposed_graph(self) -> None:
        def undeclared_pass(gm, settings):
            return gm

        with self.assertLogs(
            "torch_tensorrt.dynamo.lowering.passes._undecomposed_graph", "WARNING"
        ) as logs:
            self.assertTrue(
                any_needs_undecomposed_graph([undeclared_pass], CompilationSettings())
            )
        self.assertIn("undeclared_pass", "\n".join(logs.output))

    def test_default_settings_keep_one_shot_export(self) -> None:
        self.assertFalse(
            pre_export_lowering_needs_undecomposed_graph(CompilationSettings())
        )

    def test_interleave_rewrite_sees_stack_after_trace(self) -> None:
        inputs = (torch.randn(2, 4, 8), torch.randn(2, 4, 8))
        settings = CompilationSettings(replace_interleave_stack_with_cat=True)
        exported_program = trace(
            _Interleave(), inputs, replace_interleave_stack_with_cat=True
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

    def test_autocast_exports_without_decompositions(self) -> None:
        exported_program = trace(
            _Interleave(), (torch.randn(2, 4, 8),) * 2, enable_autocast=True
        )
        self.assertIn(torch.ops.aten.stack.default, _call_targets(exported_program))
        with _watch_run_decompositions(exported_program) as mocked:
            maybe_run_decompositions(
                exported_program, CompilationSettings(enable_autocast=True)
            )
            mocked.assert_called_once()


if __name__ == "__main__":
    unittest.main()
