"""
Unit / integration tests for stream_plan / apply_stream_plan.

Tests are layered:
  - Unit tests exercise _resolve_plan and _apply_stream_plan in isolation by
    mocking TRT submodule detection so no real CUDA engine is required.
  - Integration tests compile a real model with torchtrt and apply the plan.
  - SetDe integration tests compile → save → load → apply stream plan.
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.fx
from torch.testing._internal.common_utils import TestCase, run_tests

# Stream plan lives in the runtime public API
from torch_tensorrt.runtime.stream_plan import (
    StreamPlan,
    StreamPlanError,
    _apply_stream_plan,
    _resolve_plan,
    _trt_nodes,
    apply_stream_plan,
    stream_plan,
)

DEVICE = torch.device("cuda", 0)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_fake_trt_module(device_id: int = 0) -> MagicMock:
    """Return a mock that looks like a TorchTensorRTModule to _is_trt_module."""
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule

    mod = MagicMock(spec=TorchTensorRTModule)
    mod.engine = MagicMock()
    mod.engine.device_info = MagicMock()
    mod.engine.device_info.id = device_id
    return mod


def _build_two_subgraph_gm() -> torch.fx.GraphModule:
    """
    Builds a GraphModule with two call_module nodes (_run_on_acc_0,
    _run_on_acc_1) backed by mock TRT modules.

      x ─► acc0 ─► acc1 ─► output
    """
    parent = torch.nn.Module()
    parent.add_module("_run_on_acc_0", _make_fake_trt_module())
    parent.add_module("_run_on_acc_1", _make_fake_trt_module())

    g = torch.fx.Graph()
    x = g.placeholder("x")
    a0 = g.call_module("_run_on_acc_0", args=(x,))
    a1 = g.call_module("_run_on_acc_1", args=(a0,))
    g.output(a1)

    return torch.fx.GraphModule(parent, g)


def _build_fan_out_gm() -> torch.fx.GraphModule:
    """
    Builds a GraphModule with two independent TRT branches and a non-TRT join:

      x ─► acc0 ─┐
                  ├─► add ─► output
      x ─► acc1 ─┘

    Both branches are independent: neither feeds the other.  The final add
    runs on the caller stream and must wait for both streams to finish.
    """
    parent = torch.nn.Module()
    parent.add_module("_run_on_acc_0", _make_fake_trt_module())
    parent.add_module("_run_on_acc_1", _make_fake_trt_module())

    g = torch.fx.Graph()
    x = g.placeholder("x")
    a0 = g.call_module("_run_on_acc_0", args=(x,))
    a1 = g.call_module("_run_on_acc_1", args=(x,))
    out = g.call_function(torch.add, args=(a0, a1))
    g.output(out)

    return torch.fx.GraphModule(parent, g)


def _build_fan_in_gm() -> torch.fx.GraphModule:
    """
    Builds a GraphModule modelling the VLA fan-in pattern:

      x ─► vision ─┐
                    ├─► action ─► output
      x ─► lang  ─┘
    """
    parent = torch.nn.Module()
    parent.add_module("_run_on_acc_0", _make_fake_trt_module())  # vision
    parent.add_module("_run_on_acc_1", _make_fake_trt_module())  # language
    parent.add_module("_run_on_acc_2", _make_fake_trt_module())  # action

    g = torch.fx.Graph()
    x = g.placeholder("x")
    vision = g.call_module("_run_on_acc_0", args=(x,))
    lang = g.call_module("_run_on_acc_1", args=(x,))
    action = g.call_module("_run_on_acc_2", args=(vision, lang))
    g.output(action)

    return torch.fx.GraphModule(parent, g)


# ── Unit tests ────────────────────────────────────────────────────────────────


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestResolvePlan(TestCase):
    """Tests for _resolve_plan logic without a real TRT engine."""

    def _patch_cudagraphs(self):
        """Patch the cudagraphs check so it always looks like STANDARD mode."""
        from torch_tensorrt.runtime._cudagraphs import CudaGraphsMode

        # _PY_RT_CUDAGRAPHS is imported locally inside _resolve_plan, so patch
        # it at its definition site in _cudagraphs, not in stream_plan.
        return patch(
            "torch_tensorrt.runtime._cudagraphs._PY_RT_CUDAGRAPHS",
            CudaGraphsMode.STANDARD,
        )

    def test_auto_stream_allocation(self):
        gm = _build_two_subgraph_gm()
        with self._patch_cudagraphs():
            plan = _resolve_plan(gm, streams=None, hints=None)
        self.assertEqual(len(plan.assignment), 2)
        for s in plan.assignment.values():
            self.assertIsInstance(s, torch.cuda.Stream)

    def test_positional_streams(self):
        gm = _build_two_subgraph_gm()
        s0 = torch.cuda.Stream(device=DEVICE)
        s1 = torch.cuda.Stream(device=DEVICE)
        with self._patch_cudagraphs():
            plan = _resolve_plan(gm, streams=[s0, s1], hints=None)
        values = list(plan.assignment.values())
        self.assertIs(values[0], s0)
        self.assertIs(values[1], s1)

    def test_stream_count_mismatch_raises(self):
        gm = _build_two_subgraph_gm()
        with self._patch_cudagraphs():
            with self.assertRaises(StreamPlanError):
                _resolve_plan(
                    gm, streams=[torch.cuda.Stream(device=DEVICE)], hints=None
                )

    def test_hint_unknown_name_raises(self):
        gm = _build_two_subgraph_gm()
        s = torch.cuda.Stream(device=DEVICE)
        with self._patch_cudagraphs():
            with self.assertRaises(StreamPlanError):
                _resolve_plan(gm, streams=None, hints={"nonexistent_module": s})

    def test_hint_conflict_raises(self):
        gm = _build_two_subgraph_gm()
        s0 = torch.cuda.Stream(device=DEVICE)
        s1 = torch.cuda.Stream(device=DEVICE)
        # positional assigns s0 to acc_0; hint tries to assign s1 to same node
        with self._patch_cudagraphs():
            with self.assertRaises(StreamPlanError):
                _resolve_plan(
                    gm,
                    streams=[s0, torch.cuda.Stream(device=DEVICE)],
                    hints={"_run_on_acc_0": s1},
                )

    def test_no_trt_nodes_raises(self):
        # Empty module with no TRT subgraphs
        g = torch.fx.Graph()
        x = g.placeholder("x")
        g.output(x)
        plain_gm = torch.fx.GraphModule(torch.nn.Module(), g)
        from torch_tensorrt.runtime._cudagraphs import CudaGraphsMode

        with patch(
            "torch_tensorrt.runtime._cudagraphs._PY_RT_CUDAGRAPHS",
            CudaGraphsMode.STANDARD,
        ):
            with self.assertRaises(StreamPlanError):
                _resolve_plan(plain_gm, streams=None, hints=None)

    def test_cudagraphs_active_raises(self):
        gm = _build_two_subgraph_gm()
        from torch_tensorrt.runtime._cudagraphs import CudaGraphsMode

        with patch(
            "torch_tensorrt.runtime._cudagraphs._PY_RT_CUDAGRAPHS",
            CudaGraphsMode.SUBGRAPH_CUDAGRAPHS,
        ):
            with self.assertRaises(StreamPlanError):
                _resolve_plan(gm, streams=None, hints=None)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestApplyStreamPlan(TestCase):
    """Tests for the FX pass (_apply_stream_plan)."""

    def _make_plan(self, gm, streams):
        assignment = {}
        nodes = _trt_nodes(gm)
        for n, s in zip(nodes, streams):
            assignment[n.target] = s
        return StreamPlan(
            assignment=assignment,
            device=DEVICE,
        )

    def test_graph_lints_after_pass(self):
        gm = _build_two_subgraph_gm()
        s0 = torch.cuda.Stream(device=DEVICE)
        s1 = torch.cuda.Stream(device=DEVICE)
        plan = self._make_plan(gm, [s0, s1])
        planned = _apply_stream_plan(gm, plan)
        planned.graph.lint()  # must not raise

    def test_enter_exit_present(self):
        gm = _build_two_subgraph_gm()
        s0 = torch.cuda.Stream(device=DEVICE)
        plan = self._make_plan(gm, [s0, s0])  # same stream, no barriers needed
        planned = _apply_stream_plan(gm, plan)
        ops = {n.target for n in planned.graph.nodes if n.op == "call_function"}
        self.assertIn(torch.ops.tensorrt.enter_compute_stream.default, ops)
        self.assertIn(torch.ops.tensorrt.exit_compute_stream.default, ops)

    def test_sync_streams_inserted_for_fan_in(self):
        gm = _build_fan_in_gm()
        s_vis = torch.cuda.Stream(device=DEVICE)
        s_lang = torch.cuda.Stream(device=DEVICE)
        s_act = torch.cuda.Stream(device=DEVICE)
        plan = self._make_plan(gm, [s_vis, s_lang, s_act])
        planned = _apply_stream_plan(gm, plan)
        sync_nodes = [
            n
            for n in planned.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops.tensorrt.sync_streams.default
        ]
        # vision→action and lang→action syncs, plus final exit sync
        self.assertGreaterEqual(len(sync_nodes), 2)

    def test_original_gm_untouched(self):
        gm = _build_two_subgraph_gm()
        original_nodes = list(gm.graph.nodes)
        s0 = torch.cuda.Stream(device=DEVICE)
        # Public API contract: apply_stream_plan must not mutate the input gm.
        from torch_tensorrt.runtime._cudagraphs import CudaGraphsMode

        with patch(
            "torch_tensorrt.runtime._cudagraphs._PY_RT_CUDAGRAPHS",
            CudaGraphsMode.STANDARD,
        ):
            apply_stream_plan(gm, streams=[s0, s0])
        self.assertEqual(list(gm.graph.nodes), original_nodes)

    def test_stream_attrs_set_on_planned(self):
        gm = _build_two_subgraph_gm()
        s0 = torch.cuda.Stream(device=DEVICE)
        s1 = torch.cuda.Stream(device=DEVICE)
        plan = self._make_plan(gm, [s0, s1])
        planned = _apply_stream_plan(gm, plan)
        stream_attrs = [a for a in vars(planned) if a.startswith("_trt_stream")]
        self.assertTrue(len(stream_attrs) > 0)

    def test_same_stream_no_sync_between_nodes(self):
        gm = _build_two_subgraph_gm()
        s = torch.cuda.Stream(device=DEVICE)
        plan = self._make_plan(gm, [s, s])
        planned = _apply_stream_plan(gm, plan)
        # With same stream: one sync caller→compute (step 4) + one compute→caller (step 5)
        # No inter-node syncs since both nodes share the same stream.
        mid_syncs = [
            n
            for n in planned.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops.tensorrt.sync_streams.default
        ]
        self.assertEqual(len(mid_syncs), 2)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestStreamPlanContextManager(TestCase):
    """Tests for the RAII stream_plan() context manager."""

    def test_attrs_released_on_exit(self):
        gm = _build_two_subgraph_gm()
        s0 = torch.cuda.Stream(device=DEVICE)
        s1 = torch.cuda.Stream(device=DEVICE)
        from torch_tensorrt.runtime._cudagraphs import CudaGraphsMode

        with patch(
            "torch_tensorrt.runtime._cudagraphs._PY_RT_CUDAGRAPHS",
            CudaGraphsMode.STANDARD,
        ):
            with stream_plan(gm, streams=[s0, s1]) as planned:
                stream_attrs_inside = [
                    a for a in vars(planned) if a.startswith("_trt_stream")
                ]
            stream_attrs_outside = [
                a for a in vars(planned) if a.startswith("_trt_stream")
            ]
        self.assertTrue(len(stream_attrs_inside) > 0)
        self.assertEqual(len(stream_attrs_outside), 0)

    def test_attrs_released_on_exception(self):
        gm = _build_two_subgraph_gm()
        s0 = torch.cuda.Stream(device=DEVICE)
        from torch_tensorrt.runtime._cudagraphs import CudaGraphsMode

        planned_ref = None
        try:
            with patch(
                "torch_tensorrt.runtime._cudagraphs._PY_RT_CUDAGRAPHS",
                CudaGraphsMode.STANDARD,
            ):
                with stream_plan(gm, streams=[s0, s0]) as planned:
                    planned_ref = planned
                    raise RuntimeError("simulated error")
        except RuntimeError:
            pass
        self.assertIsNotNone(planned_ref)
        leaked = [a for a in vars(planned_ref) if a.startswith("_trt_stream")]
        self.assertEqual(leaked, [])


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestApplyStreamPlanAPI(TestCase):
    """Tests for the permanent apply_stream_plan() helper."""

    def test_returns_new_graph_module(self):
        gm = _build_two_subgraph_gm()
        s0 = torch.cuda.Stream(device=DEVICE)
        from torch_tensorrt.runtime._cudagraphs import CudaGraphsMode

        with patch(
            "torch_tensorrt.runtime._cudagraphs._PY_RT_CUDAGRAPHS",
            CudaGraphsMode.STANDARD,
        ):
            planned = apply_stream_plan(gm, streams=[s0, s0])
        self.assertIsNot(planned, gm)
        self.assertIsInstance(planned, torch.fx.GraphModule)

    def test_hint_based_assignment(self):
        gm = _build_two_subgraph_gm()
        s = torch.cuda.Stream(device=DEVICE)
        from torch_tensorrt.runtime._cudagraphs import CudaGraphsMode

        with patch(
            "torch_tensorrt.runtime._cudagraphs._PY_RT_CUDAGRAPHS",
            CudaGraphsMode.STANDARD,
        ):
            planned = apply_stream_plan(gm, hints={"_run_on_acc_0": s})
        self.assertIsNotNone(planned)


# ── Integration tests (require TRT) ──────────────────────────────────────────


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestStreamPlanIntegration(TestCase):
    """End-to-end: compile a real model, apply stream plan, run inference."""

    def _compile_simple_model(self):
        import torch_tensorrt

        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x) + 1.0

        device = DEVICE
        model = SimpleModel().eval().to(device)
        dtype = torch.float16
        inputs = [torch_tensorrt.Input(shape=(1, 4, 4), dtype=dtype)]
        compiled = torch_tensorrt.compile(
            model,
            ir="dynamo",
            inputs=inputs,
            min_block_size=1,
            device=device,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )
        return compiled, model

    def test_apply_stream_plan_produces_correct_outputs(self):
        compiled, model = self._compile_simple_model()
        device = DEVICE
        dtype = torch.float16

        nodes = _trt_nodes(compiled)
        if len(nodes) == 0:
            self.skipTest("No TRT nodes in compiled model (likely graph break)")

        streams = [torch.cuda.Stream(device=device) for _ in nodes]
        from torch_tensorrt.runtime._cudagraphs import CudaGraphsMode

        with patch(
            "torch_tensorrt.runtime._cudagraphs._PY_RT_CUDAGRAPHS",
            CudaGraphsMode.STANDARD,
        ):
            planned = apply_stream_plan(compiled, streams=streams)

        x = torch.randn(1, 4, 4, dtype=dtype, device=device)
        with torch.inference_mode():
            ref = compiled(x)
            out = planned(x)

        torch.testing.assert_close(ref, out, atol=1e-2, rtol=1e-2)

    def test_stream_plan_context_manager_runs(self):
        compiled, model = self._compile_simple_model()
        device = DEVICE
        dtype = torch.float16

        nodes = _trt_nodes(compiled)
        if len(nodes) == 0:
            self.skipTest("No TRT nodes in compiled model")

        streams = [torch.cuda.Stream(device=device) for _ in nodes]
        from torch_tensorrt.runtime._cudagraphs import CudaGraphsMode

        with patch(
            "torch_tensorrt.runtime._cudagraphs._PY_RT_CUDAGRAPHS",
            CudaGraphsMode.STANDARD,
        ):
            with stream_plan(compiled, streams=streams) as planned:
                x = torch.randn(1, 4, 4, dtype=dtype, device=device)
                with torch.inference_mode():
                    out = planned(x)
                    ref = compiled(x)

        torch.testing.assert_close(ref, out, atol=1e-2, rtol=1e-2)
        # Confirm attrs were cleaned up
        leaked = [a for a in vars(planned) if a.startswith("_trt_stream")]
        self.assertEqual(leaked, [])


# ── SetDe + stream plan integration tests ────────────────────────────────────


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestStreamPlanAfterSerDe(TestCase):
    """
    End-to-end: compile → save → load → apply stream plan → run inference.

    Verifies that:
      - TRT engines survive the serialization round-trip.
      - Submodule names (_run_on_acc_N) are preserved through save/load.
      - apply_stream_plan produces correct outputs on the deserialized module.
      - The stream_plan() context manager works the same way.
    """

    def _compile_and_save(self, tmpdir: str):
        import torch_tensorrt

        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x) + 1.0

        device = DEVICE
        model = SimpleModel().eval().to(device)
        dtype = torch.float16
        inputs = [torch_tensorrt.Input(shape=(1, 4, 4), dtype=dtype)]
        compiled = torch_tensorrt.compile(
            model,
            ir="dynamo",
            inputs=inputs,
            min_block_size=1,
            device=device,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )
        ep_path = os.path.join(tmpdir, "model.pt2")
        torch_tensorrt.save(compiled, ep_path, retrace=False)
        return ep_path, compiled

    def test_apply_stream_plan_after_save_load(self):
        """Compile → save → load → apply_stream_plan → outputs match."""
        import torch_tensorrt

        device = DEVICE
        dtype = torch.float16
        x = torch.randn(1, 4, 4, dtype=dtype, device=device)

        with tempfile.TemporaryDirectory() as tmpdir:
            ep_path, compiled = self._compile_and_save(tmpdir)

            nodes_before = _trt_nodes(compiled)
            if len(nodes_before) == 0:
                self.skipTest("No TRT nodes in compiled model (likely graph break)")

            with torch.inference_mode():
                ref = compiled(x)

            loaded = torch_tensorrt.load(ep_path).module()

        # tmpdir deleted here — loaded model must be fully self-contained
        loaded_nodes = _trt_nodes(loaded)
        if len(loaded_nodes) == 0:
            self.skipTest("No TRT nodes in loaded model")

        streams = [torch.cuda.Stream(device=device) for _ in loaded_nodes]
        from torch_tensorrt.runtime._cudagraphs import CudaGraphsMode

        with patch(
            "torch_tensorrt.runtime._cudagraphs._PY_RT_CUDAGRAPHS",
            CudaGraphsMode.STANDARD,
        ):
            planned = apply_stream_plan(loaded, streams=streams)

        with torch.inference_mode():
            out = planned(x)

        torch.testing.assert_close(ref, out, atol=1e-2, rtol=1e-2)

    def test_stream_plan_context_manager_after_save_load(self):
        """Compile → save → load → stream_plan() context manager → outputs match."""
        import torch_tensorrt

        device = DEVICE
        dtype = torch.float16
        x = torch.randn(1, 4, 4, dtype=dtype, device=device)

        with tempfile.TemporaryDirectory() as tmpdir:
            ep_path, compiled = self._compile_and_save(tmpdir)

            if len(_trt_nodes(compiled)) == 0:
                self.skipTest("No TRT nodes in compiled model (likely graph break)")

            with torch.inference_mode():
                ref = compiled(x)

            loaded = torch_tensorrt.load(ep_path).module()

        loaded_nodes = _trt_nodes(loaded)
        if len(loaded_nodes) == 0:
            self.skipTest("No TRT nodes in loaded model")

        streams = [torch.cuda.Stream(device=device) for _ in loaded_nodes]
        from torch_tensorrt.runtime._cudagraphs import CudaGraphsMode

        with patch(
            "torch_tensorrt.runtime._cudagraphs._PY_RT_CUDAGRAPHS",
            CudaGraphsMode.STANDARD,
        ):
            with stream_plan(loaded, streams=streams) as planned:
                with torch.inference_mode():
                    out = planned(x)

        torch.testing.assert_close(ref, out, atol=1e-2, rtol=1e-2)
        # Stream attrs must be released after context exit
        leaked = [a for a in vars(planned) if a.startswith("_trt_stream")]
        self.assertEqual(leaked, [])

    def test_hint_based_assignment_works_through_serde(self):
        """
        After save/load, the dynamo exporter rewrites TRT call_module nodes to
        call_function(execute_engine) with TorchBind engine get_attrs.  The
        node identity changes (engine attr name = ``_run_on_acc_0_engine``),
        but hint-based stream assignment must still work — apply_stream_plan
        accepts both the submodule-name spelling (``_run_on_acc_0``) and the
        engine-attr-name spelling (``_run_on_acc_0_engine``).
        """
        import torch_tensorrt
        from torch_tensorrt.runtime.stream_plan import _trt_node_key

        device = DEVICE

        with tempfile.TemporaryDirectory() as tmpdir:
            ep_path, compiled = self._compile_and_save(tmpdir)

            original_keys = {_trt_node_key(n) for n in _trt_nodes(compiled)}
            if not original_keys:
                self.skipTest("No TRT nodes in compiled model (likely graph break)")

            loaded = torch_tensorrt.load(ep_path).module()

        # After load the keys are engine-attr-name form (e.g. _run_on_acc_0_engine).
        loaded_keys = {_trt_node_key(n) for n in _trt_nodes(loaded)}
        self.assertGreater(len(loaded_keys), 0, "load lost all TRT nodes")
        # Each original submodule name should map 1:1 to a loaded engine attr name.
        for orig in original_keys:
            self.assertIn(
                f"{orig}_engine" if not orig.endswith("_engine") else orig,
                loaded_keys,
            )

        # Hints accept the original submodule names — apply_stream_plan
        # normalizes them to the engine-attr-name form internally.
        s = torch.cuda.Stream(device=device)
        hints = {name: s for name in original_keys}
        from torch_tensorrt.runtime._cudagraphs import CudaGraphsMode

        with patch(
            "torch_tensorrt.runtime._cudagraphs._PY_RT_CUDAGRAPHS",
            CudaGraphsMode.STANDARD,
        ):
            planned = apply_stream_plan(loaded, hints=hints)

        self.assertIsInstance(planned, torch.fx.GraphModule)

    def test_save_stream_planned_module_succeeds(self):
        """
        torch_tensorrt.save() of a stream-planned module produces a re-loadable
        artifact.  StreamGuard ScriptObjects pickle handle-less so the saved
        .pt2 has no dangling cudaStream_t pointers.  Stream handles are bound
        fresh at load time via bind_stream_plan_streams() (or auto_bind=True
        in the StreamGuard's own __setstate__).
        """
        import torch_tensorrt

        device = DEVICE
        dtype = torch.float16

        with tempfile.TemporaryDirectory() as tmpdir:
            ep_path, compiled = self._compile_and_save(tmpdir)
            loaded = torch_tensorrt.load(ep_path).module()

        loaded_nodes = _trt_nodes(loaded)
        if len(loaded_nodes) == 0:
            self.skipTest("No TRT nodes in loaded model")

        streams = [torch.cuda.Stream(device=device) for _ in loaded_nodes]
        from torch_tensorrt.runtime._cudagraphs import CudaGraphsMode

        with patch(
            "torch_tensorrt.runtime._cudagraphs._PY_RT_CUDAGRAPHS",
            CudaGraphsMode.STANDARD,
        ):
            planned = apply_stream_plan(loaded, streams=streams)

        with tempfile.TemporaryDirectory() as tmpdir2:
            # Should not raise — planned modules now survive serde via
            # TorchBind pickle on the StreamGuard attributes.
            torch_tensorrt.save(
                planned,
                os.path.join(tmpdir2, "planned.pt2"),
                inputs=[torch.randn(1, 4, 4, dtype=dtype, device=device)],
            )


# ── Fan-out correctness + concurrency ────────────────────────────────────────


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestFanOutStreamPlan(TestCase):
    """
    Verify that _apply_stream_plan correctly handles the fan-out topology:

      x ─► eng0 (stream s0) ─┐
                               ├─► add ─► output
      x ─► eng1 (stream s1) ─┘

    Step 5 of _apply_stream_plan must sync BOTH s0 and s1 back to the caller
    stream before the non-TRT add node reads their outputs.  The original code
    only synced the last-in-graph-order TRT stream, which was None for a
    fan-out where the last node is a non-TRT join.
    """

    def _make_plan(self, gm, streams):
        assignment = {}
        for n, s in zip(_trt_nodes(gm), streams):
            assignment[n.target] = s
        return StreamPlan(assignment=assignment, device=DEVICE)

    def test_both_streams_synced_to_caller(self):
        """sync_streams nodes must cover both s0 and s1 → caller.

        After the StreamGuard refactor, sync_streams args are get_attr nodes
        pointing at top-level _trt_stream_guard_N attributes rather than int
        handles.  We resolve each get_attr to its bound StreamGuard and check
        that the handles cover s0 and s1.
        """
        gm = _build_fan_out_gm()
        s0 = torch.cuda.Stream(device=DEVICE)
        s1 = torch.cuda.Stream(device=DEVICE)
        plan = self._make_plan(gm, [s0, s1])
        planned = _apply_stream_plan(gm, plan)

        sync_nodes = [
            n
            for n in planned.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops.tensorrt.sync_streams.default
        ]

        def _resolved_handle(arg):
            if isinstance(arg, torch.fx.Node) and arg.op == "get_attr":
                guard = getattr(planned, arg.target, None)
                if guard is not None and hasattr(guard, "get_handle"):
                    return guard.get_handle()
            return None

        src_handles = {
            h for n in sync_nodes if (h := _resolved_handle(n.args[0])) is not None
        }
        self.assertIn(s0.cuda_stream, src_handles, "s0 → caller sync missing")
        self.assertIn(s1.cuda_stream, src_handles, "s1 → caller sync missing")

    def test_graph_lints_fan_out(self):
        gm = _build_fan_out_gm()
        s0 = torch.cuda.Stream(device=DEVICE)
        s1 = torch.cuda.Stream(device=DEVICE)
        plan = self._make_plan(gm, [s0, s1])
        planned = _apply_stream_plan(gm, plan)
        planned.graph.lint()


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestStreamConcurrency(TestCase):
    """
    Verify that two independent TRT branches actually execute concurrently on
    the GPU when assigned to different CUDA streams.

    Strategy: replace the fake TRT modules' side_effects with functions that
    do a real large matmul and record CUDA events at start/end.  After one
    measured run, check that:

        ev_start_1  <  ev_end_0    (branch 1 started before branch 0 finished)

    This proves genuine GPU overlap — the two engines ran in parallel.
    Event timestamps are global GPU timestamps comparable across streams.
    """

    MATRIX_SIZE = 4096  # large enough to take several ms on any modern GPU

    def _make_recording_mod(self, ev_start, ev_end):
        """Return a fake TRT module that does a real matmul and records events."""
        mod = _make_fake_trt_module()
        size = self.MATRIX_SIZE

        def _forward(x):
            ev_start.record()  # recorded on current CUDA stream (set by set_stream)
            out = torch.mm(x, x)  # dispatched async — CPU returns immediately
            ev_end.record()
            return out

        mod.side_effect = _forward
        return mod

    def _build_recording_fan_out_gm(self, mod0, mod1):
        parent = torch.nn.Module()
        parent.add_module("_run_on_acc_0", mod0)
        parent.add_module("_run_on_acc_1", mod1)

        g = torch.fx.Graph()
        x = g.placeholder("x")
        a0 = g.call_module("_run_on_acc_0", args=(x,))
        a1 = g.call_module("_run_on_acc_1", args=(x,))
        out = g.call_function(torch.add, args=(a0, a1))
        g.output(out)
        return torch.fx.GraphModule(parent, g)

    def test_independent_branches_overlap_on_gpu(self):
        """
        Branch 0 runs on s0, branch 1 runs on s1.  Both are dispatched to the
        GPU with almost no CPU time between them (async dispatch).  We verify
        that the GPU intervals overlap: ev_start_1 < ev_end_0.
        """
        from torch_tensorrt.runtime._cudagraphs import CudaGraphsMode

        ev_start_0 = torch.cuda.Event(enable_timing=True)
        ev_end_0 = torch.cuda.Event(enable_timing=True)
        ev_start_1 = torch.cuda.Event(enable_timing=True)
        ev_end_1 = torch.cuda.Event(enable_timing=True)

        mod0 = self._make_recording_mod(ev_start_0, ev_end_0)
        mod1 = self._make_recording_mod(ev_start_1, ev_end_1)
        gm = self._build_recording_fan_out_gm(mod0, mod1)

        x = torch.randn(
            self.MATRIX_SIZE, self.MATRIX_SIZE, dtype=torch.float16, device=DEVICE
        )

        with patch(
            "torch_tensorrt.runtime._cudagraphs._PY_RT_CUDAGRAPHS",
            CudaGraphsMode.STANDARD,
        ):
            s0 = torch.cuda.Stream(device=DEVICE)
            s1 = torch.cuda.Stream(device=DEVICE)
            planned = apply_stream_plan(gm, streams=[s0, s1])

        with torch.inference_mode():
            planned(x)  # warmup — JIT, caching, etc.
            torch.cuda.synchronize(DEVICE)
            planned(x)  # measured run
            torch.cuda.synchronize(DEVICE)

        # elapsed_time(e): ms from self → e  (positive = e is later)
        t0_to_1_start = ev_start_0.elapsed_time(ev_start_1)  # gap between branch starts
        t0_duration = ev_start_0.elapsed_time(ev_end_0)  # GPU time for branch 0

        # Concurrent iff branch 1 started before branch 0 finished
        overlap = t0_to_1_start < t0_duration

        print(
            f"\n[stream concurrency]  "
            f"branch0={t0_duration:.2f}ms  "
            f"branch1 started {t0_to_1_start:.2f}ms after branch0  "
            f"overlap={overlap}"
        )
        self.assertTrue(
            overlap,
            f"Expected GPU overlap: branch 1 started {t0_to_1_start:.2f}ms after "
            f"branch 0, but branch 0 takes {t0_duration:.2f}ms — streams are sequential.",
        )


def _build_wide_dag_gm() -> torch.fx.GraphModule:
    """
    1 → 3 → 2 → 3 → 1 topology.

    Level 2 (fan-out):   acc0(x), acc1(x), acc2(x)          — 3 independent TRT nodes
    Level 3 (converge):  acc3(j01), acc4(j12)                — 2 TRT nodes
                         j01 = add(acc0, acc1)  ← non-TRT
                         j12 = add(acc1, acc2)  ← non-TRT
    Level 4 (expand):    acc5(acc3), acc6(j34), acc7(acc4)   — 3 TRT nodes
                         j34 = add(acc3, acc4)  ← non-TRT
    Level 5 (output):    add(add(acc5, acc6), acc7)           — non-TRT

    Stream plan inserts cross-stream syncs at every level boundary and
    syncs all 8 streams (s0–s7) back to the caller before the output node.
    """
    parent = torch.nn.Module()
    for i in range(8):
        parent.add_module(f"_run_on_acc_{i}", _make_fake_trt_module())

    g = torch.fx.Graph()
    x = g.placeholder("x")

    # Level 2 — three independent branches reading x
    a0 = g.call_module("_run_on_acc_0", args=(x,))
    a1 = g.call_module("_run_on_acc_1", args=(x,))
    a2 = g.call_module("_run_on_acc_2", args=(x,))

    # Level 2→3 non-TRT joins
    j01 = g.call_function(torch.add, args=(a0, a1))
    j12 = g.call_function(torch.add, args=(a1, a2))

    # Level 3 — two converging TRT nodes
    a3 = g.call_module("_run_on_acc_3", args=(j01,))
    a4 = g.call_module("_run_on_acc_4", args=(j12,))

    # Level 3→4 non-TRT join
    j34 = g.call_function(torch.add, args=(a3, a4))

    # Level 4 — three expanding TRT nodes
    a5 = g.call_module("_run_on_acc_5", args=(a3,))  # direct dep on level-3 TRT
    a6 = g.call_module("_run_on_acc_6", args=(j34,))  # dep via non-TRT join
    a7 = g.call_module("_run_on_acc_7", args=(a4,))  # direct dep on level-3 TRT

    # Level 5 — non-TRT output
    out = g.call_function(
        torch.add, args=(g.call_function(torch.add, args=(a5, a6)), a7)
    )
    g.output(out)

    return torch.fx.GraphModule(parent, g)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestWideDagStreamPlan(TestCase):
    """
    Structural and concurrency tests for the 1→3→2→3→1 topology.

    Eight TRT nodes across eight streams exercise every combination of:
      - parallel fan-out (level 2 nodes all read x independently)
      - convergence via non-TRT joins (j01, j12 funnel level-2 output to level-3)
      - re-expansion with both direct TRT→TRT and join-mediated TRT→TRT deps
      - final caller-stream sync of all eight streams before a non-TRT output
    """

    MATRIX_SIZE = 4096

    def _make_plan(self, gm, streams):
        assignment = {n.target: s for n, s in zip(_trt_nodes(gm), streams)}
        return StreamPlan(assignment=assignment, device=DEVICE)

    def _make_streams(self, n=8):
        return [torch.cuda.Stream(device=DEVICE) for _ in range(n)]

    def test_graph_lints(self):
        gm = _build_wide_dag_gm()
        streams = self._make_streams()
        plan = self._make_plan(gm, streams)
        planned = _apply_stream_plan(gm, plan)
        planned.graph.lint()

    def test_all_eight_streams_synced_to_caller(self):
        """Every one of the eight TRT streams must appear as src in a sync node.

        sync_streams args are get_attr nodes pointing at top-level
        _trt_stream_guard_N TorchBind attributes; we resolve each to its
        bound StreamGuard handle.
        """
        gm = _build_wide_dag_gm()
        streams = self._make_streams()
        plan = self._make_plan(gm, streams)
        planned = _apply_stream_plan(gm, plan)

        def _resolved_handle(arg):
            if isinstance(arg, torch.fx.Node) and arg.op == "get_attr":
                guard = getattr(planned, arg.target, None)
                if guard is not None and hasattr(guard, "get_handle"):
                    return guard.get_handle()
            return None

        sync_src_handles = {
            h
            for n in planned.graph.nodes
            if n.op == "call_function"
            and n.target == torch.ops.tensorrt.sync_streams.default
            and (h := _resolved_handle(n.args[0])) is not None
        }
        for i, s in enumerate(streams):
            self.assertIn(
                s.cuda_stream,
                sync_src_handles,
                f"stream s{i} (level-{i//3+2} node) never synced",
            )

    def test_plan_applied_flag(self):
        gm = _build_wide_dag_gm()
        streams = self._make_streams()
        plan = self._make_plan(gm, streams)
        planned = _apply_stream_plan(gm, plan)
        self.assertTrue(getattr(planned, "_stream_plan_applied", False))

    def test_level2_branches_overlap_on_gpu(self):
        """
        The three level-2 branches (acc0, acc1, acc2) are fully independent —
        each reads x on its own stream.  Record start/end events on all three
        and verify that at least branch 1 and branch 2 started before branch 0
        finished, proving the GPU executed them in parallel.
        """
        ev = {
            i: (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for i in range(3)
        }

        # Build the GM directly so every mock has the right side_effect from the start.
        parent = torch.nn.Module()
        for i in range(8):
            mod = _make_fake_trt_module()
            if i < 3:
                ev_s, ev_e = ev[i]

                def _fwd(x, _s=ev_s, _e=ev_e):
                    _s.record()
                    out = torch.mm(x, x)
                    _e.record()
                    return out

                mod.side_effect = _fwd
            else:
                mod.side_effect = lambda x: x  # passthrough for non-timing nodes
            parent.add_module(f"_run_on_acc_{i}", mod)

        g = torch.fx.Graph()
        x = g.placeholder("x")
        a0 = g.call_module("_run_on_acc_0", args=(x,))
        a1 = g.call_module("_run_on_acc_1", args=(x,))
        a2 = g.call_module("_run_on_acc_2", args=(x,))
        j01 = g.call_function(torch.add, args=(a0, a1))
        j12 = g.call_function(torch.add, args=(a1, a2))
        a3 = g.call_module("_run_on_acc_3", args=(j01,))
        a4 = g.call_module("_run_on_acc_4", args=(j12,))
        j34 = g.call_function(torch.add, args=(a3, a4))
        a5 = g.call_module("_run_on_acc_5", args=(a3,))
        a6 = g.call_module("_run_on_acc_6", args=(j34,))
        a7 = g.call_module("_run_on_acc_7", args=(a4,))
        out = g.call_function(
            torch.add, args=(g.call_function(torch.add, args=(a5, a6)), a7)
        )
        g.output(out)
        gm = torch.fx.GraphModule(parent, g)

        streams = self._make_streams()
        plan = self._make_plan(gm, streams)

        from torch_tensorrt.runtime._cudagraphs import CudaGraphsMode

        with patch(
            "torch_tensorrt.runtime._cudagraphs._PY_RT_CUDAGRAPHS",
            CudaGraphsMode.STANDARD,
        ):
            planned = apply_stream_plan(gm, streams=streams)

        x = torch.randn(
            self.MATRIX_SIZE, self.MATRIX_SIZE, dtype=torch.float16, device=DEVICE
        )

        with torch.inference_mode():
            planned(x)  # warmup
            torch.cuda.synchronize(DEVICE)
            planned(x)  # measured
            torch.cuda.synchronize(DEVICE)

        s0, e0 = ev[0]
        s1, e1 = ev[1]
        s2, e2 = ev[2]

        t0_duration = s0.elapsed_time(e0)
        t1_start = s0.elapsed_time(s1)
        t2_start = s0.elapsed_time(s2)

        overlap_1 = t1_start < t0_duration
        overlap_2 = t2_start < t0_duration

        print(
            f"\n[wide-dag concurrency]  "
            f"branch0={t0_duration:.2f}ms  "
            f"branch1 +{t1_start:.2f}ms  "
            f"branch2 +{t2_start:.2f}ms  "
            f"overlap=(b1={overlap_1}, b2={overlap_2})"
        )
        self.assertTrue(
            overlap_1 or overlap_2,
            f"Expected at least one level-2 branch to overlap branch0. "
            f"branch0 duration={t0_duration:.2f}ms, "
            f"branch1 start={t1_start:.2f}ms, branch2 start={t2_start:.2f}ms",
        )


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestWideDagNumericalCorrectness(TestCase):
    """
    Numerical correctness of stream_plan applied to a real compiled model with
    multiple TRT subgraphs.

    torch_executed_ops={"torch.ops.aten.relu.default"} forces graph breaks at
    each relu, producing three TRT subgraphs separated by PyTorch-executed nodes:

        x ──► [TRT: enc] ──► relu* ──► [TRT: b0+b1+b2+add+div] ──► relu* ──► [TRT: dec] ──► output

    The planned module must produce outputs numerically identical to the PyTorch
    reference across NUM_RUNS independent inputs, regardless of stream assignment.
    """

    NUM_RUNS = 20
    D = 64

    def _build_and_compile(self):
        import torch_tensorrt

        class WideModel(torch.nn.Module):
            def __init__(self, d):
                super().__init__()
                self.enc = torch.nn.Linear(d, d, bias=False)
                self.b0 = torch.nn.Linear(d, d, bias=False)
                self.b1 = torch.nn.Linear(d, d, bias=False)
                self.b2 = torch.nn.Linear(d, d, bias=False)
                self.dec = torch.nn.Linear(d, d, bias=False)

            def forward(self, x):
                h = self.enc(x)
                h = torch.relu(h)  # graph break 1
                a = self.b0(h)
                b = self.b1(h)
                c = self.b2(h)
                h2 = (a + b + c) / 3
                h2 = torch.relu(h2)  # graph break 2
                return self.dec(h2)

        model = WideModel(self.D).eval().cuda().half()
        inputs = [torch_tensorrt.Input(shape=(8, self.D), dtype=torch.float16)]
        compiled = torch_tensorrt.compile(
            model,
            ir="dynamo",
            inputs=inputs,
            min_block_size=1,
            torch_executed_ops={"torch.ops.aten.relu.default"},
            device=DEVICE,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )
        return compiled, model

    def test_correctness_over_multiple_runs(self):
        compiled, model = self._build_and_compile()

        nodes = _trt_nodes(compiled)
        if len(nodes) < 2:
            self.skipTest(
                f"Expected ≥2 TRT subgraphs for stream plan; got {len(nodes)}. "
                "Check that torch_executed_ops forced graph breaks correctly."
            )

        streams = [torch.cuda.Stream(device=DEVICE) for _ in nodes]
        from torch_tensorrt.runtime._cudagraphs import CudaGraphsMode

        with patch(
            "torch_tensorrt.runtime._cudagraphs._PY_RT_CUDAGRAPHS",
            CudaGraphsMode.STANDARD,
        ):
            planned = apply_stream_plan(compiled, streams=streams)

        # Warmup both paths
        x_warm = torch.randn(8, self.D, dtype=torch.float16, device=DEVICE)
        with torch.inference_mode():
            for _ in range(3):
                compiled(x_warm)
                planned(x_warm)
            torch.cuda.synchronize(DEVICE)

        # Numerical correctness over NUM_RUNS independent inputs
        for i in range(self.NUM_RUNS):
            xi = torch.randn(8, self.D, dtype=torch.float16, device=DEVICE)
            with torch.inference_mode():
                ref = model(xi)
                out = planned(xi)
            torch.testing.assert_close(
                out,
                ref,
                atol=5e-2,
                rtol=5e-2,
                msg=f"Output mismatch on run {i}",
            )

    def test_stream_plan_subgraph_count(self):
        """Verify the model actually yields ≥2 TRT subgraphs (smoke-test for the
        torch_executed_ops graph-break strategy)."""
        compiled, _ = self._build_and_compile()
        nodes = _trt_nodes(compiled)
        self.assertGreaterEqual(
            len(nodes),
            2,
            f"Expected ≥2 TRT subgraphs; got {len(nodes)}. "
            "torch_executed_ops may not have forced graph breaks as expected.",
        )


if __name__ == "__main__":
    run_tests()
