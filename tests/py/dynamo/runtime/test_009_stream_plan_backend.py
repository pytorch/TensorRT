"""
Tests for stream_plan integration with the torch.compile(backend="tensorrt") path.

Exercises:
  - _try_apply_stream_plan directly (unit)
  - stream_plan=True kwarg wiring through torch.compile options (integration)
  - GPU concurrency proof via CUDA events when using the backend path
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.fx
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo.backend.backends import _try_apply_stream_plan
from torch_tensorrt.runtime.stream_plan import _trt_nodes

DEVICE = torch.device("cuda", 0)
MATRIX_SIZE = 4096


# ── Helpers (same as test_007) ────────────────────────────────────────────────


def _make_fake_trt_module(device_id: int = 0) -> MagicMock:
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule

    mod = MagicMock(spec=TorchTensorRTModule)
    mod.engine = MagicMock()
    mod.engine.device_info = MagicMock()
    mod.engine.device_info.id = device_id
    return mod


def _build_fan_out_gm() -> torch.fx.GraphModule:
    """Two independent TRT branches joined by a non-TRT add."""
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


def _build_chain_gm() -> torch.fx.GraphModule:
    """Two chained TRT nodes: x → acc0 → acc1."""
    parent = torch.nn.Module()
    parent.add_module("_run_on_acc_0", _make_fake_trt_module())
    parent.add_module("_run_on_acc_1", _make_fake_trt_module())

    g = torch.fx.Graph()
    x = g.placeholder("x")
    a0 = g.call_module("_run_on_acc_0", args=(x,))
    a1 = g.call_module("_run_on_acc_1", args=(a0,))
    g.output(a1)

    return torch.fx.GraphModule(parent, g)


def _build_single_trt_gm() -> torch.fx.GraphModule:
    """Single TRT node: no parallelism, plan should be skipped."""
    parent = torch.nn.Module()
    parent.add_module("_run_on_acc_0", _make_fake_trt_module())

    g = torch.fx.Graph()
    x = g.placeholder("x")
    a0 = g.call_module("_run_on_acc_0", args=(x,))
    g.output(a0)

    return torch.fx.GraphModule(parent, g)


# ── Unit tests for _try_apply_stream_plan ─────────────────────────────────────


class TestTryApplyStreamPlan(TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_returns_planned_gm_for_fan_out(self):
        """_try_apply_stream_plan should insert stream control nodes for fan-out."""
        gm = _build_fan_out_gm()
        result = _try_apply_stream_plan(gm)
        # The original gm should not be returned (a new planned module is created)
        self.assertTrue(getattr(result, "_stream_plan_applied", False))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_returns_planned_gm_for_chain(self):
        """_try_apply_stream_plan works on a simple chain topology."""
        gm = _build_chain_gm()
        result = _try_apply_stream_plan(gm)
        self.assertTrue(getattr(result, "_stream_plan_applied", False))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_skips_single_trt_node(self):
        """_try_apply_stream_plan must skip when < 2 TRT subgraphs exist."""
        gm = _build_single_trt_gm()
        result = _try_apply_stream_plan(gm)
        # Unchanged module returned — no plan applied
        self.assertFalse(getattr(result, "_stream_plan_applied", False))
        self.assertIs(result, gm)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_skips_when_cudagraphs_active(self):
        """_try_apply_stream_plan must skip when TRT cudagraphs mode is active."""
        from torch_tensorrt.runtime._cudagraphs import CudaGraphsMode

        gm = _build_fan_out_gm()
        with patch(
            "torch_tensorrt.runtime._cudagraphs._PY_RT_CUDAGRAPHS",
            CudaGraphsMode.WHOLE_GRAPH_CUDAGRAPHS,
        ):
            result = _try_apply_stream_plan(gm)
        self.assertFalse(getattr(result, "_stream_plan_applied", False))
        self.assertIs(result, gm)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_stream_attributes_stored_on_module(self):
        """Planned module must keep stream objects alive as _trt_stream_N attrs."""
        gm = _build_fan_out_gm()
        result = _try_apply_stream_plan(gm)
        stream_attrs = [k for k in vars(result) if k.startswith("_trt_stream")]
        self.assertGreater(len(stream_attrs), 0)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_graph_lints_after_plan(self):
        """FX graph should lint cleanly after _try_apply_stream_plan."""
        for build_fn in (_build_fan_out_gm, _build_chain_gm):
            gm = build_fn()
            result = _try_apply_stream_plan(gm)
            result.graph.lint()  # raises on malformed graph


# ── torch.compile kwarg wiring ────────────────────────────────────────────────


class TestBackendStreamPlanKwarg(TestCase):
    """Verify stream_plan=True is extracted from torch.compile options and wired."""

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_stream_plan_kwarg_calls_try_apply(self):
        """When stream_plan=True is passed, _try_apply_stream_plan is called on
        the compiled module.  We patch it to capture the call."""
        from torch_tensorrt.dynamo.backend import backends as _backends

        called_with = []

        def _capture(gm):
            called_with.append(gm)
            return gm

        # Build a tiny real model for torch.compile
        class TwoLayerModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(16, 16, bias=False)
                self.fc2 = torch.nn.Linear(16, 16, bias=False)

            def forward(self, x):
                return self.fc2(self.fc1(x))

        model = TwoLayerModel().cuda().eval()
        x = torch.randn(4, 16, device="cuda")

        with patch.object(_backends, "_try_apply_stream_plan", side_effect=_capture):
            compiled = torch.compile(
                model,
                backend="tensorrt",
                options={"stream_plan": True, "min_block_size": 1},
            )
            _ = compiled(x)

        self.assertTrue(
            len(called_with) > 0,
            "_try_apply_stream_plan was never called — kwarg not wired through",
        )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_stream_plan_false_does_not_call_try_apply(self):
        """When stream_plan is absent (default False), _try_apply_stream_plan is
        never called."""
        from torch_tensorrt.dynamo.backend import backends as _backends

        called_with = []

        def _capture(gm):
            called_with.append(gm)
            return gm

        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return x + 1

        model = SimpleModel().cuda().eval()
        x = torch.randn(4, 16, device="cuda")

        with patch.object(_backends, "_try_apply_stream_plan", side_effect=_capture):
            compiled = torch.compile(model, backend="tensorrt")
            _ = compiled(x)

        self.assertEqual(len(called_with), 0)


# ── GPU concurrency proof via CUDA events ──────────────────────────────────────


class TestBackendStreamConcurrency(TestCase):
    """
    End-to-end GPU concurrency proof using _try_apply_stream_plan directly.

    Builds a fan-out GM whose mock TRT modules inject real torch.mm work and
    record CUDA events.  After running the planned module on a real stream,
    checks that branch1 started before branch0 finished on the GPU — proving
    true GPU overlap.
    """

    WARMUP = 3

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_backend_fan_out_branches_overlap_on_gpu(self):
        ev_start_0 = torch.cuda.Event(enable_timing=True)
        ev_end_0 = torch.cuda.Event(enable_timing=True)
        ev_start_1 = torch.cuda.Event(enable_timing=True)
        ev_end_1 = torch.cuda.Event(enable_timing=True)

        def _make_recording_mod(ev_start, ev_end, label=""):
            mod = _make_fake_trt_module()

            def _fwd(x):
                ev_start.record()
                out = torch.mm(x, x)
                ev_end.record()
                return out

            mod.side_effect = _fwd
            return mod

        parent = torch.nn.Module()
        parent.add_module(
            "_run_on_acc_0", _make_recording_mod(ev_start_0, ev_end_0, "b0")
        )
        parent.add_module(
            "_run_on_acc_1", _make_recording_mod(ev_start_1, ev_end_1, "b1")
        )

        g = torch.fx.Graph()
        x = g.placeholder("x")
        a0 = g.call_module("_run_on_acc_0", args=(x,))
        a1 = g.call_module("_run_on_acc_1", args=(x,))
        out = g.call_function(torch.add, args=(a0, a1))
        g.output(out)
        gm = torch.fx.GraphModule(parent, g)

        planned = _try_apply_stream_plan(gm)
        self.assertTrue(
            getattr(planned, "_stream_plan_applied", False),
            "_try_apply_stream_plan did not apply — check guard conditions",
        )

        x_tensor = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device="cuda")
        for _ in range(self.WARMUP):
            planned(x_tensor)
        torch.cuda.synchronize()

        planned(x_tensor)
        torch.cuda.synchronize()

        t0_to_1_start = ev_start_0.elapsed_time(ev_start_1)
        t0_duration = ev_start_0.elapsed_time(ev_end_0)

        overlap = t0_to_1_start < t0_duration
        print(
            f"\n[backend stream concurrency]  "
            f"branch0={t0_duration:.2f}ms  "
            f"branch1 started {t0_to_1_start:.2f}ms after branch0  "
            f"overlap={overlap}"
        )
        self.assertTrue(
            overlap,
            f"Expected GPU overlap: branch1 should start before branch0 finishes. "
            f"branch0 duration={t0_duration:.2f}ms, "
            f"branch1 start offset={t0_to_1_start:.2f}ms",
        )


if __name__ == "__main__":
    run_tests()
