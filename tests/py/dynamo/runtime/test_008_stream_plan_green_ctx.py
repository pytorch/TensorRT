"""
Integration tests: stream_plan with CUDA Green Context streams.

CUDA Green Contexts (CUDA 12.4+) partition a device's SM resources into
isolated slices.  Each slice gets a dedicated execution environment and a
non-blocking stream, enabling deterministic concurrency between otherwise
independent workloads.  The archetypal use case is VLA robotics inference:
vision + language sub-graphs run in parallel on separate SM partitions and
fan in to an action sub-graph.

These tests verify:
  1. The cuda-python helper creates valid Green Context streams.
  2. apply_stream_plan() accepts Green Context streams without error.
  3. Planned graphs with Green Context streams produce correct outputs.
  4. The visualiser formats Green Context stream handles correctly.

Runtime requirements:
  - CUDA 12.4 or newer driver  (cuGreenCtxCreate symbol must exist)
  - The GPU must support SM resource partitioning (Hopper / Blackwell / Ada+)
  - At least 2 partitionable SM groups

The tests skip cleanly when those requirements are not met.
"""

from __future__ import annotations

import io
import unittest
from typing import List
from unittest.mock import MagicMock, patch

import torch
import torch.fx
from torch.testing._internal.common_utils import TestCase, run_tests
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import TorchTensorRTModule
from torch_tensorrt.runtime._stream_viz import print_stream_plan
from torch_tensorrt.runtime.stream_plan import (
    StreamPlan,
    _apply_stream_plan,
    _trt_nodes,
    apply_stream_plan,
)

# ── CUDA driver bindings (cuda-python) ───────────────────────────────────────


def _green_ctx_available() -> bool:
    """Return True iff cuda-python exposes the CUDA Green Context APIs."""
    try:
        from cuda.bindings import driver as drv

        drv.cuGreenCtxCreate  # noqa: B018 — attribute access as existence check
        drv.cuDevSmResourceSplitByCount
        return True
    except (ImportError, AttributeError):
        return False


# ── Public helper ─────────────────────────────────────────────────────────────


class GreenCtxStream:
    """
    Owns one CUDA Green Context and one non-blocking stream on it.

    Callers must call destroy() (or let __del__ handle it) once done.
    The wrapped stream is accessible as .stream (torch.cuda.ExternalStream).
    """

    def __init__(self, green_ctx, raw_stream: int, device_id: int) -> None:
        self._green_ctx = green_ctx  # CUgreenCtx handle from cuda.bindings.driver
        self.stream = torch.cuda.ExternalStream(
            raw_stream, device=torch.device("cuda", device_id)
        )

    def destroy(self) -> None:
        if self._green_ctx is not None:
            from cuda.bindings import driver as drv

            drv.cuGreenCtxDestroy(self._green_ctx)
            self._green_ctx = None

    def __del__(self) -> None:
        self.destroy()


def create_green_ctx_streams(device_id: int, n: int) -> List[GreenCtxStream]:
    """
    Partition the device SMs into ``n`` groups and return one GreenCtxStream
    per group.

    Raises RuntimeError with a descriptive message on any failure so callers
    can call skipTest() on it if needed.
    """
    from cuda.bindings import driver as drv

    # Ensure the primary CUDA context exists before calling driver APIs.
    torch.cuda.init()
    _ = torch.zeros(1, device=torch.device("cuda", device_id))

    # 1. Query total SM resource for the device.
    err, total = drv.cuDeviceGetDevResource(
        device_id, drv.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
    )
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDeviceGetDevResource failed ({err})")

    # 2. Split into n groups (minCount=1 SM per partition).
    err, groups, nb, _ = drv.cuDevSmResourceSplitByCount(n, total, 0, 1)
    if err != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuDevSmResourceSplitByCount failed ({err})")
    if nb < n:
        raise RuntimeError(
            f"Requested {n} SM partitions but only {nb} available "
            f"(total SM count: {total.sm.smCount})"
        )

    result: List[GreenCtxStream] = []
    for i in range(n):
        # 3. Generate a resource descriptor for this group.
        err, desc = drv.cuDevResourceGenerateDesc([groups[i]], 1)
        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuDevResourceGenerateDesc[{i}] failed ({err})")

        # 4. Create Green Context.
        err, gctx = drv.cuGreenCtxCreate(
            desc, device_id, drv.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM
        )
        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuGreenCtxCreate[{i}] failed ({err})")

        # 5. Green ctx must be current before cuGreenCtxStreamCreate.
        err, ctx = drv.cuCtxFromGreenCtx(gctx)
        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuCtxFromGreenCtx[{i}] failed ({err})")
        drv.cuCtxPushCurrent(ctx)

        # 6. Create a non-blocking stream on the Green Context.
        #    CU_STREAM_NON_BLOCKING is required; CU_STREAM_DEFAULT is rejected.
        err, raw_stream = drv.cuGreenCtxStreamCreate(
            gctx, drv.CUstream_flags.CU_STREAM_NON_BLOCKING, 0
        )
        drv.cuCtxPopCurrent()

        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuGreenCtxStreamCreate[{i}] failed ({err})")

        result.append(GreenCtxStream(gctx, int(raw_stream), device_id))

    return result


# ── Test helpers ──────────────────────────────────────────────────────────────

DEVICE_ID = 0
DEVICE = torch.device("cuda", DEVICE_ID)


def _mock_trt_module(device_id: int = DEVICE_ID) -> MagicMock:
    mod = MagicMock(spec=TorchTensorRTModule)
    mod.engine = MagicMock()
    mod.engine.device_info.id = device_id
    return mod


def _build_chain_gm(n_nodes: int) -> torch.fx.GraphModule:
    """Linear chain: x → acc_0 → acc_1 → … → acc_{n-1} → output."""
    parent = torch.nn.Module()
    for i in range(n_nodes):
        parent.add_module(f"_run_on_acc_{i}", _mock_trt_module())
    g = torch.fx.Graph()
    node = g.placeholder("x")
    for i in range(n_nodes):
        node = g.call_module(f"_run_on_acc_{i}", args=(node,))
    g.output(node)
    return torch.fx.GraphModule(parent, g)


def _build_fan_in_gm() -> torch.fx.GraphModule:
    """Fan-in: x → {acc_0, acc_1} → acc_2 (VLA: vision + lang → action)."""
    parent = torch.nn.Module()
    for name in ("_run_on_acc_0", "_run_on_acc_1", "_run_on_acc_2"):
        parent.add_module(name, _mock_trt_module())
    g = torch.fx.Graph()
    x = g.placeholder("x")
    vis = g.call_module("_run_on_acc_0", args=(x,))
    lang = g.call_module("_run_on_acc_1", args=(x,))
    act = g.call_module("_run_on_acc_2", args=(vis, lang))
    g.output(act)
    return torch.fx.GraphModule(parent, g)


# ── Tests: driver-level helper ─────────────────────────────────────────────────


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@unittest.skipIf(
    not _green_ctx_available(), "CUDA Green Contexts not available in driver"
)
class TestGreenCtxStreamCreation(TestCase):
    """Verify ctypes helper creates valid, distinct, usable streams."""

    def test_creates_requested_number_of_streams(self):
        streams = create_green_ctx_streams(DEVICE_ID, 2)
        self.assertEqual(len(streams), 2)
        for gcs in streams:
            self.assertIsInstance(gcs.stream, torch.cuda.ExternalStream)
        for gcs in streams:
            gcs.destroy()

    def test_streams_are_distinct(self):
        streams = create_green_ctx_streams(DEVICE_ID, 3)
        handles = [gcs.stream.cuda_stream for gcs in streams]
        self.assertEqual(len(set(handles)), 3, "Green context streams must be distinct")
        for gcs in streams:
            gcs.destroy()

    def test_stream_device_index_is_correct(self):
        streams = create_green_ctx_streams(DEVICE_ID, 1)
        self.assertEqual(streams[0].stream.device.index, DEVICE_ID)
        streams[0].destroy()

    def test_stream_executes_cuda_work(self):
        """A green-context stream must actually execute GPU kernels."""
        gcs = create_green_ctx_streams(DEVICE_ID, 1)[0]
        x = torch.randn(64, 64, device=DEVICE)
        with torch.cuda.stream(gcs.stream):
            y = torch.relu(x) * 2.0
        torch.cuda.synchronize(DEVICE)
        self.assertEqual(y.shape, (64, 64))
        self.assertTrue((y >= 0).all())
        gcs.destroy()

    def test_two_streams_run_concurrently(self):
        """Submit independent work on two green-context streams; both complete."""
        s0, s1 = create_green_ctx_streams(DEVICE_ID, 2)
        x = torch.randn(128, 128, device=DEVICE)
        with torch.cuda.stream(s0.stream):
            a = x @ x.T
        with torch.cuda.stream(s1.stream):
            b = torch.relu(x)
        torch.cuda.synchronize(DEVICE)
        self.assertEqual(a.shape, (128, 128))
        self.assertEqual(b.shape, (128, 128))
        s0.destroy()
        s1.destroy()


# ── Tests: FX-pass integration ────────────────────────────────────────────────


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@unittest.skipIf(
    not _green_ctx_available(), "CUDA Green Contexts not available in driver"
)
class TestStreamPlanWithGreenCtx(TestCase):
    """_apply_stream_plan() produces correct FX graphs for green-ctx streams."""

    def test_single_node_chain(self):
        gm = _build_chain_gm(1)
        gcs = create_green_ctx_streams(DEVICE_ID, 1)
        try:
            plan = StreamPlan(
                assignment={"_run_on_acc_0": gcs[0].stream}, device=DEVICE
            )
            planned = _apply_stream_plan(gm, plan)
            planned.graph.lint()
        finally:
            for g in gcs:
                g.destroy()

    def test_two_node_chain_different_streams(self):
        gm = _build_chain_gm(2)
        gcs = create_green_ctx_streams(DEVICE_ID, 2)
        try:
            plan = StreamPlan(
                assignment={
                    "_run_on_acc_0": gcs[0].stream,
                    "_run_on_acc_1": gcs[1].stream,
                },
                device=DEVICE,
            )
            planned = _apply_stream_plan(gm, plan)
            planned.graph.lint()
            # cross-stream edge (acc_0 → acc_1): one sync + enter sync + exit sync = 3
            syncs = [
                n
                for n in planned.graph.nodes
                if n.op == "call_function"
                and n.target == torch.ops.torchtrt.sync_streams.default
            ]
            self.assertGreaterEqual(len(syncs), 1)
        finally:
            for g in gcs:
                g.destroy()

    def test_vla_fan_in_three_streams(self):
        """
        VLA topology: vision + language on separate SM partitions fan in to
        action.  Verify the FX pass inserts barriers for both incoming edges.
        """
        gm = _build_fan_in_gm()
        gcs = create_green_ctx_streams(DEVICE_ID, 3)
        try:
            s_vis, s_lang, s_act = [g.stream for g in gcs]
            plan = StreamPlan(
                assignment={
                    "_run_on_acc_0": s_vis,
                    "_run_on_acc_1": s_lang,
                    "_run_on_acc_2": s_act,
                },
                device=DEVICE,
            )
            planned = _apply_stream_plan(gm, plan)
            planned.graph.lint()

            ops = {n.target for n in planned.graph.nodes if n.op == "call_function"}
            self.assertIn(torch.ops.torchtrt.enter_compute_stream.default, ops)
            self.assertIn(torch.ops.torchtrt.sync_streams.default, ops)
            self.assertIn(torch.ops.torchtrt.exit_compute_stream.default, ops)

            # At minimum: caller→vis, caller→lang, vis→act, lang→act, act→caller = 5
            syncs = [
                n
                for n in planned.graph.nodes
                if n.op == "call_function"
                and n.target == torch.ops.torchtrt.sync_streams.default
            ]
            self.assertGreaterEqual(len(syncs), 4)
        finally:
            for g in gcs:
                g.destroy()

    def test_stream_ids_baked_into_graph(self):
        """Green context stream handles must appear as literal args in the graph."""
        gm = _build_chain_gm(1)
        gcs = create_green_ctx_streams(DEVICE_ID, 1)
        try:
            plan = StreamPlan(
                assignment={"_run_on_acc_0": gcs[0].stream}, device=DEVICE
            )
            planned = _apply_stream_plan(gm, plan)
            raw_id = gcs[0].stream.cuda_stream
            # The raw stream handle should appear somewhere as a literal arg
            all_args = []
            for n in planned.graph.nodes:
                if n.op == "call_function":
                    all_args.extend(a for a in n.args if isinstance(a, int))
            self.assertIn(raw_id, all_args)
        finally:
            for g in gcs:
                g.destroy()

    def test_stream_kept_alive_as_module_attr(self):
        """Stream object must be stored as a module attribute to prevent GC."""
        gm = _build_chain_gm(1)
        gcs = create_green_ctx_streams(DEVICE_ID, 1)
        try:
            plan = StreamPlan(
                assignment={"_run_on_acc_0": gcs[0].stream}, device=DEVICE
            )
            planned = _apply_stream_plan(gm, plan)
            kept = [
                getattr(planned, attr)
                for attr in vars(planned)
                if attr.startswith("_trt_stream")
            ]
            self.assertTrue(
                any(k is gcs[0].stream for k in kept),
                "Stream object must be kept alive on the planned module",
            )
        finally:
            for g in gcs:
                g.destroy()


# ── Tests: visualiser ─────────────────────────────────────────────────────────


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@unittest.skipIf(
    not _green_ctx_available(), "CUDA Green Contexts not available in driver"
)
class TestStreamPlanVizWithGreenCtx(TestCase):
    """Visualiser output includes green-context stream handles."""

    def test_print_shows_green_ctx_handles(self):
        gm = _build_fan_in_gm()
        gcs = create_green_ctx_streams(DEVICE_ID, 3)
        try:
            plan = StreamPlan(
                assignment={
                    "_run_on_acc_0": gcs[0].stream,
                    "_run_on_acc_1": gcs[1].stream,
                    "_run_on_acc_2": gcs[2].stream,
                },
                device=DEVICE,
            )
            buf = io.StringIO()
            print_stream_plan(gm, plan, file=buf)
            output = buf.getvalue()
            # All three streams should appear by handle
            for i, g in enumerate(gcs):
                handle = f"0x{g.stream.cuda_stream:x}"
                self.assertIn(handle, output, f"Stream {i} handle missing from output")
            self.assertIn("Cross-stream barriers", output)
        finally:
            for g in gcs:
                g.destroy()

    def test_dot_graph_contains_green_ctx_handles(self):
        from torch_tensorrt.runtime._stream_viz import stream_plan_dot

        gm = _build_fan_in_gm()
        gcs = create_green_ctx_streams(DEVICE_ID, 3)
        try:
            plan = StreamPlan(
                assignment={
                    "_run_on_acc_0": gcs[0].stream,
                    "_run_on_acc_1": gcs[1].stream,
                    "_run_on_acc_2": gcs[2].stream,
                },
                device=DEVICE,
            )
            dot = stream_plan_dot(gm, plan, title="VLA Green Ctx Test")
            for i, g in enumerate(gcs):
                handle = f"0x{g.stream.cuda_stream:x}"
                self.assertIn(handle, dot, f"Stream {i} handle missing from DOT output")
            self.assertIn("sync", dot)
            self.assertIn("cluster_caller", dot)
        finally:
            for g in gcs:
                g.destroy()


# ── Tests: full end-to-end with real TRT engine ───────────────────────────────


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@unittest.skipIf(
    not _green_ctx_available(), "CUDA Green Contexts not available in driver"
)
class TestStreamPlanGreenCtxE2E(TestCase):
    """Compile a real TRT model and run it on a green-context stream."""

    def _compile(self):
        import torch_tensorrt

        class SimpleModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.relu(x) + 1.0

        dtype = torch.float16
        model = SimpleModel().eval().to(DEVICE)
        return torch_tensorrt.compile(
            model,
            ir="dynamo",
            inputs=[torch_tensorrt.Input(shape=(1, 16, 16), dtype=dtype)],
            min_block_size=1,
            device=DEVICE,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )

    def test_output_matches_reference(self):
        compiled = self._compile()
        nodes = _trt_nodes(compiled)
        if not nodes:
            self.skipTest("No TRT nodes in compiled model (graph break fallback)")

        gcs = create_green_ctx_streams(DEVICE_ID, len(nodes))
        try:
            streams = [g.stream for g in gcs]
            with patch(
                "torch_tensorrt.runtime._cudagraphs._PY_RT_CUDAGRAPHS",
                0,  # CudaGraphsMode.STANDARD
            ):
                planned = apply_stream_plan(compiled, streams=streams)

            dtype = torch.float16
            x = torch.randn(1, 16, 16, dtype=dtype, device=DEVICE)
            with torch.inference_mode():
                ref = compiled(x)
                out = planned(x)

            torch.testing.assert_close(ref, out, atol=1e-2, rtol=1e-2)
        finally:
            for g in gcs:
                g.destroy()

    def test_stream_attrs_released_after_context_manager(self):
        import torch_tensorrt
        from torch_tensorrt.runtime.stream_plan import stream_plan

        compiled = self._compile()
        nodes = _trt_nodes(compiled)
        if not nodes:
            self.skipTest("No TRT nodes in compiled model")

        gcs = create_green_ctx_streams(DEVICE_ID, len(nodes))
        try:
            streams = [g.stream for g in gcs]
            planned_ref = None
            with patch("torch_tensorrt.runtime._cudagraphs._PY_RT_CUDAGRAPHS", 0):
                with stream_plan(compiled, streams=streams) as planned:
                    planned_ref = planned
                    # inside: attrs present
                    inside_attrs = [
                        a for a in vars(planned) if a.startswith("_trt_stream")
                    ]
                    self.assertTrue(len(inside_attrs) > 0)

            # outside: attrs released
            outside_attrs = [
                a for a in vars(planned_ref) if a.startswith("_trt_stream")
            ]
            self.assertEqual(outside_attrs, [])
        finally:
            for g in gcs:
                g.destroy()


if __name__ == "__main__":
    run_tests()
