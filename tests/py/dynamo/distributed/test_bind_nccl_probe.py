"""
Tests for TRTEngine.bind_nccl_comm() — probe loop (detection) and binding.

Background
----------
bind_nccl_comm() has two distinct responsibilities that this file tests
independently:

  PART 1 — DETECTION (probe loop)
  --------------------------------
  Triggered when: group_name is empty — i.e. the save→load path where
  torch.export.load() deserializes the engine but bypasses
  _TorchTensorRTModule.__init__(), so set_group_name() is never called.

  The probe loop iterates numeric group names "0".."19" and collects all
  groups with an NCCL backend. PyTorch assigns numeric names via a monotonic
  group_count counter:
    - init_process_group()                       → always numeric
    - new_group(use_local_synchronization=False)  → numeric (default)
    - new_group(use_local_synchronization=True)   → hashed name, but still
                                                    increments group_count,
                                                    leaving a gap in numerics
  In PyTorch 2.x, resolve_process_group throws c10::Error for missing names
  instead of returning nullptr. The probe catches and continues (not breaks)
  so gaps don't stop it from finding groups beyond them (PR #4428).

  Outcomes:
    nccl_groups.size() == 1 → auto-resolve: group_name set, binding proceeds
    nccl_groups.size() >  1 → ambiguous: warns, defers — user must pin via
                               distributed_context(group, model)
    nccl_groups.size() == 0 → warns, defers — dist not initialized

  PART 2 — BINDING
  ----------------
  Triggered when: group_name is non-empty (either auto-resolved by the probe,
  or pre-set by distributed_context() / set_group_name() before first execute).

  On the torch.compile path, _TorchTensorRTModule.__init__() runs and calls
  get_active_group_name() → set_group_name() — so group_name is set BEFORE
  the first forward. The probe loop is never entered. bind_nccl_comm() goes
  straight to resolve_process_group(group_name) → extract ncclComm_t → bind
  to IExecutionContext.

  Key difference from detection:
    - Detection tests: probe loop RUNS (save→load path, group_name empty)
    - Binding tests:   probe loop NEVER RUNS (torch.compile path or explicit
                       pin via distributed_context, group_name pre-set)

Test structure
--------------
  TestBindNcclProbeDetection
    Verifies probe loop behavior: whether auto-resolve fires (single group)
    or deferral fires (multiple groups requiring explicit pin). Asserts on
    output correctness as a proxy — correct output confirms probe resolved
    and bound the right communicator.

  TestBindNcclProbeBinding
    Verifies binding produces numerically correct output. Uses torch.compile
    (no save/load) so probe loop is never entered — pure binding path.
    Covers: auto-resolve, explicit world group pin, explicit subgroup pin.

  TestBindNcclProbeE2E
    Covers both parts together via the save→load cycle:
    compile → save → load → first forward (probe fires → auto-resolve → bind).
    Asserts on output correctness end-to-end.

Run
---
    pytest distributed/test_bind_nccl_probe.py -v
    pytest distributed/test_bind_nccl_probe.py::TestBindNcclProbeDetection -v
    pytest distributed/test_bind_nccl_probe.py::TestBindNcclProbeBinding -v
    pytest distributed/test_bind_nccl_probe.py::TestBindNcclProbeE2E -v

    # via torchrun (2 GPUs):
    torchrun --nproc_per_node=2 distributed/test_bind_nccl_probe.py --multirank
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _has_nccl_collectives() -> bool:
    try:
        from torch_tensorrt._features import ENABLED_FEATURES

        return bool(ENABLED_FEATURES.native_trt_collectives) or bool(
            ENABLED_FEATURES.trtllm_for_nccl
        )
    except Exception:
        return False


def _check_close(a: torch.Tensor, b: torch.Tensor, name: str) -> None:
    try:
        torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)
        print(f"[PASS] {name}")
    except AssertionError as e:
        print(f"[FAIL] {name}: {e}")
        raise


def _world_group_name() -> str:
    g = dist.group.WORLD
    return str(g.group_name) if hasattr(g, "group_name") else ""


class _AllReduceModel(nn.Module):
    def __init__(self, group_name: str) -> None:
        super().__init__()
        self.group_name = group_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.ops._c10d_functional.all_reduce.default(x, "sum", self.group_name)
        return torch.ops._c10d_functional.wait_tensor.default(out)


def _compile_and_save(model, inp, save_path):
    """Export → TRT compile → save (path must end in .pt2). Returns compiled model."""
    import torch_tensorrt
    from torch_tensorrt.distributed._nccl_utils import initialize_nccl_comm

    assert save_path.endswith(".pt2"), f"save_path must end in .pt2, got {save_path}"
    initialize_nccl_comm()
    with torch.no_grad():
        ep = torch.export.export(model, (inp,))
    trt_model = torch_tensorrt.dynamo.compile(
        ep,
        inputs=[inp],
        use_python_runtime=False,
        min_block_size=1,
        use_distributed_mode_trace=True,
        enabled_precisions={torch.float32},
    )
    torch_tensorrt.save(trt_model, save_path, retrace=False)
    dist.barrier()
    return trt_model


def _load(save_path):
    """Load saved TRT engine. group_name will be empty — probe fires on first execute."""
    import torch_tensorrt
    from torch_tensorrt.distributed._nccl_utils import initialize_nccl_comm

    initialize_nccl_comm()
    return torch_tensorrt.load(save_path).module()


# ---------------------------------------------------------------------------
# PART 1 — DETECTION test functions
# ---------------------------------------------------------------------------


def _detect_single_group_probe_resolves(rank, world_size, device):
    """Only world group "0" exists — probe auto-resolves and binding succeeds.

    Group state: only "0" (world NCCL).
    Probe: i=0 found, i=1 throws (c10::Error) → nccl_groups=["0"] → auto-resolve.

    Assert: output after load matches output at compile time — confirms probe
    correctly resolved "0" and bound the communicator.
    """
    from torch_tensorrt.distributed._nccl_utils import setup_nccl_for_torch_tensorrt

    setup_nccl_for_torch_tensorrt()

    model = _AllReduceModel(_world_group_name()).to(device).eval()
    inp = torch.full((1, 4), float(rank + 1), device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        trt_model = _compile_and_save(model, inp, f"{tmpdir}/r{rank}.pt2")
        with torch.no_grad():
            out_compile = trt_model(inp)

        trt_loaded = _load(f"{tmpdir}/r{rank}.pt2")
        with torch.no_grad():
            out_load = trt_loaded(inp)

    _check_close(out_compile, out_load, f"detect_single_group rank={rank}")


def _detect_gap_from_local_sync(rank, world_size, device):
    """Probe skips gap from use_local_synchronization=True, finds both "0" and "2".

    Group state:
      "0" → world NCCL
      gap at "1" (use_local_sync consumed group_count, hashed name)
      "2" → new_group() numeric NCCL

    Probe must use continue (not break) at "1" to find both "0" and "2".
    Result: nccl_groups=["0","2"] → size=2 → deferred.

    Assert: with explicit distributed_context pin, output is correct — confirms
    probe found both groups (didn't stop at gap) and deferred correctly.
    """
    from torch_tensorrt.distributed._distributed import distributed_context
    from torch_tensorrt.distributed._nccl_utils import setup_nccl_for_torch_tensorrt

    setup_nccl_for_torch_tensorrt()

    _ = dist.new_group(ranks=list(range(world_size)), use_local_synchronization=True)
    sg = dist.new_group(ranks=list(range(world_size)))  # gets name "2"
    dist.barrier(group=sg)

    group = dist.group.WORLD
    model = _AllReduceModel(_world_group_name()).to(device).eval()
    inp = torch.full((1, 4), float(rank + 1), device=device)
    expected = torch.full(
        (1, 4), float(world_size * (world_size + 1) // 2), device=device
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        with distributed_context(group):
            _compile_and_save(model, inp, f"{tmpdir}/r{rank}.pt2")

        # Explicit pin needed because probe finds multiple groups → deferred.
        # Without this pin, forward would crash (no comm bound).
        trt_loaded = _load(f"{tmpdir}/r{rank}.pt2")
        with distributed_context(group, trt_loaded):
            with torch.no_grad():
                out = trt_loaded(inp)

    _check_close(out, expected, f"detect_gap_local_sync rank={rank}")


def _detect_multiple_groups_defers(rank, world_size, device):
    """Probe finds two NCCL groups ("0","1") → defers, requires explicit pin.

    Group state:
      "0" → world NCCL
      "1" → new_group() numeric NCCL

    Probe finds both → nccl_groups.size()==2 → cannot auto-select → deferred.
    Without explicit pin, forward crashes (TRT assertion comm != nullptr).

    Assert: with explicit distributed_context(group, module) pin, output correct.
    This confirms probe found both groups and correctly required the user to pin.
    """
    from torch_tensorrt.distributed._distributed import distributed_context
    from torch_tensorrt.distributed._nccl_utils import setup_nccl_for_torch_tensorrt

    setup_nccl_for_torch_tensorrt()

    tp_group = dist.new_group(ranks=list(range(world_size)))
    dist.barrier(group=tp_group)

    group = dist.group.WORLD
    model = _AllReduceModel(_world_group_name()).to(device).eval()
    inp = torch.full((1, 4), float(rank + 1), device=device)
    expected = torch.full(
        (1, 4), float(world_size * (world_size + 1) // 2), device=device
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        with distributed_context(group):
            _compile_and_save(model, inp, f"{tmpdir}/r{rank}.pt2")

        # Explicit pin required — probe finds "0" and "1" → defers.
        # distributed_context(group, module) calls set_group_name before
        # first execute → probe skipped → comm bound → correct output.
        trt_loaded = _load(f"{tmpdir}/r{rank}.pt2")
        with distributed_context(group, trt_loaded):
            with torch.no_grad():
                out = trt_loaded(inp)

    _check_close(out, expected, f"detect_multiple_groups_defers rank={rank}")


# ---------------------------------------------------------------------------
# PART 2 — BINDING test functions
# ---------------------------------------------------------------------------


def _bind_auto_resolved_single_group(rank, world_size, device):
    """Single NCCL group — probe auto-resolves, binding produces correct output.

    Tests the full binding path without save/load:
    distributed_context not needed — probe auto-resolves at first execute.
    """
    from torch_tensorrt.distributed._nccl_utils import (
        initialize_nccl_comm,
        setup_nccl_for_torch_tensorrt,
    )

    setup_nccl_for_torch_tensorrt()

    model = _AllReduceModel(_world_group_name()).to(device).eval()
    inp = torch.full((1, 4), float(rank + 1), device=device)
    expected = torch.full(
        (1, 4), float(world_size * (world_size + 1) // 2), device=device
    )

    initialize_nccl_comm()
    trt_model = torch.compile(
        model,
        backend="torch_tensorrt",
        dynamic=False,
        options={
            "use_python_runtime": False,
            "min_block_size": 1,
            "use_distributed_mode_trace": True,
        },
    )
    with torch.no_grad():
        out = trt_model(inp)

    _check_close(out, expected, f"bind_auto_resolved rank={rank}")


def _bind_explicit_pin_world_group(rank, world_size, device):
    """Explicit distributed_context pin — binding correct, probe skipped.

    distributed_context(group) calls set_group_name() before first execute,
    so group_name is non-empty and the probe loop condition is false.
    """
    from torch_tensorrt.distributed._distributed import distributed_context
    from torch_tensorrt.distributed._nccl_utils import setup_nccl_for_torch_tensorrt

    setup_nccl_for_torch_tensorrt()

    group = dist.group.WORLD
    model = _AllReduceModel(_world_group_name()).to(device).eval()
    inp = torch.full((1, 4), float(rank + 1), device=device)
    expected = torch.full(
        (1, 4), float(world_size * (world_size + 1) // 2), device=device
    )

    with distributed_context(group):
        trt_model = torch.compile(
            model,
            backend="torch_tensorrt",
            dynamic=False,
            options={
                "use_python_runtime": False,
                "min_block_size": 1,
                "use_distributed_mode_trace": True,
            },
        )
        with torch.no_grad():
            out = trt_model(inp)

    _check_close(out, expected, f"bind_explicit_pin_world rank={rank}")


def _bind_explicit_pin_subgroup(rank, world_size, device):
    """Explicit pin to a TP subgroup — correct subgroup comm bound, not world comm.

    Creates world group "0" and TP subgroup "1". Pins TP subgroup explicitly.
    Verifies the TP subgroup all_reduce produces the same result as world
    all_reduce (since both contain all ranks in this 2-rank test) — confirming
    the engine used the pinned subgroup comm, not the world comm or nothing.
    """
    from torch_tensorrt.distributed._distributed import distributed_context
    from torch_tensorrt.distributed._nccl_utils import setup_nccl_for_torch_tensorrt

    setup_nccl_for_torch_tensorrt()

    tp_group = dist.new_group(ranks=list(range(world_size)))
    dist.barrier(group=tp_group)
    tp_name = tp_group.group_name

    model = _AllReduceModel(tp_name).to(device).eval()
    inp = torch.full((1, 4), float(rank + 1), device=device)
    expected = torch.full(
        (1, 4), float(world_size * (world_size + 1) // 2), device=device
    )

    with distributed_context(tp_group):
        trt_model = torch.compile(
            model,
            backend="torch_tensorrt",
            dynamic=False,
            options={
                "use_python_runtime": False,
                "min_block_size": 1,
                "use_distributed_mode_trace": True,
            },
        )
        with torch.no_grad():
            out = trt_model(inp)

    _check_close(out, expected, f"bind_explicit_pin_subgroup rank={rank}")


# ---------------------------------------------------------------------------
# PART 1 + 2 — E2E save→load test functions
# ---------------------------------------------------------------------------


def _e2e_single_group_save_load(rank, world_size, device):
    """Regression test for PR #4428: save→load with single world group.

    At load time group_name="" → probe fires → auto-resolves "0" → binding
    produces correct output. Without the fix, probe threw on i=1.
    """
    from torch_tensorrt.distributed._nccl_utils import setup_nccl_for_torch_tensorrt

    setup_nccl_for_torch_tensorrt()

    model = _AllReduceModel(_world_group_name()).to(device).eval()
    inp = torch.full((1, 4), float(rank + 1), device=device)
    expected = torch.full(
        (1, 4), float(world_size * (world_size + 1) // 2), device=device
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        _compile_and_save(model, inp, f"{tmpdir}/r{rank}.pt2")
        trt_loaded = _load(f"{tmpdir}/r{rank}.pt2")
        with torch.no_grad():
            out_load = trt_loaded(inp)

    _check_close(out_load, expected, f"e2e_single_group load rank={rank}")


def _e2e_multi_group_save_load_with_pin(rank, world_size, device):
    """Multi-group save→load requires explicit distributed_context at load time.

    At save time: world "0" + tp_group "1" exist → probe would be ambiguous.
    Use distributed_context(world) at both save and load to pin explicitly.
    Verifies explicit pin resolves ambiguity end-to-end.
    """
    from torch_tensorrt.distributed._distributed import distributed_context
    from torch_tensorrt.distributed._nccl_utils import setup_nccl_for_torch_tensorrt

    setup_nccl_for_torch_tensorrt()

    tp_group = dist.new_group(ranks=list(range(world_size)))
    dist.barrier(group=tp_group)

    group = dist.group.WORLD
    model = _AllReduceModel(_world_group_name()).to(device).eval()
    inp = torch.full((1, 4), float(rank + 1), device=device)
    expected = torch.full(
        (1, 4), float(world_size * (world_size + 1) // 2), device=device
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        with distributed_context(group):
            _compile_and_save(model, inp, f"{tmpdir}/r{rank}.pt2")

        # Pass module to distributed_context so set_group_name() pre-pins
        # the engine before first execute, bypassing the ambiguous probe.
        trt_loaded = _load(f"{tmpdir}/r{rank}.pt2")
        with distributed_context(group, trt_loaded):
            with torch.no_grad():
                out = trt_loaded(inp)

    _check_close(out, expected, f"e2e_multi_group_with_pin rank={rank}")


# ---------------------------------------------------------------------------
# pytest test classes
# ---------------------------------------------------------------------------


@unittest.skipIf(
    not _has_nccl_collectives(),
    "Skipped: No NCCL collective support.",
)
class TestBindNcclProbeDetection(MultiProcessTestCase):
    """Part 1 — probe loop finds the right groups.

    Tests verify that the probe correctly detects process groups without
    asserting on output correctness (that's Part 2 / E2E).

        pytest distributed/test_bind_nccl_probe.py::TestBindNcclProbeDetection -v
    """

    world_size = 2

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def _init_dist(self) -> torch.device:
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        local = self.rank % torch.cuda.device_count()
        torch.cuda.set_device(local)
        dist.barrier()
        return torch.device(f"cuda:{local}")

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_single_group_probe_resolves(self) -> None:
        """Single world group — probe finds "0", auto-resolves."""
        device = self._init_dist()
        _detect_single_group_probe_resolves(self.rank, self.world_size, device)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_gap_from_local_sync(self) -> None:
        """Probe continues past gap from use_local_synchronization=True."""
        device = self._init_dist()
        _detect_gap_from_local_sync(self.rank, self.world_size, device)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_multiple_groups_defers(self) -> None:
        """Probe finds multiple NCCL groups — defers, explicit pin required."""
        device = self._init_dist()
        _detect_multiple_groups_defers(self.rank, self.world_size, device)


@unittest.skipIf(
    not _has_nccl_collectives(),
    "Skipped: No NCCL collective support.",
)
class TestBindNcclProbeBinding(MultiProcessTestCase):
    """Part 2 — correct communicator gets bound, output is numerically correct.

    Tests use torch.compile (no save/load) to focus on binding behavior.

        pytest distributed/test_bind_nccl_probe.py::TestBindNcclProbeBinding -v
    """

    world_size = 2

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def _init_dist(self) -> torch.device:
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        local = self.rank % torch.cuda.device_count()
        torch.cuda.set_device(local)
        dist.barrier()
        return torch.device(f"cuda:{local}")

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_auto_resolved_single_group(self) -> None:
        """Auto-resolved world group — binding produces correct output."""
        device = self._init_dist()
        _bind_auto_resolved_single_group(self.rank, self.world_size, device)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_explicit_pin_world_group(self) -> None:
        """Explicit pin to world group — probe skipped, correct output."""
        device = self._init_dist()
        _bind_explicit_pin_world_group(self.rank, self.world_size, device)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_explicit_pin_subgroup(self) -> None:
        """Explicit pin to TP subgroup — subgroup comm bound, correct output."""
        device = self._init_dist()
        _bind_explicit_pin_subgroup(self.rank, self.world_size, device)


@unittest.skipIf(
    not _has_nccl_collectives(),
    "Skipped: No NCCL collective support.",
)
class TestBindNcclProbeE2E(MultiProcessTestCase):
    """Part 1 + 2 — save→load path end-to-end.

    Tests the full cycle: compile → save → load → first execute.
    group_name is empty on load (torch.export.load bypasses __init__),
    so probe fires on first execute.

        pytest distributed/test_bind_nccl_probe.py::TestBindNcclProbeE2E -v
    """

    world_size = 2

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def _init_dist(self) -> torch.device:
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        local = self.rank % torch.cuda.device_count()
        torch.cuda.set_device(local)
        dist.barrier()
        return torch.device(f"cuda:{local}")

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_single_group_save_load(self) -> None:
        """Regression test for PR #4428: save→load, probe auto-resolves world group."""
        device = self._init_dist()
        _e2e_single_group_save_load(self.rank, self.world_size, device)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_multi_group_save_load_with_pin(self) -> None:
        """Multi-group save→load: explicit distributed_context at both ends."""
        device = self._init_dist()
        _e2e_multi_group_save_load_with_pin(self.rank, self.world_size, device)


# ---------------------------------------------------------------------------
# torchrun entry point
# ---------------------------------------------------------------------------


def _run_multirank() -> None:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")
    # Seed ncclComm_t so initialize_nccl_comm() finds a non-null handle.
    dist.barrier()

    detection_tests = [
        _detect_single_group_probe_resolves,
        _detect_gap_from_local_sync,
        _detect_multiple_groups_defers,
    ]
    binding_tests = [
        _bind_auto_resolved_single_group,
        _bind_explicit_pin_world_group,
        _bind_explicit_pin_subgroup,
    ]
    e2e_tests = [
        _e2e_single_group_save_load,
        _e2e_multi_group_save_load_with_pin,
    ]

    failed = []
    for section, tests in [
        ("Detection", detection_tests),
        ("Binding", binding_tests),
        ("E2E", e2e_tests),
    ]:
        if rank == 0:
            print(f"\n=== {section} ===")
        for fn in tests:
            dist.barrier()
            try:
                fn(rank, world_size, device)
            except Exception as e:
                failed.append((fn.__name__, str(e)))
                if rank == 0:
                    print(f"[FAIL] {fn.__name__}: {e}")

    dist.barrier()
    dist.destroy_process_group()

    total = len(detection_tests) + len(binding_tests) + len(e2e_tests)
    if failed:
        if rank == 0:
            print(f"\n{len(failed)}/{total} tests FAILED:")
            for name, err in failed:
                print(f"  - {name}: {err}")
        os._exit(1)
    else:
        if rank == 0:
            print(f"\nAll {total} tests PASSED.")
    # os._exit avoids SIGSEGV from TRT/CUDA destructors running in wrong order
    # during Python interpreter shutdown (debug build only).
    os._exit(0)


if __name__ == "__main__":
    if "--multirank" in sys.argv:
        sys.argv.remove("--multirank")
        _run_multirank()
    else:
        run_tests()
