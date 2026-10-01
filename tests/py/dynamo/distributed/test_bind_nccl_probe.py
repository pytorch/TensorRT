# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Native NCCL binding tests for fresh and saved TensorRT engines.

Python validates WORLD before passing its actual process-group name to C++.
Newly compiled wrappers perform this setup automatically. Loaded C++ engines
must be configured with distributed_context(dist.group.WORLD, loaded_model)
before execution; they never choose a group by scanning the registry.

Tests cover missing setup with one or several groups, arbitrary registry gaps,
WORLD binding, subgroup rejection, and save/load followed by explicit setup.

Run: pytest distributed/test_bind_nccl_probe.py -v
Or:  torchrun --nproc_per_node=2 distributed/test_bind_nccl_probe.py --multirank
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

        return bool(ENABLED_FEATURES.native_trt_collectives)
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
    """Load an engine whose C++ process-group name is not configured."""
    import torch_tensorrt
    from torch_tensorrt.distributed._nccl_utils import initialize_nccl_comm

    initialize_nccl_comm()
    return torch_tensorrt.load(save_path).module()


# ---------------------------------------------------------------------------
# PART 1 — LOADED ENGINE SETUP test functions
# ---------------------------------------------------------------------------


def _detect_single_group_probe_resolves(rank, world_size, device):
    """A loaded native engine requires explicit setup even when only WORLD exists."""
    from torch_tensorrt.distributed import distributed_context
    from torch_tensorrt.distributed._nccl_utils import setup_nccl_for_torch_tensorrt

    setup_nccl_for_torch_tensorrt()

    model = _AllReduceModel(_world_group_name()).to(device).eval()
    inp = torch.full((1, 4), float(rank + 1), device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        trt_model = _compile_and_save(model, inp, f"{tmpdir}/r{rank}.pt2")
        with torch.no_grad():
            out_compile = trt_model(inp)

        trt_loaded = _load(f"{tmpdir}/r{rank}.pt2")
        with unittest.TestCase().assertRaisesRegex(
            RuntimeError, "no process group configured"
        ):
            trt_loaded(inp)
        with distributed_context(dist.group.WORLD, trt_loaded):
            with torch.no_grad():
                out_load = trt_loaded(inp)

    _check_close(out_compile, out_load, f"detect_single_group rank={rank}")


def _detect_gap_from_local_sync(rank, world_size, device):
    """Explicit WORLD setup works with hashed groups and gaps in numeric names."""
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

        # Registry contents do not establish which group is WORLD.
        trt_loaded = _load(f"{tmpdir}/r{rank}.pt2")
        with unittest.TestCase().assertRaisesRegex(
            RuntimeError, "no process group configured"
        ):
            trt_loaded(inp)
        with distributed_context(group, trt_loaded):
            with torch.no_grad():
                out = trt_loaded(inp)

    _check_close(out, expected, f"detect_gap_local_sync rank={rank}")


def _detect_multiple_groups_defers(rank, world_size, device):
    """A loaded engine never guesses a parent when multiple groups exist."""
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

        # Configure the validated WORLD name before execution.
        trt_loaded = _load(f"{tmpdir}/r{rank}.pt2")
        with unittest.TestCase().assertRaisesRegex(
            RuntimeError, "no process group configured"
        ):
            trt_loaded(inp)
        with distributed_context(group, trt_loaded):
            with torch.no_grad():
                out = trt_loaded(inp)

    _check_close(out, expected, f"detect_multiple_groups_defers rank={rank}")


# ---------------------------------------------------------------------------
# PART 2 — BINDING test functions
# ---------------------------------------------------------------------------


def _bind_auto_resolved_single_group(rank, world_size, device):
    """A newly compiled wrapper validates and pins the default WORLD group."""
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
    """Explicit WORLD setup passes the actual group name to the C++ engine."""
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
    """A loaded global-ID engine rejects subgroup binding and still accepts WORLD."""
    from torch_tensorrt.distributed._distributed import distributed_context
    from torch_tensorrt.distributed._nccl_utils import setup_nccl_for_torch_tensorrt

    setup_nccl_for_torch_tensorrt()
    tp_group = dist.new_group(ranks=list(range(world_size)))
    model = _AllReduceModel(_world_group_name()).to(device).eval()
    inp = torch.full((1, 4), float(rank + 1), device=device)
    expected = torch.full(
        (1, 4), float(world_size * (world_size + 1) // 2), device=device
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        with distributed_context(dist.group.WORLD):
            _compile_and_save(model, inp, f"{tmpdir}/r{rank}.pt2")
        loaded = _load(f"{tmpdir}/r{rank}.pt2")
        with unittest.TestCase().assertRaisesRegex(
            RuntimeError, "require the WORLD communicator"
        ):
            with distributed_context(tp_group, loaded):
                pass
        with distributed_context(dist.group.WORLD, loaded):
            with torch.no_grad():
                out = loaded(inp)
    _check_close(out, expected, f"WORLD after rejected subgroup pin rank={rank}")


# ---------------------------------------------------------------------------
# PART 1 + 2 — E2E save→load test functions
# ---------------------------------------------------------------------------


def _e2e_single_group_save_load(rank, world_size, device):
    """A saved engine executes correctly after explicit WORLD setup."""
    from torch_tensorrt.distributed import distributed_context
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
        with distributed_context(dist.group.WORLD, trt_loaded):
            with torch.no_grad():
                out_load = trt_loaded(inp)

    _check_close(out_load, expected, f"e2e_single_group load rank={rank}")


def _e2e_multi_group_save_load_with_pin(rank, world_size, device):
    """Explicit WORLD setup works after save/load with several registered groups."""
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
        # the engine before its first execution.
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
    "Skipped: No native NCCL collective support.",
)
class TestBindNcclProbeDetection(MultiProcessTestCase):
    """Loaded native engines require explicit setup, regardless of registry contents."""

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
        """A loaded engine requires WORLD setup even with only one group."""
        device = self._init_dist()
        _detect_single_group_probe_resolves(self.rank, self.world_size, device)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_gap_from_local_sync(self) -> None:
        """Explicit WORLD setup is independent of gaps in registry names."""
        device = self._init_dist()
        _detect_gap_from_local_sync(self.rank, self.world_size, device)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_multiple_groups_defers(self) -> None:
        """Multiple registered groups never trigger automatic selection."""
        device = self._init_dist()
        _detect_multiple_groups_defers(self.rank, self.world_size, device)


@unittest.skipIf(
    not _has_nccl_collectives(),
    "Skipped: No native NCCL collective support.",
)
class TestBindNcclProbeBinding(MultiProcessTestCase):
    """Newly compiled and loaded engines bind the validated WORLD communicator."""

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
        """A newly compiled wrapper configures WORLD without an explicit context."""
        device = self._init_dist()
        _bind_auto_resolved_single_group(self.rank, self.world_size, device)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_explicit_pin_world_group(self) -> None:
        """Explicit WORLD setup produces correct output."""
        device = self._init_dist()
        _bind_explicit_pin_world_group(self.rank, self.world_size, device)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_explicit_pin_subgroup(self) -> None:
        """A loaded native engine rejects a subgroup parent and accepts WORLD."""
        device = self._init_dist()
        _bind_explicit_pin_subgroup(self.rank, self.world_size, device)

    @unittest.skipUnless(
        hasattr(dist, "ProcessGroupNCCL2"),
        "ProcessGroupNCCL2 is unavailable in this PyTorch build",
    )
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl2_backend(self) -> None:
        """Header-detected NCCL2 support binds the default NCCL communicator."""
        device = self._init_dist()
        backend = dist.get_backend_impl(dist.group.WORLD)
        if not isinstance(backend, dist.ProcessGroupNCCL2):
            self.skipTest("ProcessGroupNCCL2 is not the default NCCL backend")

        # Do not force NCCL2 through TORCH_DIST_USE_NCCL2 here. Successful
        # binding proves that build-time detection compiled support for the
        # backend selected naturally by the installed PyTorch.
        _bind_explicit_pin_world_group(self.rank, self.world_size, device)


@unittest.skipIf(
    not _has_nccl_collectives(),
    "Skipped: No native NCCL collective support.",
)
class TestBindNcclProbeE2E(MultiProcessTestCase):
    """Save/load retains engine contents; execution requires fresh parent setup."""

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
        """Save/load followed by explicit WORLD setup produces correct output."""
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
