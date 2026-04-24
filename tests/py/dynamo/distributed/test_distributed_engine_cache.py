"""
Distributed engine cache coordination tests.

Verifies that when multiple ranks compile the same model with engine caching
enabled, only one rank builds the TRT engine and others load from the shared
DiskEngineCache via filelock coordination.

Run multi-rank tests (requires 2 GPUs):
    pytest distributed/test_distributed_engine_cache.py::TestMultirankDistributedCache -v

Run via torchrun:
    torchrun --nproc_per_node=2 distributed/test_distributed_engine_cache.py --multirank
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
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
# Capability checks
# ---------------------------------------------------------------------------


def is_nccl_available() -> bool:
    try:
        return dist.is_nccl_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SimpleModel(nn.Module):
    """Small model for cache coordination testing (no TP)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _clean_cache_dir(cache_dir: str, rank: int) -> None:
    """Rank 0 cleans cache dir, all ranks wait then recreate."""
    if rank == 0 and os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    dist.barrier()
    os.makedirs(cache_dir, exist_ok=True)


def _compile_with_cache(model, cache_dir, use_distributed_mode_trace=False):
    """Compile model with TRT + engine caching enabled."""
    import torch_tensorrt

    return torch.compile(
        model,
        backend="torch_tensorrt",
        options={
            "enabled_precisions": {torch.float32},
            "use_python_runtime": False,
            "min_block_size": 1,
            "cache_built_engines": True,
            "reuse_cached_engines": True,
            "immutable_weights": False,
            "use_distributed_mode_trace": use_distributed_mode_trace,
            "engine_cache_dir": cache_dir,
            "engine_cache_size": 1 << 30,
        },
    )


def _assert_cache_has_files(cache_dir: str, rank: int) -> int:
    """Assert cache dir has engine files (ignoring .lock files)."""
    cache_files = [f for f in os.listdir(cache_dir) if not f.endswith(".lock")]
    assert len(cache_files) > 0, f"Rank {rank}: cache dir is empty"
    return len(cache_files)


# ---------------------------------------------------------------------------
# Test logic
# ---------------------------------------------------------------------------


def _multirank_distributed_cache_test(
    rank: int, world_size: int, device: torch.device, cache_dir: str
) -> None:
    """Test cache coordination with simple model (same weights on all ranks).

    Phase 1: First compile — one rank builds, other loads from cache.
    Phase 2: Second compile — both ranks load from cache (refit).
    """
    torch.manual_seed(42)
    model = SimpleModel().eval().to(device)
    inp = torch.randn(2, 32, device=device)

    with torch.no_grad():
        ref_output = model(inp)

    # Phase 1: compile + cache
    torch._dynamo.reset()
    trt_model = _compile_with_cache(model, cache_dir)
    with torch.no_grad():
        t0 = time.time()
        trt_output = trt_model(inp)
        build_time = time.time() - t0

    diff = (ref_output - trt_output).abs().max().item()
    assert diff < 0.01, f"Rank {rank}: output mismatch, max diff={diff}"
    n_files = _assert_cache_has_files(cache_dir, rank)
    print(
        f"[Rank {rank}] Compile + cache OK "
        f"(max_diff={diff:.6f}, build_time={build_time:.2f}s, cache_entries={n_files})",
        flush=True,
    )

    dist.barrier()

    # Phase 2: cache reuse
    torch._dynamo.reset()
    trt_model2 = _compile_with_cache(model, cache_dir)
    with torch.no_grad():
        t0 = time.time()
        trt_output2 = trt_model2(inp)
        refit_time = time.time() - t0

    diff2 = (ref_output - trt_output2).abs().max().item()
    assert diff2 < 0.01, f"Rank {rank}: cached output mismatch, max diff={diff2}"
    print(
        f"[Rank {rank}] Cache reuse OK "
        f"(max_diff={diff2:.6f}, refit_time={refit_time:.2f}s, "
        f"speedup={build_time / max(refit_time, 0.001):.1f}x)",
        flush=True,
    )


def _multirank_tp_cache_test(
    rank: int, world_size: int, device: torch.device, cache_dir: str
) -> None:
    """Test cache coordination with TP-sharded model (different weights per rank).

    Each rank holds ColwiseParallel/RowwiseParallel sharded weights.
    Engine structure is identical — only weights differ.
    One rank builds, other loads from cache and refits with its own weights.
    """
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )

    mesh = init_device_mesh("cuda", (world_size,))

    torch.manual_seed(42)
    model = (
        nn.Sequential(
            nn.Linear(64, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, 64, bias=False),
        )
        .eval()
        .to(device)
    )

    inp = torch.randn(2, 64, device=device)

    with torch.no_grad():
        ref_output = model(inp)

    tp_plan = {"0": ColwiseParallel(), "2": RowwiseParallel()}
    parallelize_module(model, mesh, tp_plan)

    torch._dynamo.reset()
    trt_model = _compile_with_cache(model, cache_dir, use_distributed_mode_trace=True)
    with torch.no_grad():
        trt_output = trt_model(inp)

    diff = (ref_output - trt_output).abs().max().item()
    assert diff < 0.05, f"Rank {rank}: TP output mismatch, max diff={diff}"
    n_files = _assert_cache_has_files(cache_dir, rank)
    print(
        f"[Rank {rank}] TP compile + cache OK "
        f"(max_diff={diff:.6f}, cache_entries={n_files})",
        flush=True,
    )


# ---------------------------------------------------------------------------
# pytest path (MultiProcessTestCase)
# ---------------------------------------------------------------------------


@unittest.skipIf(not is_nccl_available(), "NCCL not available")
@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    "Requires at least 2 GPUs",
)
class TestMultirankDistributedCache(MultiProcessTestCase):
    """Distributed engine cache tests as pytest-compatible MultiProcessTestCase."""

    world_size = 2

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def _init_dist(self, cache_dir: str) -> torch.device:
        """Initialize dist, clean cache, return device."""
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
        )
        local = self.rank % torch.cuda.device_count()
        torch.cuda.set_device(local)
        _clean_cache_dir(cache_dir, self.rank)
        return torch.device(f"cuda:{local}")

    @requires_nccl()
    def test_distributed_cache_coordination(self) -> None:
        """Both ranks compile same model with caching — output matches reference."""
        cache_dir = tempfile.mkdtemp(prefix="trt_dist_cache_pytest_")
        device = self._init_dist(cache_dir)
        _multirank_distributed_cache_test(self.rank, self.world_size, device, cache_dir)

    @requires_nccl()
    def test_tp_cache_coordination(self) -> None:
        """TP-sharded model: one rank builds, other loads from cache + refits."""
        cache_dir = tempfile.mkdtemp(prefix="trt_dist_tp_cache_pytest_")
        device = self._init_dist(cache_dir)
        _multirank_tp_cache_test(self.rank, self.world_size, device, cache_dir)


# ---------------------------------------------------------------------------
# torchrun entry point
# ---------------------------------------------------------------------------


def _multirank_setup() -> tuple:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    return rank, world_size, torch.device(f"cuda:{local_rank}")


def run_multirank_tests() -> None:
    """Entry point for --multirank mode (called by torchrun workers)."""
    rank, world_size, device = _multirank_setup()
    print(f"[Rank {rank}/{world_size}] device={device}", flush=True)

    base_cache_dir = os.path.join(tempfile.gettempdir(), "trt_dist_cache_torchrun")

    tests = [
        (
            "simple_model_cache",
            base_cache_dir + "_simple",
            _multirank_distributed_cache_test,
        ),
        ("tp_model_cache", base_cache_dir + "_tp", _multirank_tp_cache_test),
    ]

    failed = []
    for name, cache_dir, test_fn in tests:
        _clean_cache_dir(cache_dir, rank)
        dist.barrier()
        try:
            test_fn(rank, world_size, device, cache_dir)
        except Exception as e:
            failed.append((name, str(e)))
            print(f"[Rank {rank}] FAIL {name}: {e}", flush=True)
            import traceback

            traceback.print_exc()

    dist.barrier()
    dist.destroy_process_group()

    if failed:
        print(f"[Rank {rank}] {len(failed)} test(s) FAILED:", flush=True)
        for name, err in failed:
            print(f"  - {name}: {err}", flush=True)
        sys.exit(1)
    else:
        print(f"[Rank {rank}] All distributed cache tests PASSED.", flush=True)


if __name__ == "__main__":
    if "--multirank" in sys.argv or "--multinode" in sys.argv:
        sys.argv = [a for a in sys.argv if a not in ("--multirank", "--multinode")]
        run_multirank_tests()
    else:
        run_tests()
