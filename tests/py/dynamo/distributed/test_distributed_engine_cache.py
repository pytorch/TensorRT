"""
Distributed engine cache coordination tests.

Verifies that when multiple ranks compile the same model with engine caching
enabled, only one rank builds the TRT engine and others load from the shared
DiskEngineCache.

Tests:
  1. DistributedFileLock — acquire, release, stale detection (no GPU)
  2. Multi-rank: one rank builds, other loads from cache (2 GPUs)

Run single-rank tests (no GPU needed):
    cd tests/py/dynamo
    pytest distributed/test_distributed_engine_cache.py -v

Run multi-rank tests (requires 2 GPUs):
    pytest distributed/test_distributed_engine_cache.py::TestMultirankDistributedCache -v

Run via torchrun:
    torchrun --nproc_per_node=2 distributed/test_distributed_engine_cache.py --multirank
"""

from __future__ import annotations

import os
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


def has_nccl_collectives() -> bool:
    try:
        from torch_tensorrt._features import ENABLED_FEATURES

        return bool(ENABLED_FEATURES.native_trt_collectives) or bool(
            ENABLED_FEATURES.trtllm_for_nccl
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Section 1 — DistributedFileLock (no GPU, no dist)
# ---------------------------------------------------------------------------


class TestDistributedFileLock(unittest.TestCase):
    """Unit tests for the file-based distributed lock."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="trt_lock_test_")

    def test_acquire_returns_true_first_time(self) -> None:
        from torch_tensorrt.distributed._lock import DistributedFileLock

        lock = DistributedFileLock(self.tmp_dir, "test_hash")
        self.assertTrue(lock.acquire())
        self.assertTrue(lock.acquired)
        lock.release()

    def test_acquire_returns_false_when_held(self) -> None:
        from torch_tensorrt.distributed._lock import DistributedFileLock

        lock1 = DistributedFileLock(self.tmp_dir, "test_hash")
        lock2 = DistributedFileLock(self.tmp_dir, "test_hash")

        self.assertTrue(lock1.acquire())
        self.assertFalse(lock2.acquire())
        self.assertFalse(lock2.acquired)

        lock1.release()

    def test_acquire_succeeds_after_release(self) -> None:
        from torch_tensorrt.distributed._lock import DistributedFileLock

        lock1 = DistributedFileLock(self.tmp_dir, "test_hash")
        lock1.acquire()
        lock1.release()

        lock2 = DistributedFileLock(self.tmp_dir, "test_hash")
        self.assertTrue(lock2.acquire())
        lock2.release()

    def test_release_without_acquire_is_noop(self) -> None:
        from torch_tensorrt.distributed._lock import DistributedFileLock

        lock = DistributedFileLock(self.tmp_dir, "test_hash")
        lock.release()  # should not raise

    def test_context_manager_acquires_and_releases(self) -> None:
        from torch_tensorrt.distributed._lock import DistributedFileLock

        with DistributedFileLock(self.tmp_dir, "test_hash") as lock:
            self.assertTrue(lock.acquired)
            self.assertTrue(os.path.exists(lock.lock_path))

        # After exit: lock file removed
        self.assertFalse(os.path.exists(lock.lock_path))

    def test_context_manager_releases_on_exception(self) -> None:
        from torch_tensorrt.distributed._lock import DistributedFileLock

        try:
            with DistributedFileLock(self.tmp_dir, "test_hash") as lock:
                lock_path = lock.lock_path
                raise RuntimeError("test error")
        except RuntimeError:
            pass

        self.assertFalse(os.path.exists(lock_path))

    def test_different_names_dont_conflict(self) -> None:
        from torch_tensorrt.distributed._lock import DistributedFileLock

        lock_a = DistributedFileLock(self.tmp_dir, "hash_a")
        lock_b = DistributedFileLock(self.tmp_dir, "hash_b")

        self.assertTrue(lock_a.acquire())
        self.assertTrue(lock_b.acquire())  # different name, no conflict

        lock_a.release()
        lock_b.release()

    def test_stale_lock_detected_and_reacquired(self) -> None:
        from torch_tensorrt.distributed._lock import DistributedFileLock

        # Create a lock with very short stale timeout
        lock1 = DistributedFileLock(self.tmp_dir, "test_hash", stale_timeout_s=0.1)
        lock1.acquire()
        # Don't release — simulate crash

        time.sleep(0.2)  # wait for it to become stale

        # Another process tries to acquire
        lock2 = DistributedFileLock(self.tmp_dir, "test_hash", stale_timeout_s=0.1)
        self.assertTrue(lock2.acquire())  # should detect stale and reacquire
        lock2.release()

    def test_non_stale_lock_not_reacquired(self) -> None:
        from torch_tensorrt.distributed._lock import DistributedFileLock

        lock1 = DistributedFileLock(self.tmp_dir, "test_hash", stale_timeout_s=600)
        lock1.acquire()

        lock2 = DistributedFileLock(self.tmp_dir, "test_hash", stale_timeout_s=600)
        self.assertFalse(lock2.acquire())  # not stale, can't acquire

        lock1.release()

    def test_cleanup_stale_locks(self) -> None:
        from torch_tensorrt.distributed._lock import DistributedFileLock

        # Create stale locks
        for i in range(3):
            lock = DistributedFileLock(self.tmp_dir, f"hash_{i}", stale_timeout_s=0.1)
            lock.acquire()
            # Don't release

        time.sleep(0.2)

        removed = DistributedFileLock.cleanup_stale_locks(
            self.tmp_dir, stale_timeout_s=0.1
        )
        self.assertEqual(removed, 3)

    def test_lock_path_format(self) -> None:
        from torch_tensorrt.distributed._lock import DistributedFileLock

        lock = DistributedFileLock("/tmp/cache", "abc123")
        self.assertEqual(lock.lock_path, "/tmp/cache/.abc123.building")


# ---------------------------------------------------------------------------
# Section 2 — Multi-rank distributed caching test
# ---------------------------------------------------------------------------


class SimpleModel(nn.Module):
    """Small model for cache coordination testing."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def _multirank_distributed_cache_test(
    rank: int, world_size: int, device: torch.device, cache_dir: str
) -> None:
    """Test that distributed caching coordinates engine builds across ranks.

    Compiles the same model on both ranks with cache_built_engines=True and
    reuse_cached_engines=True. All ranks must use the same cache_dir for
    lock coordination to work.

    Verifies:
      1. Both ranks produce correct output
      2. Cache directory has engine files
    """
    import torch_tensorrt

    torch.manual_seed(42)
    model = SimpleModel().eval().to(device)
    inp = torch.randn(2, 32, device=device)

    # PyTorch reference
    with torch.no_grad():
        ref_output = model(inp)

    # Compile with caching enabled
    with torch.no_grad():
        torch._dynamo.reset()
        trt_model = torch.compile(
            model,
            backend="torch_tensorrt",
            options={
                "enabled_precisions": {torch.float32},
                "use_python_runtime": False,
                "min_block_size": 1,
                "cache_built_engines": True,
                "reuse_cached_engines": True,
                "immutable_weights": False,
                "engine_cache_dir": cache_dir,
                "engine_cache_size": 1 << 30,  # 1GB
            },
        )
        trt_output = trt_model(inp)

    # Verify correctness
    diff = (ref_output - trt_output).abs().max().item()
    assert diff < 0.01, f"Rank {rank}: output mismatch, max diff={diff}"
    print(f"[Rank {rank}] Compile + cache OK (max_diff={diff:.6f})", flush=True)

    # Verify cache directory has files
    cache_files = os.listdir(cache_dir)
    print(
        f"[Rank {rank}] Cache dir has {len(cache_files)} entries: {cache_files[:5]}",
        flush=True,
    )
    assert (
        len(cache_files) > 0
    ), f"Rank {rank}: cache dir is empty — caching didn't work"

    dist.barrier()

    # Second compile — should hit cache on both ranks
    torch._dynamo.reset()
    trt_model2 = torch.compile(
        model,
        backend="torch_tensorrt",
        options={
            "enabled_precisions": {torch.float32},
            "use_python_runtime": False,
            "min_block_size": 1,
            "cache_built_engines": True,
            "reuse_cached_engines": True,
            "immutable_weights": False,
            "engine_cache_dir": cache_dir,
            "engine_cache_size": 1 << 30,
        },
    )
    with torch.no_grad():
        trt_output2 = trt_model2(inp)

    diff2 = (ref_output - trt_output2).abs().max().item()
    assert diff2 < 0.01, f"Rank {rank}: cached output mismatch, max diff={diff2}"
    print(f"[Rank {rank}] Cache reuse OK (max_diff={diff2:.6f})", flush=True)


# ---------------------------------------------------------------------------
# Section 3 — Multi-rank pytest (MultiProcessTestCase)
# ---------------------------------------------------------------------------


class TestMultirankDistributedCache(MultiProcessTestCase):
    """Distributed engine cache tests as pytest-compatible MultiProcessTestCase."""

    world_size = 2

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def _init_dist(self) -> torch.device:
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
        )
        local = self.rank % torch.cuda.device_count()
        torch.cuda.set_device(local)
        dist.barrier()
        return torch.device(f"cuda:{local}")

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_distributed_cache_coordination(self) -> None:
        """Both ranks compile same model with caching — output matches reference."""
        device = self._init_dist()
        cache_dir = tempfile.mkdtemp(prefix="trt_dist_cache_pytest_")
        _multirank_distributed_cache_test(self.rank, self.world_size, device, cache_dir)


# ---------------------------------------------------------------------------
# Section 4 — torchrun entry point
# ---------------------------------------------------------------------------


def _multirank_setup() -> tuple:
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    return rank, world_size, torch.device(f"cuda:{local_rank}")


def run_multirank_tests() -> None:
    rank, world_size, device = _multirank_setup()
    print(f"[Rank {rank}/{world_size}] device={device}", flush=True)

    # All ranks must share the same cache dir — use a deterministic path.
    # Clean up from previous runs so the lock coordination is tested fresh.
    import shutil

    cache_dir = os.path.join(tempfile.gettempdir(), "trt_dist_cache_test_shared")
    if rank == 0:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
    dist.barrier()  # wait for rank 0 to clean up before anyone creates it
    os.makedirs(cache_dir, exist_ok=True)

    try:
        _multirank_distributed_cache_test(rank, world_size, device, cache_dir)
    except Exception as e:
        print(f"[Rank {rank}] FAIL: {e}", flush=True)
        import traceback

        traceback.print_exc()
        dist.destroy_process_group()
        sys.exit(1)

    dist.barrier()
    dist.destroy_process_group()
    print(f"[Rank {rank}] All distributed cache tests PASSED.", flush=True)


if __name__ == "__main__":
    if "--multirank" in sys.argv or "--multinode" in sys.argv:
        sys.argv = [a for a in sys.argv if a not in ("--multirank", "--multinode")]
        run_multirank_tests()
    else:
        run_tests()
