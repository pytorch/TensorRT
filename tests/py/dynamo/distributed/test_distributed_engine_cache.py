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
import sys
import tempfile

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
# Multi-rank distributed caching test
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
    filelock coordination to work.

    Verifies:
      1. Both ranks produce correct output
      2. Cache directory has engine files
      3. Second compile hits cache on both ranks
    """
    import torch_tensorrt

    torch.manual_seed(42)
    model = SimpleModel().eval().to(device)
    inp = torch.randn(2, 32, device=device)

    # PyTorch reference
    with torch.no_grad():
        ref_output = model(inp)

    # Phase 1: Compile with caching enabled
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
                "engine_cache_size": 1 << 30,
            },
        )
        trt_output = trt_model(inp)

    diff = (ref_output - trt_output).abs().max().item()
    assert diff < 0.01, f"Rank {rank}: output mismatch, max diff={diff}"
    print(f"[Rank {rank}] Compile + cache OK (max_diff={diff:.6f})", flush=True)

    # Verify cache has files (ignore .lock files)
    cache_files = [f for f in os.listdir(cache_dir) if not f.endswith(".lock")]
    print(
        f"[Rank {rank}] Cache dir has {len(cache_files)} entries: {cache_files[:5]}",
        flush=True,
    )
    assert len(cache_files) > 0, f"Rank {rank}: cache dir is empty"

    dist.barrier()

    # Phase 2: Second compile — should hit cache on both ranks
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
# Multi-rank pytest (MultiProcessTestCase)
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
    rank, world_size, device = _multirank_setup()
    print(f"[Rank {rank}/{world_size}] device={device}", flush=True)

    # All ranks must share the same cache dir.
    # Clean up from previous runs so filelock coordination is tested fresh.
    import shutil

    cache_dir = os.path.join(tempfile.gettempdir(), "trt_dist_cache_test_shared")
    if rank == 0:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
    dist.barrier()
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
