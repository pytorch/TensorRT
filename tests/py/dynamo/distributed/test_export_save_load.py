"""
Tensor Parallel export → save → load → inference tests.

Tests the full serialization round-trip for tensor-parallel models with
native TRT NCCL collectives (C++ runtime):
  1. torch.export → TRT AOT compile → save per-rank engine
  2. Load per-rank engine → inference (no model weights needed)
  3. Loaded engine output matches freshly compiled engine output

Run single-rank pytest tests (no multi-GPU needed):
    cd tests/py/dynamo
    pytest distributed/test_export_save_load.py -v

Run multi-rank pytest tests (requires 2 GPUs, spawned automatically):
    cd tests/py/dynamo
    pytest distributed/test_export_save_load.py::TestMultirankExportSaveLoad -v

Run multi-rank tests via torchrun (legacy):
    torchrun --nproc_per_node=2 distributed/test_export_save_load.py --multirank

Run multi-rank tests (multinode, 1 GPU per node — run on each node):
    # Node 0:
    RANK=0 WORLD_SIZE=2 LOCAL_RANK=0 MASTER_ADDR=<node0-ip> MASTER_PORT=29500 \\
        python distributed/test_export_save_load.py --multinode
    # Node 1:
    RANK=1 WORLD_SIZE=2 LOCAL_RANK=0 MASTER_ADDR=<node0-ip> MASTER_PORT=29500 \\
        python distributed/test_export_save_load.py --multinode
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils._pytree
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


def is_trtllm_for_nccl() -> bool:
    try:
        from torch_tensorrt._features import ENABLED_FEATURES

        return bool(ENABLED_FEATURES.trtllm_for_nccl)
    except Exception:
        return False


def has_nccl_collectives() -> bool:
    """Check if any NCCL collective backend is available (native TRT or TRT-LLM)."""
    try:
        from torch_tensorrt._features import ENABLED_FEATURES

        return bool(ENABLED_FEATURES.native_trt_collectives) or bool(
            ENABLED_FEATURES.trtllm_for_nccl
        )
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Model (small dims for fast TRT compilation in CI)
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    """Single tensor-parallel stage: colwise fc1 + rowwise fc2 + all-reduce."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 16)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class _RowParallelLinear(nn.Module):
    """Linear + NCCL all-reduce (replaces DTensor RowwiseParallel for export)."""

    def __init__(self, linear, group_name):
        super().__init__()
        self.linear = linear
        self.group_name = group_name

    def forward(self, x):
        out = self.linear(x)
        out = torch.ops._c10d_functional.all_reduce(out, "sum", self.group_name)
        out = torch.ops._c10d_functional.wait_tensor(out)
        return out


def build_exportable_model(rank: int, world_size: int) -> nn.Module:
    """Return per-rank model with manually sliced weights + explicit all-reduce.

    fc1 is ColwiseParallel: each rank holds rows [rank*32 : (rank+1)*32] of weight.
    fc2 is RowwiseParallel: each rank holds cols [rank*32 : (rank+1)*32] of weight,
    followed by an all-reduce to produce the replicated output.
    """
    group_name = dist.distributed_c10d._get_default_group().group_name
    model = TinyModel().to("cuda")

    # ColwiseParallel on fc1: split output features
    w = model.fc1.weight.data  # (64, 16)
    chunk = w.shape[0] // world_size
    model.fc1.weight = nn.Parameter(w[rank * chunk : (rank + 1) * chunk].contiguous())
    if model.fc1.bias is not None:
        b = model.fc1.bias.data  # (64,)
        model.fc1.bias = nn.Parameter(b[rank * chunk : (rank + 1) * chunk].contiguous())

    # RowwiseParallel on fc2: split input features + all-reduce
    w2 = model.fc2.weight.data  # (16, 64)
    chunk2 = w2.shape[1] // world_size
    model.fc2.weight = nn.Parameter(
        w2[:, rank * chunk2 : (rank + 1) * chunk2].contiguous()
    )
    model.fc2 = _RowParallelLinear(model.fc2, group_name)

    return model


def rank_path(save_dir: str, rank: int, world_size: int) -> str:
    return str(Path(save_dir) / f"tp_rank{rank}_of_{world_size}.pt2")


# ---------------------------------------------------------------------------
# Single-rank pytest tests (run under plain pytest, no multi-GPU needed)
# ---------------------------------------------------------------------------


class TestTinyModelBasics(unittest.TestCase):
    """Structural / shape tests — no distributed setup required."""

    def test_rank_path_format(self) -> None:
        """rank_path() produces the expected filename."""
        path = rank_path("/tmp/test", rank=1, world_size=4)
        self.assertTrue(path.endswith("tp_rank1_of_4.pt2"))

    def test_tiny_model_output_shape(self) -> None:
        """TinyModel produces a (batch, 16) tensor."""
        model = TinyModel().eval()
        inp = torch.randn(4, 16)
        with torch.no_grad():
            out = model(inp)
        self.assertEqual(out.shape, torch.Size([4, 16]))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_tiny_model_cuda_forward(self) -> None:
        """TinyModel forward pass runs on CUDA."""
        model = TinyModel().cuda().eval()
        inp = torch.randn(4, 16, device="cuda")
        with torch.no_grad():
            out = model(inp)
        self.assertEqual(out.shape, torch.Size([4, 16]))
        self.assertEqual(out.device.type, "cuda")


# ---------------------------------------------------------------------------
# Multi-rank helpers (run via torchrun --multirank)
# ---------------------------------------------------------------------------


def _multirank_setup() -> tuple:
    """Initialize the distributed environment for multi-rank tests."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device


def _multirank_export_compile_save(
    rank: int, world_size: int, device: torch.device, save_dir: str
):
    """Export + TRT AOT compile + save per-rank engine.

    Returns the compiled TRT model so the caller can use it for comparison.
    """
    import torch_tensorrt
    from torch_tensorrt.distributed._nccl_utils import setup_nccl_for_torch_tensorrt

    setup_nccl_for_torch_tensorrt()

    torch.utils._pytree.register_constant(
        torch.distributed.tensor._dtensor_spec.DTensorSpec
    )

    torch.manual_seed(0)
    model = build_exportable_model(rank, world_size)
    inp = torch.randn(4, 16, device=device)

    with torch.no_grad():
        ref_output = model(inp)

    ep = torch.export.export(model, args=(inp,), strict=False)

    trt_model = torch_tensorrt.dynamo.compile(
        ep,
        inputs=[inp],
        use_explicit_typing=True,
        use_fp32_acc=True,
        device=device,
        disable_tf32=True,
        use_python_runtime=False,
        min_block_size=1,
        use_distributed_mode_trace=True,
        assume_dynamic_shape_support=True,
    )

    with torch.no_grad():
        output = trt_model(inp)

    std = float((ref_output - output).std())
    assert std < 0.05, f"Export compile output mismatch: std={std}"
    print(f"[Rank {rank}] Export+compile OK (std={std:.6f})", flush=True)

    path = rank_path(save_dir, rank, world_size)
    torch_tensorrt.save(trt_model, path, inputs=[inp], retrace=False)
    dist.barrier()

    assert os.path.isfile(path), f"Engine file not found: {path}"
    print(
        f"[Rank {rank}] Saved engine → {path} ({os.path.getsize(path) / 1e6:.1f} MB)",
        flush=True,
    )
    return trt_model, inp


def _multirank_load_and_infer(
    rank: int, world_size: int, device: torch.device, save_dir: str, inp: torch.Tensor
) -> None:
    """Load per-rank engine and verify inference matches reference."""
    import torch_tensorrt
    from torch_tensorrt.distributed._distributed import distributed_context
    from torch_tensorrt.distributed._nccl_utils import (
        initialize_nccl_comm,
        setup_nccl_for_torch_tensorrt,
    )

    setup_nccl_for_torch_tensorrt()
    initialize_nccl_comm()

    path = rank_path(save_dir, rank, world_size)
    loaded = torch_tensorrt.load(path)
    loaded_model = loaded.module()

    with distributed_context(dist.group.WORLD, loaded_model) as m:
        with torch.no_grad():
            loaded_output = m(inp)

    torch.manual_seed(0)
    ref_model = build_exportable_model(rank, world_size)
    with torch.no_grad():
        ref_output = ref_model(inp)

    std = float((ref_output - loaded_output).std())
    assert std < 0.05, f"Loaded engine output mismatch: std={std}"
    print(f"[Rank {rank}] Load+infer OK (std={std:.6f})", flush=True)


def _multirank_loaded_matches_compiled(
    rank: int,
    world_size: int,
    device: torch.device,
    save_dir: str,
    compiled_model,
    inp: torch.Tensor,
) -> None:
    """Verify loaded engine output is numerically identical to compiled engine output."""
    import torch_tensorrt
    from torch_tensorrt.distributed._distributed import distributed_context
    from torch_tensorrt.distributed._nccl_utils import initialize_nccl_comm

    initialize_nccl_comm()

    path = rank_path(save_dir, rank, world_size)
    loaded = torch_tensorrt.load(path)
    loaded_model = loaded.module()

    with distributed_context(dist.group.WORLD, [loaded_model, compiled_model]) as (
        lm,
        cm,
    ):
        with torch.no_grad():
            loaded_output = lm(inp)
            compiled_output = cm(inp)

    diff = float((compiled_output - loaded_output).abs().max())
    assert diff < 1e-3, f"Compiled vs loaded mismatch: max_diff={diff}"
    print(f"[Rank {rank}] Compiled==loaded OK (max_diff={diff:.6f})", flush=True)


# ---------------------------------------------------------------------------
# Multi-rank pytest tests (MultiProcessTestCase, requires 2 GPUs)
# ---------------------------------------------------------------------------


class TestMultirankExportSaveLoad(MultiProcessTestCase):
    """Export → save → load → inference round-trip as a pytest-compatible test.

    Spawns 2 worker processes automatically.  Requires 2 CUDA GPUs.  Run with:

        pytest distributed/test_export_save_load.py::TestMultirankExportSaveLoad -v
    """

    world_size = 2

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def _init_dist(self) -> torch.device:
        """Init NCCL process group via FileStore (no env-var dependency).

        The dist.barrier() seeds the NCCL communicator so bind_nccl_comm()
        sees a non-null getCommPtr() on the first TRT forward pass.
        """
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
        )
        # Overwrite any stale RANK/WORLD_SIZE left by single-rank test setup
        # so TRT converter env-var reads agree with torch.distributed.
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        local = self.rank % torch.cuda.device_count()
        torch.cuda.set_device(local)
        dist.barrier()  # seeds ncclComm_t before any TRT bind_nccl_comm() call
        return torch.device(f"cuda:{local}")

    @unittest.skipIf(not has_nccl_collectives(), "No NCCL collective support available")
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_export_save_load_round_trip(self) -> None:
        """Full round-trip: export → AOT compile → save → load → verify output."""
        device = self._init_dist()
        save_dir = tempfile.mkdtemp(prefix="trt_export_pytest_")

        compiled_model, inp = _multirank_export_compile_save(
            self.rank, self.world_size, device, save_dir
        )
        dist.barrier()
        _multirank_load_and_infer(self.rank, self.world_size, device, save_dir, inp)
        dist.barrier()
        _multirank_loaded_matches_compiled(
            self.rank, self.world_size, device, save_dir, compiled_model, inp
        )


# ---------------------------------------------------------------------------
# torchrun / mpirun entry point (legacy)
# ---------------------------------------------------------------------------


def run_multirank_tests() -> None:
    """Entry point for --multirank / --multinode mode."""
    rank, world_size, device = _multirank_setup()
    print(f"[Rank {rank}/{world_size}] device={device}", flush=True)

    if world_size < 2:
        print(
            f"[Rank {rank}] world_size={world_size}: distributed export/save/load tests "
            "require world_size >= 2. Run with torchrun --nproc_per_node=2 or two nodes.",
            flush=True,
        )
        dist.destroy_process_group()
        sys.exit(0)

    # Each rank creates its own local tmpdir (works on shared FS and multinode).
    save_dir = tempfile.mkdtemp(prefix="trt_export_test_")

    compiled_model = None
    inp = None
    failed = []

    tests = [
        (
            "export_compile_save",
            lambda: _multirank_export_compile_save(rank, world_size, device, save_dir),
        ),
    ]

    for name, fn in tests:
        dist.barrier()
        try:
            result = fn()
            if name == "export_compile_save":
                compiled_model, inp = result
        except Exception as e:
            import traceback

            failed.append((name, str(e)))
            print(f"[Rank {rank}] FAIL {name}: {e}", flush=True)
            traceback.print_exc()

    if inp is not None:
        for name, fn in [
            (
                "load_and_infer",
                lambda: _multirank_load_and_infer(
                    rank, world_size, device, save_dir, inp
                ),
            ),
        ]:
            dist.barrier()
            try:
                fn()
            except Exception as e:
                import traceback

                failed.append((name, str(e)))
                print(f"[Rank {rank}] FAIL {name}: {e}", flush=True)
                traceback.print_exc()

    if compiled_model is not None and inp is not None:
        dist.barrier()
        try:
            _multirank_loaded_matches_compiled(
                rank, world_size, device, save_dir, compiled_model, inp
            )
        except Exception as e:
            import traceback

            failed.append(("loaded_matches_compiled", str(e)))
            print(f"[Rank {rank}] FAIL loaded_matches_compiled: {e}", flush=True)
            traceback.print_exc()

    del compiled_model
    torch.cuda.empty_cache()

    all_test_names = [
        "export_compile_save",
        "load_and_infer",
        "loaded_matches_compiled",
    ]
    passed = [n for n in all_test_names if n not in dict(failed)]
    print(
        f"[Rank {rank}] Results — passed: {passed}  failed: {[n for n, _ in failed]}",
        flush=True,
    )

    dist.barrier()
    dist.destroy_process_group()

    if failed:
        if rank == 0:
            print(f"\n{len(failed)} test(s) FAILED:")
            for name, err in failed:
                print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        if rank == 0:
            print(f"\nAll multi-rank export/save/load tests PASSED.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--multirank" in sys.argv or "--multinode" in sys.argv:
        sys.argv = [a for a in sys.argv if a not in ("--multirank", "--multinode")]
        run_multirank_tests()
    else:
        run_tests()
