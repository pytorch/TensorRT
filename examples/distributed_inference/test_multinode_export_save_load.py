"""
Two-node test: torch.export → TRT AOT compile → save → load → inference.

Tests the full serialization round-trip for tensor-parallel models with
native TRT NCCL collectives (C++ runtime).

Reads RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT from the environment.

Usage
-----
  NCCL_LIB=$(python -c "from torch_tensorrt.distributed._nccl_utils import get_nccl_library_path; print(get_nccl_library_path())")

# Rank 0:
  LD_LIBRARY_PATH="$NCCL_LIB:$LD_LIBRARY_PATH" \\
    RANK=0 WORLD_SIZE=2 MASTER_ADDR=<rank0_ip> MASTER_PORT=29500 \\
    uv run python examples/distributed_inference/test_multinode_export_save_load.py

# Rank 1:
  LD_LIBRARY_PATH="$NCCL_LIB:$LD_LIBRARY_PATH" \\
    RANK=1 WORLD_SIZE=2 MASTER_ADDR=<rank0_ip> MASTER_PORT=29500 \\
    uv run python examples/distributed_inference/test_multinode_export_save_load.py
"""

import datetime
import faulthandler
import os
import sys
import tempfile
from pathlib import Path

faulthandler.enable()

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils._pytree
import torch_tensorrt
from torch.distributed._tensor import Shard
from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch_tensorrt.distributed import setup_nccl_for_torch_tensorrt

torch.utils._pytree.register_constant(
    torch.distributed.tensor._dtensor_spec.DTensorSpec
)

setup_nccl_for_torch_tensorrt()

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])

torch.cuda.set_device(0)  # one GPU per node
dist.init_process_group(
    "nccl",
    rank=rank,
    world_size=world_size,
    timeout=datetime.timedelta(hours=2),
)
torch.manual_seed(0)

device_mesh = init_device_mesh("cuda", (world_size,))
print(f"[Rank {rank}/{world_size}] distributed init OK", flush=True)


# ---------------------------------------------------------------------------
# Model: simple TP MLP (same as test_multinode_nccl.py)
# ---------------------------------------------------------------------------


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(10, 3200)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(3200, 1600)
        self.in_proj2 = nn.Linear(1600, 500)
        self.out_proj2 = nn.Linear(500, 100)

    def forward(self, x):
        x = self.out_proj(self.relu(self.in_proj(x)))
        x = self.relu(x)
        x = self.out_proj2(self.relu(self.in_proj2(x)))
        return x


# ---------------------------------------------------------------------------
# Exportable model: manual sharding + explicit all-reduce
# ---------------------------------------------------------------------------


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


def build_exportable_model(rank, world_size):
    """Build model with manually sliced weights + explicit all-reduce."""
    group_name = dist.distributed_c10d._get_default_group().group_name
    model = ToyModel().to("cuda")

    # Column-parallel: slice output dim (dim 0)
    for proj in [model.in_proj, model.in_proj2]:
        w = proj.weight.data
        chunk = w.shape[0] // world_size
        proj.weight = nn.Parameter(w[rank * chunk : (rank + 1) * chunk].contiguous())
        if proj.bias is not None:
            b = proj.bias.data
            proj.bias = nn.Parameter(b[rank * chunk : (rank + 1) * chunk].contiguous())

    # Row-parallel: slice input dim (dim 1) + wrap with all-reduce
    for attr in ["out_proj", "out_proj2"]:
        proj = getattr(model, attr)
        w = proj.weight.data
        chunk = w.shape[1] // world_size
        proj.weight = nn.Parameter(w[:, rank * chunk : (rank + 1) * chunk].contiguous())
        setattr(model, attr, _RowParallelLinear(proj, group_name))

    return model


def rank_path(save_dir, rank, world_size):
    return str(Path(save_dir) / f"tp_rank{rank}_of_{world_size}.pt2")


# ---------------------------------------------------------------------------
# Build DTensor baseline for comparison
# ---------------------------------------------------------------------------

tp_model = ToyModel().to("cuda")
tp_model = parallelize_module(
    tp_model,
    device_mesh,
    {
        "in_proj": ColwiseParallel(input_layouts=Shard(0)),
        "out_proj": RowwiseParallel(output_layouts=Shard(0)),
        "in_proj2": ColwiseParallel(input_layouts=Shard(0)),
        "out_proj2": RowwiseParallel(output_layouts=Shard(0)),
    },
)

inp = torch.rand(20, 10, device="cuda")
python_result = tp_model(inp)
print(f"[Rank {rank}] PyTorch TP baseline OK, shape={python_result.shape}", flush=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

PASSED = []
FAILED = []
save_dir = tempfile.mkdtemp(prefix="trt_export_test_")


def test_export_compile_save():
    """Test 1: torch.export → TRT AOT compile → save per-rank."""
    torch.manual_seed(0)
    model = build_exportable_model(rank, world_size)

    # Get the manually-sharded model's own PyTorch output as reference
    with torch.no_grad():
        ref_output = model(inp)

    # Export with static shapes (dynamic shapes + DTensor not supported)
    ep = torch.export.export(model, args=(inp,), strict=False)

    trt_model = torch_tensorrt.dynamo.compile(
        ep,
        inputs=[inp],
        use_explicit_typing=True,
        use_fp32_acc=True,
        device=torch.device("cuda:0"),
        disable_tf32=True,
        use_python_runtime=False,
        min_block_size=1,
        use_distributed_mode_trace=True,
        assume_dynamic_shape_support=True,
    )

    # Verify TRT output matches the same model's PyTorch output
    output = trt_model(inp)
    std = float((ref_output - output).std())
    assert std < 0.01, f"Export compile output mismatch: std={std}"
    print(f"[Rank {rank}] Export compile OK (std={std:.6f})", flush=True)

    # Save per-rank engine
    path = rank_path(save_dir, rank, world_size)
    torch_tensorrt.save(trt_model, path, inputs=[inp], retrace=False)
    dist.barrier()

    assert os.path.isfile(path), f"Engine file not found: {path}"
    size_mb = os.path.getsize(path) / 1e6
    print(f"[Rank {rank}] Saved engine to {path} ({size_mb:.1f} MB)", flush=True)
    return trt_model


def test_load_and_infer():
    """Test 2: Load per-rank engine → inference (no model weights needed)."""
    # Eagerly initialize PyTorch's NCCL communicator so TRT's
    # bind_nccl_comm() can extract the ncclComm_t on first engine execution.
    from torch_tensorrt.distributed._nccl_utils import initialize_nccl_comm

    initialize_nccl_comm()

    path = rank_path(save_dir, rank, world_size)
    loaded = torch_tensorrt.load(path)
    trt_model = loaded.module()

    output = trt_model(inp)

    # Compare against a manually-sharded PyTorch model with the same seed
    torch.manual_seed(0)
    ref_model = build_exportable_model(rank, world_size)
    with torch.no_grad():
        ref_output = ref_model(inp)

    std = float((ref_output - output).std())
    assert std < 0.01, f"Loaded engine output mismatch: std={std}"
    print(f"[Rank {rank}] Load + infer OK (std={std:.6f})", flush=True)


def test_loaded_matches_compiled(compiled_output):
    """Test 3: Loaded engine produces same output as freshly compiled."""
    path = rank_path(save_dir, rank, world_size)
    loaded = torch_tensorrt.load(path)
    trt_model = loaded.module()

    loaded_output = trt_model(inp)
    compiled_output_val = compiled_output(inp)

    diff = float((compiled_output_val - loaded_output).abs().max())
    assert diff < 1e-3, f"Compiled vs loaded mismatch: max_diff={diff}"
    print(f"[Rank {rank}] Compiled vs loaded match (max_diff={diff:.6f})", flush=True)


compiled_model = None

# Run tests
for name, fn in [
    ("export_compile_save", lambda: test_export_compile_save()),
    ("load_and_infer", lambda: test_load_and_infer()),
]:
    dist.barrier()
    try:
        result = fn()
        PASSED.append(name)
        if name == "export_compile_save":
            compiled_model = result
    except Exception as e:
        print(f"[Rank {rank}] FAIL  {name}: {e}", flush=True)
        import traceback

        traceback.print_exc()
        FAILED.append(name)

# Test 3 only if test 1 passed
if compiled_model is not None:
    dist.barrier()
    try:
        test_loaded_matches_compiled(compiled_model)
        PASSED.append("loaded_matches_compiled")
    except Exception as e:
        print(f"[Rank {rank}] FAIL  loaded_matches_compiled: {e}", flush=True)
        FAILED.append("loaded_matches_compiled")

# Delete TRT engines before destroying the process group — the engines hold
# a reference to the NCCL communicator and will segfault if NCCL is torn
# down first.
del compiled_model
torch.cuda.empty_cache()
dist.destroy_process_group()

print(f"[Rank {rank}] Results — passed: {PASSED}  failed: {FAILED}", flush=True)
os._exit(0 if not FAILED else 1)
