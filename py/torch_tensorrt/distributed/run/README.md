# torchtrtrun

A `torchrun`-compatible launcher that automatically configures NCCL for
TensorRT distributed inference before spawning worker processes.

## Problem

TRT 10.x's internal `libLoader.cpp` calls `dlopen("libnccl.so")` at
`setCommunicator()` time. This happens inside the C++ runtime before any
Python code runs, so setting `LD_LIBRARY_PATH` or `LD_PRELOAD` inside the
script is too late. `torchtrtrun` sets both **before** spawning worker
processes so TRT finds the correct NCCL library instance. Fixed in TRT 11.0.

## Usage

```bash
# Single node, 2 GPUs
torchtrtrun --nproc_per_node=2 script.py [args...]

# Multinode, 1 GPU per node — run on each node independently
# Node 0:
torchtrtrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
    --rdzv_endpoint=<node0-ip>:29500 \
    script.py [args...]

# Node 1:
torchtrtrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \
    --rdzv_endpoint=<node0-ip>:29500 \
    script.py [args...]
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--nproc_per_node` | `1` | GPUs / worker processes per node |
| `--nnodes` | `1` | Total number of nodes |
| `--node_rank` | `0` | Rank of this node |
| `--rdzv_endpoint` | `` | Rendezvous endpoint as `host:port` |
| `--rdzv_backend` | `static` | Rendezvous backend |
| `--rdzv_id` | `none` | Rendezvous job id |
| `--master_addr` | `127.0.0.1` | Legacy: IP of master node (use `--rdzv_endpoint`) |
| `--master_port` | `29500` | Legacy: port on master node (use `--rdzv_endpoint`) |

## What it does

1. Finds PyTorch's `nvidia.nccl` pip package (skips setup for system/NGC NCCL)
2. Creates the `libnccl.so → libnccl.so.2` symlink if missing (TRT looks for `libnccl.so` by name)
3. Prepends the NCCL lib directory to `LD_LIBRARY_PATH`
4. Sets `LD_PRELOAD` to `libnccl.so.2` so TRT's `dlopen` finds the library already resident in the process
5. Spawns worker processes with `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` set

## Examples

```bash
# Llama TP (single node)
torchtrtrun --nproc_per_node=2 tools/llm/tensor_parallel_llama_llm.py

# Qwen TP (multinode)
torchtrtrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \
    --rdzv_endpoint=169.254.204.57:29500 \
    tools/llm/tensor_parallel_qwen_multinode.py
```
