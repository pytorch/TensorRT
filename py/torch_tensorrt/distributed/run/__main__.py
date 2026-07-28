#!/usr/bin/env python3
"""
torchtrtrun — torchrun-compatible launcher with automatic NCCL setup for
TensorRT distributed inference.

Spawns worker processes with LD_PRELOAD and LD_LIBRARY_PATH configured so
TRT's internal dlopen("libnccl.so") finds the correct library before
setCommunicator() is called.

Single node, 2 GPUs:
  python -m torchtrtrun --nproc_per_node=2 script.py [args...]

Multinode, 1 GPU per node — run on each node:
  # Node 0:
  python -m torchtrtrun --nproc_per_node=1 --nnodes=2 --node_rank=0 \\
      --master_addr=169.254.204.57 --master_port=29500 \\
      script.py [args...]

  # Node 1:
  python -m torchtrtrun --nproc_per_node=1 --nnodes=2 --node_rank=1 \\
      --master_addr=169.254.204.57 --master_port=29500 \\
      script.py [args...]
"""

from torch_tensorrt.distributed.run import main

if __name__ == "__main__":
    main()
