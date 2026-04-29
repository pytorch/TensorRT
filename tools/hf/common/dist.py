"""
Distributed helpers for multi-device strategies.

The patterns here mirror tools/llm/tensor_parallel_llama_multinode.py:
torchtrtrun (which is torchrun + NCCL/LD_PRELOAD setup) populates the
standard distributed env vars before launching this script:

  WORLD_SIZE   – total number of ranks
  RANK         – this rank's global id (0..WORLD_SIZE-1)
  LOCAL_RANK   – this rank's local id within its node (used for cuda:N)
  MASTER_ADDR  – rendezvous host
  MASTER_PORT  – rendezvous port

Single-process (no torchtrtrun) defaults: WORLD_SIZE=1 (or unset), so
all helpers here become no-ops.
"""
from __future__ import annotations

import datetime
import logging
import os
from typing import Optional

import torch

logger = logging.getLogger("torchtrt.hf.dist")


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed() -> bool:
    return world_size() > 1


def is_master() -> bool:
    return rank() == 0


def init_distributed(timeout_hours: int = 2) -> Optional["torch.distributed.ProcessGroup"]:
    """
    Initialize torch.distributed if WORLD_SIZE > 1.

    Long timeout (default 2 h) so TRT engine builds don't trigger the NCCL
    watchdog when one rank takes longer than another to build.

    Also pins this process's CUDA device to LOCAL_RANK and registers
    DTensorSpec as a pytree constant (required for torch.compile + TP to
    avoid mode-mismatch errors).
    """
    if not is_distributed():
        return None

    import torch.distributed as dist
    import torch.distributed.tensor._dtensor_spec
    import torch.utils._pytree

    # DTensorSpec must be a pytree constant before any TP model is traced.
    torch.utils._pytree.register_constant(
        torch.distributed.tensor._dtensor_spec.DTensorSpec
    )

    torch.cuda.set_device(local_rank())

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(hours=timeout_hours),
        )

    # Wire NCCL into TRT (sets up the symbol resolution path so TRT's
    # collectives talk to the same libnccl.so torchtrtrun preloaded).
    try:
        from torch_tensorrt.distributed import setup_nccl_for_torch_tensorrt
        setup_nccl_for_torch_tensorrt()
    except Exception as e:
        logger.warning(f"setup_nccl_for_torch_tensorrt failed: {e}")

    logger.info(
        f"[rank {rank()}/{world_size()}] dist init OK on cuda:{local_rank()}"
    )
    return dist.group.WORLD


def build_device_mesh(mesh_dim_names: tuple[str, ...] = ("tp",)):
    """
    Build a 1-D device mesh covering all WORLD_SIZE ranks.

    Multi-dim meshes (TP × PP × DP) can be added later by extending the
    shape and dim names.  For now we only support a flat TP axis.
    """
    if not is_distributed():
        return None
    from torch.distributed.device_mesh import init_device_mesh

    return init_device_mesh(
        "cuda",
        (world_size(),),
        mesh_dim_names=mesh_dim_names,
    )


def barrier() -> None:
    """torch.distributed barrier — no-op when not distributed."""
    if not is_distributed():
        return
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier()
