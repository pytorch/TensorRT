"""
Thread-local process group management for Torch-TensorRT distributed engines.

The active process group controls which NCCL communicator TRT engines use.
For the common case (default world group) no setup is needed — engines pick it
up automatically after dist.init_process_group().

For advanced parallelism strategies (e.g. TP inside a DP job) where TRT engines
should use a subgroup communicator, wrap compilation and execution with the
distributed_group() context manager.
"""

import threading
from contextlib import contextmanager
from typing import Any, Generator, Optional

import torch.distributed as dist

_state = threading.local()


def get_active_group() -> Optional[Any]:
    """Return the active ProcessGroup for TRT NCCL engines.

    Respects the distributed_group() context manager; falls back to the
    default world group when no context is active.
    """
    group = getattr(_state, "pg", None)
    if group is not None:
        return group
    if dist.is_available() and dist.is_initialized():
        return dist.group.WORLD
    return None


def get_active_group_name() -> str:
    """Return the c10d registry name for the active ProcessGroup.

    Used by C++ TRTEngine to look up the group via c10d::resolve_process_group.
    Returns "" when no distributed context is active.
    """
    group = get_active_group()
    if group is not None and hasattr(group, "group_name"):
        return str(group.group_name)
    return ""


@contextmanager
def distributed_group(group: Any) -> Generator[None, None, None]:
    """Context manager: compile and run TRT engines using *group* for NCCL.

    Only needed when the TRT engine's NCCL collective should use a non-default
    process group (e.g. tensor-parallel subgroup inside a data-parallel job).
    For the default world group, no context manager is required.

    Usage::

        tp_group = dist.new_group(ranks=[0, 1])
        with torch_tensorrt.distributed_group(tp_group):
            trt_model = torch.compile(model, backend="torch_tensorrt", ...)
            output = trt_model(inp)
    """
    old = getattr(_state, "pg", None)
    _state.pg = group
    try:
        yield
    finally:
        _state.pg = old
