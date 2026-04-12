"""
Thread-local process group management for Torch-TensorRT distributed engines.

The active process group controls which NCCL communicator TRT engines use.
For the common case (default world group) no setup is needed — engines pick it
up automatically after dist.init_process_group().

For advanced parallelism strategies (e.g. TP inside a DP job) where TRT engines
should use a subgroup communicator, wrap compilation and execution with the
distributed_group() context manager, or call set_distributed_group() to pin
the group on all engines in a loaded module before the first forward pass.
"""

import threading
from contextlib import contextmanager
from typing import Any, Generator, Optional, TypeVar

import torch.distributed as dist
import torch.nn as nn

M = TypeVar("M", bound=nn.Module)

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
def distributed_group(
    group: Any, module: Optional[M] = None
) -> Generator[Optional[M], None, None]:
    """Context manager: run TRT engines using *group* for NCCL.

    Sets the active process group for the duration of the ``with`` block.
    Only needed when the TRT engine's NCCL collective should use a non-default
    process group (e.g. tensor-parallel subgroup inside a data-parallel job).
    For the default world group, no context manager is required.

    When *module* is supplied the group is also pre-pinned on all TRT engines
    in the module via :func:`set_distributed_group`, and the configured module
    is yielded so it can be used directly::

        tp_group = dist.new_group(ranks=[0, 1])

        # Without module — use for compile-time wrapping:
        with torch_tensorrt.distributed_group(tp_group):
            trt_model = torch.compile(model, backend="torch_tensorrt", ...)
            output = trt_model(inp)

        # With module — pre-pin on a loaded model and get it back as the handle:
        model = torch_tensorrt.load("tp_model.ep")
        with torch_tensorrt.distributed_group(tp_group, model) as trt_model:
            output = trt_model(inp)

    Args:
        group:  The ``ProcessGroup`` to use for all TRT NCCL collectives.
        module: Optional compiled/loaded module.  When provided,
                :func:`set_distributed_group` is called on it and the module
                is yielded as the context value.

    Yields:
        *module* (with the group pre-pinned) when supplied, otherwise ``None``.
    """
    old = getattr(_state, "pg", None)
    _state.pg = group
    try:
        if module is not None:
            set_distributed_group(module, group)
            yield module
        else:
            yield None
    finally:
        _state.pg = old


def set_distributed_group(module: nn.Module, group: Any) -> None:
    """Pin *group* as the NCCL process group on every TRT engine in *module*.

    Walks *module* recursively and calls ``set_group_name`` on every
    multi-device TRT engine found, covering both cases:

    * **Submodule engines** — ``TorchTensorRTModule`` children produced by
      ``torch.compile(..., backend="torch_tensorrt")`` or
      ``torch_tensorrt.compile()``.
    * **Inlined engines** — ``torch.classes.tensorrt.Engine`` objects stored
      as plain attributes on an ``fx.GraphModule`` after
      ``torch_tensorrt.save()`` / ``torch_tensorrt.load()``.

    Prefer the ``distributed_group(group, module)`` context manager over
    calling this directly — it combines group activation and engine pinning
    in one step.  Call this directly only when you need the pinning to persist
    outside any ``with`` block::

        model = torch_tensorrt.load("tp_model.ep")
        tp_group = dist.new_group(ranks=[0, 1])
        torch_tensorrt.distributed.set_distributed_group(model, tp_group)
        output = model(inp)  # group already pinned, no context needed

    Args:
        module: Compiled or loaded module whose TRT engines should use *group*.
        group:  The ``ProcessGroup`` to bind for NCCL collectives.
    """
    from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

    # Resolve the group name directly from the supplied group object so this
    # function is safe to call whether or not _state.pg is already set.
    group_name = str(group.group_name) if hasattr(group, "group_name") else ""

    if not group_name:
        return

    seen: set[int] = set()

    for submod in module.modules():
        # --- Case 1: TorchTensorRTModule wrapper ---
        if isinstance(submod, TorchTensorRTModule):
            engine = getattr(submod, "engine", None)
            if engine is not None and id(engine) not in seen:
                seen.add(id(engine))
                if engine.is_md:
                    engine.set_group_name(group_name)
            # Don't fall through to the attribute scan for this submodule.
            continue

        # --- Case 2: Inlined engines on fx.GraphModule (or any module) ---
        # After inline_trt_modules(), engines are stored as plain attributes
        # (e.g. "_run_on_acc_0_engine") rather than as TorchTensorRTModule
        # submodules.  Duck-type check for torch.classes.tensorrt.Engine.
        for attr_val in vars(submod).values():
            if (
                id(attr_val) not in seen
                and hasattr(attr_val, "is_md")
                and hasattr(attr_val, "set_group_name")
                and attr_val.is_md
            ):
                seen.add(id(attr_val))
                attr_val.set_group_name(group_name)
