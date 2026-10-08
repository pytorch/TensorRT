# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Thread-local process group management for Torch-TensorRT distributed engines.

The active process group controls which NCCL communicator TRT engines use.
For the common case (default world group) no setup is needed — engines pick it
up automatically after dist.init_process_group().

Native engines compiled with WORLD use global rank IDs. Compiling inside
distributed_context(subgroup) translates each collective into that parent's
rank space. Run loaded engines with the same ordered parent membership; changing
the parent mapping requires recompilation. Python validates membership before
passing the actual registry name to C++; no registry alias is needed.
"""

import threading
from contextlib import contextmanager
from typing import Any, Generator, List, Optional, Sequence, TypeVar, Union

import torch.distributed as dist
import torch.nn as nn

M = TypeVar("M", bound=nn.Module)

_state = threading.local()


def _require_world_group(group: Any) -> None:
    """Validate the parent communicator for native converters' global rank IDs."""
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "Initialize torch.distributed before binding a native TRT engine"
        )
    if group is not dist.group.WORLD:
        raise RuntimeError(
            "Native TRT engines use global rank IDs and require the WORLD communicator. "
            "Use distributed_context(dist.group.WORLD, model); select subgroups on "
            "the collective operations, not as the engine's parent communicator."
        )


def _active_native_parent() -> str:
    """Snapshot the active parent; empty string denotes WORLD (also for AOT)."""
    group = get_active_group()
    if group is None:
        return ""
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "Initialize torch.distributed before compiling with a subgroup"
        )
    if group is dist.group.WORLD:
        return ""
    if group == dist.GroupMember.NON_GROUP_MEMBER:
        raise RuntimeError(
            "This rank is not a member of the requested parent communicator"
        )
    ranks = dist.get_process_group_ranks(group)
    if dist.get_rank() not in ranks:
        raise RuntimeError(
            "This rank is not a member of the requested parent communicator"
        )
    return ",".join(str(rank) for rank in ranks)


def _require_native_parent(group: Any, parent: str = "") -> None:
    """Check the saved parent mapping before warm-up, binding, or re-pinning."""
    if not parent:
        _require_world_group(group)
        return
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "Initialize torch.distributed before binding a native TRT engine"
        )
    if group is None or group == dist.GroupMember.NON_GROUP_MEMBER:
        raise RuntimeError(
            "This rank is not a member of the requested parent communicator"
        )
    ranks = dist.get_process_group_ranks(group)
    actual = ",".join(str(rank) for rank in ranks)
    if actual != parent or dist.get_rank() not in ranks:
        raise RuntimeError(
            f"Native TRT engine was compiled for parent [{parent}], but selected "
            f"parent has ranks {ranks}. Recompile to change the parent mapping."
        )


def register_md_engine(engine: object) -> None:
    """Register a C++ TRTEngine with requires_native_multidevice=True for teardown tracking.

    Engines are stored on the thread-local ``_state`` so they are scoped
    to the active ``distributed_context`` context.  If called before any
    context is entered, the engine is stored in a pending list that the
    next ``distributed_context.__enter__`` will adopt.
    """
    engines = getattr(_state, "md_engines", None)
    if engines is None:
        _state.md_engines = [engine]
    else:
        engines.append(engine)


def get_active_group() -> Optional[Any]:
    """Return the active ProcessGroup for TRT NCCL engines.

    Respects the distributed_context() context manager; falls back to the
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
def distributed_context(
    group: Any,
    module: Union[M, Sequence[M], None] = None,
) -> Generator[Union[M, List[M], None], None, None]:
    """Context manager: run TRT engines using *group* for NCCL.

    Sets the active process group for the duration of the ``with`` block.
    Native TRT engines use rank IDs relative to the parent selected during
    compilation. Run with WORLD for WORLD-built engines, or the same ordered
    subgroup membership for subgroup-built engines. Every collective in an
    engine must be contained in its parent; use WORLD for overlapping TP/CP
    groups that do not fit inside one subgroup. Changing parents requires
    recompilation. Saved engines preserve this contract. Subgroup-parent builds
    currently require ``cache_built_engines=False`` and ``reuse_cached_engines=False``.

    For example, create ``tp_group = dist.new_group([2, 3])`` on every rank,
    then compile and execute on its members inside ``distributed_context(tp_group)``.
    Model collectives naming that group use parent-local IDs ``[0, 1]``. On load,
    enter ``distributed_context(tp_group, loaded_model)`` before inference.
    C++ requires this explicit setup and does not auto-select a group.

    When *module* is supplied the group is also pre-pinned on all TRT engines
    in the module via :func:`set_distributed_mode`, and the configured module
    is yielded so it can be used directly.  A list of modules may be passed to
    pre-pin and yield multiple programs at once.

    **Teardown:** when *module* is supplied, ``__exit__`` calls
    ``release_nccl_comm()`` on all tracked multi-device engines, detaching the
    NCCL communicator from the TRT execution context.  This makes
    ``dist.destroy_process_group()`` safe to call after the block without
    requiring manual ``del trt_model`` / ``gc.collect()`` ordering.

    .. note::

       ``os._exit(0)`` is still recommended after ``destroy_process_group()``
       to avoid segfaults from TRT/CUDA destructors during Python interpreter
       shutdown (GC runs destructors in unpredictable order).

    Example::

        parent_group = dist.group.WORLD

        # Single module:
        with torch_tensorrt.distributed.distributed_context(parent_group, model_a) as m:
            output = m(inp)

        # Multiple modules — yields a list in the same order:
        with torch_tensorrt.distributed.distributed_context(parent_group, [encoder, decoder]) as (enc, dec):
            output = dec(enc(inp))

        # Without module — use for compile-time wrapping:
        with torch_tensorrt.distributed.distributed_context(parent_group):
            trt_model = torch.compile(model, backend="torch_tensorrt", ...)
            output = trt_model(inp)

    Args:
        group:  The ``ProcessGroup`` to use for all TRT NCCL collectives.
        module: Optional compiled/loaded module or list of modules.  When
                provided, :func:`set_distributed_mode` is called on each and
                the module(s) are yielded as the context value.  On exit,
                NCCL communicators are released from all tracked engines.

    Yields:
        *module* when a single module is supplied, a list when multiple modules
        are supplied, or ``None`` when no module is given.
    """
    old = getattr(_state, "pg", None)
    _state.pg = group

    # Normalise to a list internally; track whether caller passed a single module
    # so we can yield the right type.
    if module is None:
        modules: List[nn.Module] = []
        single = False
    elif isinstance(module, nn.Module):
        modules = [module]
        single = True
    else:
        modules = list(module)
        single = False

    try:
        for mod in modules:
            set_distributed_mode(group, mod)
        if single:
            yield modules[0]
        elif modules:
            yield modules
        else:
            yield None
    finally:
        _state.pg = old
        # Release NCCL communicators from all md engines registered on
        # this thread so dist.destroy_process_group() is safe after exit.
        if modules:
            engines = getattr(_state, "md_engines", None) or []
            for engine in engines:
                if getattr(engine, "nccl_initialized", False):
                    engine.release_nccl_comm()
            _state.md_engines = []


def set_distributed_mode(group: Any, module: nn.Module) -> None:
    """Pin *group* as the NCCL process group on every TRT engine in *module*.

    Walks *module* recursively and calls ``set_group_name`` on every
    multi-device TRT engine found.  Also covers engines registered in the
    global ``_md_modules`` registry (populated at engine creation time),
    which handles the ``torch.compile`` case where engines live in dynamo's
    code cache rather than the module tree.

    Args:
        group:  The ``ProcessGroup`` to bind for NCCL collectives.
        module: Compiled or loaded module whose TRT engines should use *group*.
    """
    from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

    group_name = str(group.group_name) if hasattr(group, "group_name") else ""

    if not group_name:
        return

    seen: set[int] = set()

    def pin_engine(engine: Any) -> None:
        _require_native_parent(group, getattr(engine, "native_collective_parent", ""))
        if hasattr(engine, "set_group_name"):
            engine.set_group_name(group_name)

    # Walk the module tree for direct submodules and inlined engines.
    for submod in module.modules():
        if isinstance(submod, TorchTensorRTModule):
            engine = getattr(submod, "engine", None)
            if engine is not None and id(engine) not in seen:
                seen.add(id(engine))
                if engine.requires_native_multidevice:
                    pin_engine(engine)
            continue

        for attr_val in vars(submod).values():
            if (
                id(attr_val) not in seen
                and hasattr(attr_val, "requires_native_multidevice")
                and hasattr(attr_val, "set_group_name")
                and attr_val.requires_native_multidevice
            ):
                seen.add(id(attr_val))
                pin_engine(attr_val)

    # Also pin on any engines in the thread-local registry (handles torch.compile).
    for engine in getattr(_state, "md_engines", None) or []:
        if id(engine) not in seen:
            seen.add(id(engine))
            if getattr(engine, "requires_native_multidevice", False):
                pin_engine(engine)
