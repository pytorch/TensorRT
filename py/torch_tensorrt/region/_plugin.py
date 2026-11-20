# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Snapshot kernel registration arguments for one compilation."""

from __future__ import annotations

import uuid
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from torch_tensorrt.kernels import KernelSpec

# Kernels are Python objects, so a traced program holds only a key into this
# table. The key is random: a program loaded in another process cannot find a
# kernel, or pick up a different one, by accident.
_KERNELS: dict[str, KernelSpec | partial[None]] = {}


# The key of each kernel request already seen. A model is traced again for
# every new input shape, and a rule is applied to every graph of a model.
_KERNEL_KEYS: dict[tuple[Any, ...], str] = {}


def remember_kernel(
    prepared: KernelSpec | partial[None], request: tuple[Any, ...] = ()
) -> str:
    """Keep a prepared kernel for this process and return the key a trace records.

    ``request`` identifies what was asked for, so that asking again for the
    same kernel and options returns the same key instead of a new entry.
    """
    try:
        known = _KERNEL_KEYS.get(request) if request else None
    except TypeError:
        # An option that cannot be hashed, such as a dict of lists.
        request, known = (), None
    if known is not None:
        return known
    key = uuid.uuid4().hex
    _KERNELS[key] = prepared
    if request:
        _KERNEL_KEYS[request] = key
    return key


def find_kernel(key: str) -> Optional[KernelSpec | partial[None]]:
    return _KERNELS.get(key)


def prepare_plugin(kernel: Any, **options: Any) -> KernelSpec | partial[None]:
    """Reuse CUDA's KernelSpec or bind arguments to kernels.triton_op."""
    from torch_tensorrt.kernels import KernelSpec, triton_op

    if isinstance(kernel, KernelSpec):
        supplied = [name for name, value in options.items() if value is not None]
        if supplied:
            raise ValueError(
                "A CUDA KernelSpec provides its own launch configuration; "
                "omit Triton options: " + ", ".join(supplied)
            )
        # Copy declarations without copying callback state or CUDA handles.
        return replace(
            kernel,
            inputs=None if kernel.inputs is None else tuple(kernel.inputs),
            outputs=None if kernel.outputs is None else tuple(kernel.outputs),
            extras=tuple(kernel.extras),
            include_paths=(
                None if kernel.include_paths is None else list(kernel.include_paths)
            ),
        )

    if not isinstance(getattr(kernel, "arg_names", None), (tuple, list)):
        raise TypeError(
            "fused_kernel kernel factory must return a @triton.jit function or a "
            f"CUDA KernelSpec; got {type(kernel).__name__}"
        )
    for name in ("signature", "constexprs"):
        value = options.get(name)
        if value is not None:
            options[name] = dict(value)
    options["constexprs"] = options.get("constexprs") or {}
    # The captured graph supplies the schema and callbacks during materialization.
    return partial(triton_op, kernel=kernel, **options)
