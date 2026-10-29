# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Snapshot kernel registration arguments for one compilation."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch_tensorrt.kernels import KernelSpec


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
            "fused_kernel kernel must be a @triton.jit function or a CUDA KernelSpec"
        )
    for name in ("signature", "constexprs"):
        value = options.get(name)
        if value is not None:
            options[name] = dict(value)
    options["constexprs"] = options.get("constexprs") or {}
    # The captured graph supplies the schema and callbacks during materialization.
    return partial(triton_op, kernel=kernel, **options)
