# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Launch information for an automatically registered region plugin."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class PluginSpec:
    """Describe a custom kernel; the region body supplies its Torch contract.

    ``kernel`` is a ``@triton.jit`` function or an existing
    :class:`torch_tensorrt.kernels.KernelSpec`. For Triton, ``grid`` and
    ``extra_args_fn`` receive ``(inputs, outputs)`` containing TensorRT tensor
    descriptors. Runtime kernel arguments must be ordered as input pointers,
    optional i32 scalar extras, then output pointers. The input pointer order
    is the captured body's first-use order, including any parameter or buffer
    live-ins. Output pointers follow production order for the tensors used
    after the region, which may differ from the model's final return order.

    Triton pointer types are inferred from the captured tensors unless
    ``signature`` is supplied. Scalar extras are inferred as i32; their values
    still require ``extra_args_fn``. Every constexpr needs an explicit value.
    A CUDA ``KernelSpec`` carries its own launch configuration, so the Triton
    options must be omitted. CUDA input declarations may be omitted, in which
    case they are named ``arg0``, ``arg1``, and so on.

    This initial interface supports static, contiguous CUDA tensor boundaries
    for inference. It does not infer a kernel implementation or prove that the
    supplied kernel computes the same values as the annotated reference body.
    """

    kernel: Any
    signature: Optional[dict[str, str]] = None
    constexprs: Optional[dict[str, Any]] = None
    grid: Optional[Callable[..., Any]] = None
    extra_args_fn: Optional[Callable[..., Any]] = None
    num_warps: Optional[int] = None
    num_stages: Optional[int] = None

    def __post_init__(self) -> None:
        # Keep importing the public region namespace independent of the kernel
        # compiler. Validation is only needed when constructing a plugin scope.
        from torch_tensorrt.kernels._dsl import KernelSpec
        from torch_tensorrt.kernels._triton import validate_compile_options

        if isinstance(self.kernel, KernelSpec):
            triton_options = (
                "signature",
                "constexprs",
                "grid",
                "extra_args_fn",
                "num_warps",
                "num_stages",
            )
            supplied = [key for key in triton_options if getattr(self, key) is not None]
            if supplied:
                raise ValueError(
                    "A CUDA KernelSpec provides its own launch configuration; "
                    "omit Triton options: " + ", ".join(supplied)
                )
            # The caller may reuse and modify a declarative spec elsewhere.
            # Keep this region's launch declarations independent of those lists.
            object.__setattr__(
                self,
                "kernel",
                replace(
                    self.kernel,
                    inputs=(
                        tuple(self.kernel.inputs)
                        if self.kernel.inputs is not None
                        else None
                    ),
                    outputs=(
                        tuple(self.kernel.outputs)
                        if self.kernel.outputs is not None
                        else None
                    ),
                    extras=tuple(self.kernel.extras),
                    include_paths=(
                        list(self.kernel.include_paths)
                        if self.kernel.include_paths is not None
                        else None
                    ),
                ),
            )
            return

        if not isinstance(getattr(self.kernel, "arg_names", None), (tuple, list)):
            raise TypeError(
                "fuse_in_plugin kernel must be a @triton.jit function or a CUDA KernelSpec"
            )
        if not callable(self.grid):
            raise TypeError("A Triton fuse_in_plugin kernel requires a callable grid")
        if self.extra_args_fn is not None and not callable(self.extra_args_fn):
            raise TypeError("extra_args_fn must be callable or None")
        for key in ("signature", "constexprs"):
            value = getattr(self, key)
            if value is not None:
                if not isinstance(value, dict):
                    raise TypeError(f"{key} must be a dict or None")
                object.__setattr__(self, key, dict(value))
        validate_compile_options(self.num_warps, self.num_stages)
