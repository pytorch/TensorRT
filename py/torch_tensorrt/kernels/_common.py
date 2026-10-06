# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Tensor metadata and symbolic argument helpers shared by kernel frontends."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

import torch

_LOGGER = logging.getLogger(__name__)


def to_trt_dtype(value: torch.dtype) -> Any:
    """Require an exact TensorRT representation for a compiled pointer dtype."""
    import tensorrt as trt

    from torch_tensorrt._enums import dtype

    try:
        converted = dtype._from(value).to(trt.DataType)
        if dtype._from(converted).to(torch.dtype) == value:
            return converted
    except (TypeError, ValueError):
        pass
    raise ValueError(
        f"Kernel signature dtype {value} has no exact TensorRT representation. "
        "Use a supported dtype; engine precision conversion cannot change a "
        "compiled kernel's pointer types."
    )


def check_aot_dtypes(
    op_name: str,
    aot_fn: Callable[..., Any],
    input_dtypes: Sequence[torch.dtype],
    output_dtypes: Sequence[torch.dtype],
) -> Callable[..., Any]:
    """Check actual engine bindings before invoking a compiled kernel's launch."""
    expected_inputs = tuple(to_trt_dtype(value) for value in input_dtypes)
    expected_outputs = tuple(to_trt_dtype(value) for value in output_dtypes)

    def checked(inputs: Any, outputs: Any, tactic: int) -> Any:
        for kind, actual, expected in (
            ("input", inputs, expected_inputs),
            ("output", outputs, expected_outputs),
        ):
            if len(actual) != len(expected):
                raise ValueError(
                    f"Kernel '{op_name}' expected {len(expected)} {kind} "
                    f"descriptor(s), got {len(actual)}."
                )
            for index, (desc, want) in enumerate(zip(actual, expected)):
                got = getattr(desc, "dtype", None)
                if got != want:
                    raise ValueError(
                        f"Kernel '{op_name}' {kind} {index} descriptor has dtype "
                        f"{got}, but the compiled kernel requires {want}."
                    )
        return aot_fn(inputs, outputs, tactic)

    return checked


def make_dtype_capability_validator(
    op_name: str,
    frontend: str,
    input_dtypes: Sequence[Optional[torch.dtype]],
    output_dtypes: Sequence[Optional[torch.dtype]],
    user_validator: Optional[Callable[..., bool]] = None,
) -> Callable[..., bool]:
    """Decline conversion unless every tensor matches the compiled kernel ABI."""

    def _value(node: Any) -> Any:
        meta = getattr(node, "meta", None)
        return meta.get("val") if isinstance(meta, dict) else None

    def _reject(reason: str, *args: Any) -> bool:
        # A declined op falls back to PyTorch; warn so that missing eager
        # implementations do not hide the reason for the eventual CUDA error.
        _LOGGER.warning(
            "Not lowering '%s' to its %s plugin: " + reason,
            op_name,
            frontend,
            *args,
        )
        return False

    def _validator(node: Any, settings: Any = None) -> bool:
        if user_validator is not None and not user_validator(node, settings):
            return False

        args = getattr(node, "args", None)
        if not isinstance(args, (tuple, list)):
            return _reject("FX node has no positional argument metadata.")
        produced = _value(node)
        outputs = list(produced) if isinstance(produced, (tuple, list)) else [produced]
        for kind, actual, expected in (
            ("input", [_value(arg) for arg in args], input_dtypes),
            ("output", outputs, output_dtypes),
        ):
            if len(actual) != len(expected):
                return _reject(
                    "expected %d tensor %s(s), but metadata describes %d.",
                    len(expected),
                    kind,
                    len(actual),
                )
            for index, (got, want) in enumerate(zip(actual, expected)):
                if not isinstance(got, torch.Tensor):
                    return _reject(
                        "%s %d has no Tensor metadata; refusing an unchecked "
                        "pointer binding.",
                        kind,
                        index,
                    )
                if got.dtype != want:
                    return _reject(
                        "%s %d is %s but the kernel was compiled for %s.",
                        kind,
                        index,
                        got.dtype,
                        want,
                    )
                if want == torch.float64 and getattr(
                    settings, "truncate_double", False
                ):
                    return _reject(
                        "%s %d would be bound as float32 with truncate_double=True, "
                        "but the kernel was compiled for float64.",
                        kind,
                        index,
                    )
        return True

    return _validator


def constant_int(value: Any) -> Optional[int]:
    """Read a concrete TensorRT expression without evaluating a symbolic one."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        is_constant = value.is_constant
        if callable(is_constant):  # raw trt.IDimensionExpr API
            is_constant = is_constant()
        if is_constant:
            getter = getattr(value, "constant_value", None)
            getter = getter if getter is not None else value.get_constant_value
            return int(getter() if callable(getter) else getter)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        # TRT 11.2's public accessor can reference an unset _is_dummy field.
        # Its wrapped IDimensionExpr remains usable in an expression builder.
        try:
            expr = value._expr
            if expr is not None and expr.is_constant():
                return int(expr.get_constant_value())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    return None


def as_symint32(value: Any, trtp: Any) -> Any:
    """Keep symbolic values intact; AOT extras require the typed i32 wrapper."""
    return value if isinstance(value, trtp.SymInt32) else trtp.SymInt32(value)


def pack_symint32_args(values: Sequence[Any], trtp: Any) -> Any:
    """Pack validated arguments without SymIntExprs converting ints to its base type."""
    args = trtp.SymIntExprs(len(values))
    for index, value in enumerate(values):
        args[index] = as_symint32(value, trtp)
    return args
