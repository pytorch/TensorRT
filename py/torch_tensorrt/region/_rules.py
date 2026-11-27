# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Rules that place whole submodules in regions without editing their source.

Pass them to a compile call as ``regions={selector: rule}``. A selector is a
module path such as ``"model.layers.*.mlp"``, a module class, or a module of
the model being compiled.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from types import FunctionType
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import torch

from ._constants import SUPPORTED_BACKENDS
from ._operators import check_operator

# A position among the tensors the module call receives, or the name of one of
# the module's parameters or buffers.
RuleInput = Union[int, str]
# A module path, a module class, or a module of the model being compiled.
RegionSelector = Union[str, type, torch.nn.Module]


def check_name(name: Optional[str], api: str) -> None:
    if name is not None and not isinstance(name, str):
        raise TypeError(f"{api} name must be a string or None")


def check_backend(backend: str) -> None:
    if not isinstance(backend, str):
        raise TypeError("execute_in_torch backend must be a string")
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported execute_in_torch backend {backend!r}; "
            "expected 'eager' or 'inductor'"
        )


def check_operator_type(op: Any) -> None:
    if not isinstance(
        op,
        (
            torch._library.custom_ops.CustomOpDef,
            torch._ops.OpOverload,
            torch._ops.OpOverloadPacket,
        ),
    ):
        raise TypeError(
            "fused_op requires a torch.library.custom_op function or "
            "torch.ops.<namespace>.<name>"
        )


def check_kernel_options(
    kernel: Any, grid: Any = None, extra_args_fn: Any = None
) -> None:
    # Compilation passes these to the kernel registry as constants: Dynamo
    # converts functions and partials, but not kernel objects or bound methods.
    if not isinstance(kernel, (FunctionType, partial)):
        raise TypeError(
            "fused_kernel kernel must be a function or functools.partial returning "
            f"the kernel, such as kernel=lambda: my_kernel; got {type(kernel).__name__}"
        )
    for option, callback in (("grid", grid), ("extra_args_fn", extra_args_fn)):
        if callback is not None and not isinstance(callback, (FunctionType, partial)):
            raise TypeError(
                f"fused_kernel {option} must be a function or functools.partial; "
                f"got {type(callback).__name__}"
            )


def _check_rule_inputs(inputs: Optional[Sequence[RuleInput]], api: str) -> None:
    if inputs is None:
        return
    if not isinstance(inputs, (tuple, list)) or not all(
        isinstance(item, (int, str)) and not isinstance(item, bool) for item in inputs
    ):
        raise TypeError(
            f"{api} inputs must be a tuple or list of positions (int) and "
            "parameter or buffer names (str)"
        )
    if any(isinstance(item, int) and item < 0 for item in inputs):
        raise ValueError(f"{api} input positions must not be negative")


@dataclass(frozen=True, eq=False)
class ExecuteInTorch:
    """Rule: keep every call of the selected modules outside TensorRT.

    The rule form of :func:`execute_in_torch`. ``backend="eager"`` runs the
    module's operations in PyTorch; ``"inductor"`` compiles them on first use.
    ``name`` labels diagnostics and defaults to the module's path.
    """

    backend: str = "eager"
    name: Optional[str] = None

    def __post_init__(self) -> None:
        check_name(self.name, "ExecuteInTorch")
        check_backend(self.backend)


@dataclass(frozen=True, eq=False)
class FusedOp:
    """Rule: replace every call of the selected modules with a custom operator.

    The rule form of :func:`fused_op`. The operator takes the tensors the
    module reads and returns the tensors the module returns, in the order the
    module returns them. A tensor the module hands through unchanged is not
    one of them.

    ``inputs`` lists the operator's arguments in order. An integer is a
    position among the tensors the module call receives (positional arguments
    first, then keyword arguments in the order the call passes them); a string
    names a parameter or buffer of the module, such as ``"weight"``. Omit
    ``inputs`` only when the module reads one tensor.

    The order of a call's tensors is known when ``torch_tensorrt.compile``
    traces the ``nn.Module`` with ``ir="dynamo"``, or when the program was
    exported with ``preserve_module_call_signature``. Otherwise, as under
    ``torch.compile``, the module must read one tensor from its caller, which
    is position 0, and the program must read one of its results.
    """

    op: Any
    inputs: Optional[Sequence[RuleInput]] = None
    name: Optional[str] = None

    def __post_init__(self) -> None:
        check_name(self.name, "FusedOp")
        check_operator_type(self.op)
        check_operator(self.op)
        _check_rule_inputs(self.inputs, "FusedOp")


@dataclass(frozen=True, eq=False)
class FusedKernel:
    """Rule: replace every call of the selected modules with one kernel plugin.

    The rule form of :func:`fused_kernel`, with the same kernel options.
    ``inputs`` and the order of the kernel's outputs follow :class:`FusedOp`.
    """

    kernel: Callable[[], Any]
    inputs: Optional[Sequence[RuleInput]] = None
    name: Optional[str] = None
    signature: Optional[dict[str, str]] = None
    constexprs: Optional[dict[str, Any]] = None
    grid: Optional[Callable[..., Any]] = None
    extra_args_fn: Optional[Callable[..., Any]] = None
    num_warps: Optional[int] = None
    num_stages: Optional[int] = None

    def __post_init__(self) -> None:
        check_name(self.name, "FusedKernel")
        check_kernel_options(self.kernel, self.grid, self.extra_args_fn)
        _check_rule_inputs(self.inputs, "FusedKernel")


RegionRule = Union[ExecuteInTorch, FusedOp, FusedKernel]
RegionRules = Mapping[RegionSelector, RegionRule]


def check_rules(regions: Any) -> None:
    """Reject a ``regions=`` value that is not ``{selector: rule}``."""
    if not isinstance(regions, Mapping):
        raise TypeError(
            "regions must map module paths, module classes or modules to region "
            'rules, such as {"encoder.layers.*.mlp": '
            "torch_tensorrt.region.ExecuteInTorch()}"
        )
    for selector, rule in regions.items():
        if isinstance(selector, str):
            if not selector:
                raise ValueError("A region rule cannot select the whole model")
        elif not isinstance(selector, torch.nn.Module) and not (
            isinstance(selector, type) and issubclass(selector, torch.nn.Module)
        ):
            raise TypeError(
                "A region rule selects modules by path (str), by module class, "
                f"or by the module itself; got {selector!r}"
            )
        if not isinstance(rule, (ExecuteInTorch, FusedOp, FusedKernel)):
            raise TypeError(
                f"The region rule for {selector!r} must be ExecuteInTorch, FusedOp "
                f"or FusedKernel; got {type(rule).__name__}"
            )
