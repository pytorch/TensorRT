# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import ctypes
import difflib
import functools
import gc
import importlib.util
import logging
import os
import platform
import sys
from dataclasses import fields, replace
from enum import Enum
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import psutil
import sympy
import tensorrt as trt
import torch
from torch._subclasses.fake_tensor import FakeScriptObject, FakeTensor
from torch.fx.experimental.proxy_tensor import unset_fake_temporarily
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._sympy.numbers import int_oo
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import dtype
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt._Input import Input
from torch_tensorrt._utils import is_tensorrt_version_supported
from torch_tensorrt.dynamo import _defaults
from torch_tensorrt.dynamo._defaults import default_device
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from torch_tensorrt.dynamo._settings import CompilationSettings

from packaging import version

from .types import TRTDataType

logger = logging.getLogger(__name__)

# When a compiler setting is removed, add a tuple to this dict with the removed option as the key and the reason as the value.
_REMOVED_COMPILER_KWARGS = {
    "debug": (
        "Use `with torch_tensorrt.dynamo.Debugger(...)` to wrap the compilation "
        "call instead."
    ),
    "enable_cross_compile_for_windows": (
        "Use `torch_tensorrt.dynamo.cross_compile_for_windows()` instead."
    ),
    "enabled_precisions": (
        "Dynamo compilation is strongly typed and uses the precisions in the "
        "model. Use the autocast options to change model precision."
    ),
    "make_refittable": ("Use `immutable_weights=False` to build a refittable engine."),
    "refit": "Use `immutable_weights=False` to build a refittable engine.",
    "truncate_long_and_double": (
        "Use `truncate_double` instead; int64 is now natively supported."
    ),
    "use_explicit_typing": (
        "Dynamo compilation is now always strongly typed, so this option is "
        "no longer necessary."
    ),
    "use_python_runtime": (
        "The Python and C++ runtimes have been merged; the runtime is selected "
        "automatically based on whether the C++ Torch-TensorRT runtime is available."
    ),
}


def validate_compiler_options(
    kwargs: Dict[str, Any],
    valid_options: Collection[str],
    api_name: str,
) -> None:
    """Reject removed, misspelled, or unsupported compiler options."""
    removed_options = _REMOVED_COMPILER_KWARGS.keys() & kwargs.keys()
    if removed_options:
        raise TypeError(
            "The following compiler options have been removed:\n- "
            + "\n- ".join(
                f"`{option}` has been removed. {_REMOVED_COMPILER_KWARGS[option]}"
                for option in sorted(removed_options)
            )
        )

    unexpected_options = kwargs.keys() - set(valid_options)
    if not unexpected_options:
        return

    option = sorted(unexpected_options)[0]
    suggestion = difflib.get_close_matches(
        option, sorted(valid_options), n=1, cutoff=0.6
    )
    suggestion_text = f" Did you mean '{suggestion[0]}'?" if suggestion else ""
    raise TypeError(
        f"{api_name}() got an unexpected keyword argument "
        f"'{option}'.{suggestion_text}"
    )


COSINE_THRESHOLD = 0.99
DYNAMIC_DIM = -1
RTOL = 5e-3
ATOL = 5e-3
CPU_DEVICE = "cpu"


class Frameworks(Enum):
    NUMPY = "numpy"
    TORCH = "torch"
    TRT = "trt"


DataTypeEquivalence: Dict[
    TRTDataType, Dict[Frameworks, Union[TRTDataType, np.dtype, torch.dtype]]
] = {
    trt.int8: {
        Frameworks.NUMPY: np.int8,
        Frameworks.TORCH: torch.int8,
        Frameworks.TRT: trt.int8,
    },
    trt.int32: {
        Frameworks.NUMPY: np.int32,
        Frameworks.TORCH: torch.int32,
        Frameworks.TRT: trt.int32,
    },
    trt.int64: {
        Frameworks.NUMPY: np.int64,
        Frameworks.TORCH: torch.int64,
        Frameworks.TRT: trt.int64,
    },
    trt.float16: {
        Frameworks.NUMPY: np.float16,
        Frameworks.TORCH: torch.float16,
        Frameworks.TRT: trt.float16,
    },
    trt.float32: {
        Frameworks.NUMPY: np.float32,
        Frameworks.TORCH: torch.float32,
        Frameworks.TRT: trt.float32,
    },
    trt.bool: {
        Frameworks.NUMPY: bool,
        Frameworks.TORCH: torch.bool,
        Frameworks.TRT: trt.bool,
    },
}

if is_tensorrt_version_supported("7.0"):
    DataTypeEquivalence[trt.bool] = {
        Frameworks.NUMPY: np.bool_,
        Frameworks.TORCH: torch.bool,
        Frameworks.TRT: trt.bool,
    }


COMPLEX_TO_REAL_DTYPE: Dict[torch.dtype, torch.dtype] = {
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}

COMPLEX_DTYPES: frozenset = frozenset(COMPLEX_TO_REAL_DTYPE)  # type: ignore[type-arg]


def unified_dtype_converter(
    dtype: Union[TRTDataType, torch.dtype, np.dtype], to: Frameworks
) -> Union[np.dtype, torch.dtype, TRTDataType]:
    """
    Convert TensorRT, Numpy, or Torch data types to any other of those data types.
    Args:
        dtype (TRTDataType, torch.dtype, np.dtype): A TensorRT, Numpy, or Torch data type.
        to (Frameworks): The framework to convert the data type to.
    Returns:
        The equivalent data type in the requested framework.
    """
    assert to in Frameworks, f"Expected valid Framework for translation, got {to}"
    trt_major_version = int(trt.__version__.split(".")[0])
    if dtype in (np.int8, torch.int8, trt.int8):
        return DataTypeEquivalence[trt.int8][to]
    elif trt_major_version >= 7 and dtype in (np.bool_, torch.bool, trt.bool):
        return DataTypeEquivalence[trt.bool][to]
    elif dtype in (np.int32, torch.int32, trt.int32):
        return DataTypeEquivalence[trt.int32][to]
    elif dtype in (np.int64, torch.int64, trt.int64):
        return DataTypeEquivalence[trt.int64][to]
    elif dtype in (np.float16, torch.float16, trt.float16):
        return DataTypeEquivalence[trt.float16][to]
    elif dtype in (np.float32, torch.float32, trt.float32):
        return DataTypeEquivalence[trt.float32][to]
    else:
        raise TypeError("%s is not a supported dtype" % dtype)


def _module_occupies_cuda(module: torch.nn.Module) -> bool:
    """True if any parameter or buffer still lives on CUDA."""
    for tensor in module.parameters():
        if tensor.is_cuda:
            return True
    for tensor in module.buffers():
        if tensor.is_cuda:
            return True
    return False


def deallocate_module(
    module: torch.nn.Module, release_every_bytes: int = 1 << 30
) -> None:
    """Move ``module``'s parameters and buffers to the CPU and release their GPU memory.

    No-op when nothing is on CUDA, so a second call after compile() already
    offloaded (and CPU-only compiles) skip the move, ``empty_cache``, and ``gc``.

    ``module.to("cpu")`` leaves every freed block in PyTorch's CUDA cache until it
    returns, so for the length of the move the process holds both the GPU and the CPU
    copy of the weights. On unified memory systems those come out of the same DRAM.
    Instead the cache is emptied each time ``release_every_bytes`` have moved, so the
    GPU copy shrinks as the CPU copy grows.
    """
    if not torch.cuda.is_available() or not _module_occupies_cuda(module):
        return

    moved_since_release = 0

    def to_cpu(t: torch.Tensor) -> torch.Tensor:
        nonlocal moved_since_release
        if t.device.type == "cuda":
            # The tensor moved before this one has already been rebound to its CPU
            # copy, so its GPU block is free by now.
            if moved_since_release >= release_every_bytes:
                torch.cuda.empty_cache()
                moved_since_release = 0
            # Only paces the releases; sparse and other layouts don't define nbytes.
            moved_since_release += t.nbytes if t.layout == torch.strided else 0
        return t.to(CPU_DEVICE)

    # Same conversion as ``module.to(CPU_DEVICE)``, which is ``_apply`` with a ``.to``.
    module._apply(to_cpu)
    gc.collect()
    torch.cuda.empty_cache()


def pin_torch_executed_state(module: torch.nn.Module, device: torch.device) -> None:
    """Move the state that a compiled module executes in PyTorch to ``device``.

    The TensorRT submodules hold their weights inside the engine, so they need no state here.
    The rest of the graph reads its state in two ways: through a submodule that the
    partitioner kept in PyTorch, or through a ``get_attr`` node in the root graph. When no
    operation converts, ``compile_module`` returns the input graph, and the root graph reads
    all the state.
    """
    # A lone TensorRT module executes nothing in PyTorch.
    if not isinstance(module, torch.fx.GraphModule):
        return
    for node in module.graph.nodes:
        if node.op == "get_attr":
            _move_attr(module, str(node.target), device)
        elif node.op == "call_module" and "_run_on_acc" not in str(node.target):
            module.get_submodule(str(node.target)).to(device)


def _move_attr(module: torch.nn.Module, target: str, device: torch.device) -> None:
    prefix, _, name = target.rpartition(".")
    owner = module.get_submodule(prefix)
    value = getattr(owner, name)
    if isinstance(value, torch.nn.Parameter):
        # nn.Module.to() sets .data in the same way. The source model holds this same
        # Parameter, so the source model sees the move too.
        value.data = value.data.to(device)
    elif isinstance(value, torch.Tensor):
        setattr(owner, name, value.to(device))


def cosine_similarity(gt_tensor: torch.Tensor, pred_tensor: torch.Tensor) -> float:
    gt_tensor = gt_tensor.flatten().to(torch.float32)
    pred_tensor = pred_tensor.flatten().to(torch.float32)
    if torch.sum(gt_tensor) == 0.0 or torch.sum(pred_tensor) == 0.0:
        if torch.allclose(gt_tensor, pred_tensor, atol=1e-4, rtol=1e-4, equal_nan=True):
            return 1.0
    res_t = torch.nn.functional.cosine_similarity(
        gt_tensor, pred_tensor, dim=0, eps=1e-6
    )
    res: float = res_t.cpu().detach().item()

    return res


def input_is_dynamic(inputs: Sequence[Union[Input, torch.Tensor, FakeTensor]]) -> bool:
    """
    Return true if any of inputs have dynamic shapes. Supported types are torch_tensorrt.Input | torch.Tensor
    """
    for input in inputs:
        if isinstance(input, torch.Tensor):
            if contains_sym_int(input.shape):
                return True
        elif isinstance(input, Input):
            if input.shape_mode == Input._ShapeMode.DYNAMIC:
                return True
        else:
            raise AssertionError(
                f"Invalid input type ({type(input)}) found. Supported types are torch_tensorrt.Input | torch.Tensor"
            )

    return False


def get_torch_tensor(
    input: Input,
    device: torch.device,
    mode: str = "",
) -> Union[int, List[int], torch.Tensor]:
    if input.is_shape_tensor:
        assert isinstance(input.shape, dict)
        opt_shape = input.shape["opt_shape"]
        # Scalar shape tensors supply SymInt parameters, while multi-element
        # shape tensors supply SymInt[] parameters such as aten.full's size.
        return opt_shape[0] if len(opt_shape) == 1 else list(opt_shape)

    if len(mode) > 0:
        return input.example_tensor(mode).to(device)
    else:
        return input.torch_tensor.to(device)


def get_torch_inputs(
    inputs: Sequence[Input] | Dict[str, Any],
    device: Union[Device, torch.device, str],
    mode: str = "",
) -> (
    Sequence[Union[int, List[int], torch.Tensor]]
    | Dict[str, Union[int, List[int], torch.Tensor]]
):
    """
    Return the torch_tensor from the Input object. If mode is set, this implies
    user is using dynamic shaped inputs and return the corresponding input based
    on the mode requested.
    """
    device = to_torch_device(device)

    if isinstance(inputs, dict):
        result_dict: Dict[str, Union[int, List[int], torch.Tensor]] = {}
        for k, v in inputs.items():
            if isinstance(v, (list, tuple, dict)):
                result_dict[k] = get_torch_inputs(v, device)
            elif isinstance(v, Input):
                result_dict[k] = get_torch_tensor(v, device, mode)
        return result_dict
    else:
        result_list: List[Union[int, List[int], torch.Tensor]] = []
        for input in inputs:
            if isinstance(input, Input):
                result_list.append(get_torch_tensor(input, device, mode))
            elif isinstance(input, torch.Tensor):
                result_list.append(input.to(device))
            else:
                raise AssertionError(f"Input type {type(input)} is not a valid type")
        return result_list


def get_model_device(module: torch.fx.GraphModule) -> torch.device:
    """
    Returns the device on which the module parameters exist.
    """
    device = None
    for parameter in list(module.parameters()):
        if isinstance(parameter, (torch.nn.parameter.Parameter, torch.Tensor)):
            return parameter.device

    for buffer in list(module.buffers()):
        if isinstance(buffer, (torch.Tensor)):
            return buffer.device

    if device is None:
        device = to_torch_device(default_device())

    return device


def set_log_level(parent_logger: Any, level: Any) -> None:
    """
    Sets the log level to the user provided level.
    This is used to set debug logging at a global level
    at entry points of tracing, dynamo and torch_compile compilation.
    And set log level for c++ torch trt logger if runtime is available.
    """
    if parent_logger:
        parent_logger.setLevel(level)

    if ENABLED_FEATURES.torch_tensorrt_runtime:
        if level == logging.DEBUG:
            log_level = trt.ILogger.Severity.VERBOSE
        elif level == logging.INFO:
            log_level = trt.ILogger.Severity.INFO
        elif level == logging.WARNING:
            log_level = trt.ILogger.Severity.WARNING
        elif level == logging.ERROR:
            log_level = trt.ILogger.Severity.ERROR
        elif level == logging.CRITICAL:
            log_level = trt.ILogger.Severity.INTERNAL_ERROR
        else:
            raise AssertionError(f"{level} is not valid log level")

        torch.ops.tensorrt.set_logging_level(int(log_level))


def prepare_inputs(
    inputs: Input | torch.Tensor | Sequence[Any] | Dict[Any, Any],
    disable_memory_format_check: bool = False,
) -> Any:
    """
    We take a nested group of torch.Tensors or scalars and convert them into torchtrt.Input's
    """
    # Any tensors created inside this call will be FakeTensors if it's inside a torch.compile session
    # So, we disable fake mode temporarily.
    with unset_fake_temporarily():
        if inputs is None:
            return None

        elif isinstance(inputs, Input):
            return inputs

        elif isinstance(inputs, torch.Tensor):
            if inputs.is_complex():
                # Complex tensors are lowered to real tensors with an extra last
                # dimension of size 2 (real, imag) by complex_graph_detection.
                # Build an Input whose shape/dtype reflects the lowered representation
                # while keeping the original complex tensor for tracing (torch.export
                # needs the complex tensor to trace the model correctly).
                real_view = torch.view_as_real(inputs.contiguous())
                inp = Input.from_tensor(
                    real_view, disable_memory_format_check=disable_memory_format_check
                )
                # Restore the original complex tensor so dynamo_trace can export
                # the model with the correct input dtype.
                inp.torch_tensor = inputs
                return inp
            # Pass the tensor directly — torch.tensor() would create a full
            # data copy, which wastes GPU memory when torch.compile lifts
            # model parameters as graph inputs.
            # Input.from_tensor only reads shape/dtype/format metadata.
            return Input.from_tensor(
                inputs,
                disable_memory_format_check=disable_memory_format_check,
            )

        elif isinstance(inputs, (int, float, bool)):
            return Input.from_tensor(
                torch.tensor(inputs),
                disable_memory_format_check=disable_memory_format_check,
            )

        elif isinstance(inputs, (list, tuple)):
            torchtrt_input_list = []
            for input_obj in inputs:
                torchtrt_input = prepare_inputs(
                    input_obj, disable_memory_format_check=disable_memory_format_check
                )
                torchtrt_input_list.append(torchtrt_input)

            return (
                torchtrt_input_list
                if isinstance(inputs, list)
                else tuple(torchtrt_input_list)
            )

        elif isinstance(inputs, dict):
            torchtrt_inputs_dict: Dict[Any, Any] = dict()

            for key, input_obj in inputs.items():
                torchtrt_input = prepare_inputs(
                    input_obj, disable_memory_format_check=disable_memory_format_check
                )
                torchtrt_inputs_dict[key] = torchtrt_input

            return torchtrt_inputs_dict

        else:
            raise ValueError(
                f"Invalid input type {type(inputs)} encountered in the dynamo_compile input parsing. "
                + "Allowed input types: {torch_tensorrt.Input, torch.Tensor, list, tuple, dict}"
            )


def parse_complex_tensor_structs(
    inputs: Input | torch.Tensor | Sequence[Any] | Dict[Any, Any],
    attribute_to_extract: str,
    apply_fn: Callable[[Any], Any] = lambda x: x,
) -> Any:
    """Parses complex structures of Tensors and returns a mirrored structure
    Extracts key attributes of each singular element, while reconstructing the struct
    Optionally applies a function to each attribute before returning
    """
    if isinstance(inputs, (torch.Tensor, Input)):
        return apply_fn(getattr(inputs, attribute_to_extract, None))
    elif isinstance(inputs, (int, float, bool)):
        # inputs is a python scalar value
        inputs_torch = torch.tensor(inputs)
        return apply_fn(getattr(inputs_torch, attribute_to_extract, None))

    elif isinstance(inputs, (list, tuple)):
        torchtrt_input_list = []
        for input_obj in inputs:
            torchtrt_input = parse_complex_tensor_structs(
                input_obj, attribute_to_extract, apply_fn
            )
            torchtrt_input_list.append(torchtrt_input)

        return (
            torchtrt_input_list
            if isinstance(inputs, list)
            else tuple(torchtrt_input_list)
        )

    elif isinstance(inputs, dict):
        torchtrt_inputs_dict: Dict[Any, Any] = dict()

        for key, input_obj in inputs.items():
            torchtrt_input = parse_complex_tensor_structs(
                input_obj, attribute_to_extract, apply_fn
            )
            torchtrt_inputs_dict[key] = torchtrt_input

        return torchtrt_inputs_dict

    else:
        raise ValueError(
            f"Invalid input type {type(inputs)} encountered during Dynamo input parsing. "
            + "Allowed input types: {torch_tensorrt.Input, torch.Tensor, list, tuple, dict}"
        )


def contains_sym_int(tensor: torch.Tensor) -> bool:
    """
    Returns true if the given tensor has symbolic shape.
    """
    return any(isinstance(dim, torch.SymInt) for dim in tensor)


def extract_var_range_info(
    symbolic_integer: torch.SymInt,
    user_symbol_bounds: Optional[Dict[sympy.Symbol, Tuple[int, int]]] = None,
) -> Dict[str, Optional[int]]:
    """Return ``{min, max, opt}`` for a symbolic integer.

    ``user_symbol_bounds`` (read-only ``{sym: (min, max)}``) is consulted only
    when the exporter's upper is unbounded; finite exporter bounds always win.
    The lower is intersected with the exporter's so the 0/1 specialization
    survives even if the user passes ``min_shape=0``.
    """
    node = symbolic_integer.node
    expr = node.expr
    shape_env = node.shape_env
    # An expr can be a independent SymInt node (eg: s0 or s1) or a composition of them eg: (48*s0 or s0*s1).
    # In the case of expr which has symbolic computation, bound_sympy evaluates them.
    # https://pytorch.org/docs/stable/generated/torch.fx.experimental.symbolic_shapes.ShapeEnv.html#torch.fx.experimental.symbolic_shapes.ShapeEnv.bound_sympy
    # expr.xreplace replaces the symbolic variables with their current values and computes the expression.
    var_range = shape_env.var_to_range.get(expr, None) or shape_env.bound_sympy(expr)
    # PyTorch commit adb73cc: var_to_val->backed_var_to_val, unbacked_var_to_val->real_tensor_prop_unbacked_vals
    var_to_val_map = getattr(shape_env, "backed_var_to_val", None) or getattr(
        shape_env, "var_to_val", None
    )
    if var_to_val_map is None:
        var_to_val_map = {}
    unbacked_map = getattr(
        shape_env, "real_tensor_prop_unbacked_vals", None
    ) or getattr(shape_env, "unbacked_var_to_val", None)
    if unbacked_map is None:
        unbacked_map = {}
    var_val = (
        var_to_val_map.get(expr, None)
        or unbacked_map.get(expr, None)
        or expr.xreplace(var_to_val_map)
    )
    assert var_range, var_val

    # ``var_to_range`` returns ``int_oo`` for unbounded; ``bound_sympy`` (used
    # for composite exprs like ``s0+s1``) returns ``sympy.oo`` instead. They
    # are distinct objects -- check both, else ``int(sympy.oo)`` raises.
    def _bound_to_int_or_none(value: Any) -> Optional[int]:
        if value is int_oo or value is -int_oo:
            return None
        if value == sympy.oo or value == -sympy.oo:
            return None
        try:
            bound = int(value)
        except (TypeError, OverflowError, AttributeError):
            return None
        # PyTorch uses the largest representable size as a finite-looking
        # sentinel for an effectively unbounded symbolic dimension.
        if abs(bound) >= sys.maxsize - 1:
            return None
        return bound

    min_val_opt = _bound_to_int_or_none(var_range.lower)
    max_val = _bound_to_int_or_none(var_range.upper)
    # Unbounded lower shouldn't happen for tensor dims; fall back to 1.
    min_val = min_val_opt if min_val_opt is not None else 1

    # Torchdynamo 0/1 specialization outlier
    min_val = 1 if min_val == 2 else min_val

    # Apply user bounds whenever present. ``_build_user_symbol_bounds`` already
    # rejects user ranges that exceed the exporter, so the only cases reaching
    # here are: Dim.DYNAMIC (max_val is None), strict subset, or the 1->2
    # specialization. Clamp defensively in case validation was skipped (no
    # ShapeEnv access path).
    if (
        user_symbol_bounds
        and isinstance(expr, sympy.Symbol)
        and expr in user_symbol_bounds
    ):
        user_min, user_max = user_symbol_bounds[expr]
        min_val = max(min_val, int(user_min))
        max_val = int(user_max) if max_val is None else min(max_val, int(user_max))

    min_max_opt: Dict[str, Optional[int]] = {}
    min_max_opt["min"] = min_val
    min_max_opt["max"] = max_val
    if isinstance(var_val, (sympy.core.numbers.Integer, int)):
        min_max_opt["opt"] = int(var_val)

    return min_max_opt


def validate_optimization_profiles(input_specs: Sequence[Input]) -> int:
    """Validate multi-profile inputs and return the number of profiles.

    ``Input.profiles`` is an ordered list; the list index is the TRT
    optimization-profile index selected at runtime. Every dynamic input that
    declares ``profiles`` must declare the *same number* of profiles.
    Static inputs (or dynamic inputs without profiles) are allowed and reuse
    their single shape in every profile.

    Returns the number of *declared* optimization profiles, i.e. ``0`` when no
    input declares ``profiles`` (the multi-profile feature is unused and the
    actual profile count is decided downstream: ``0`` for fully static engines,
    ``1`` for dynamic ones).
    """
    num_profiles = 0
    for inp in input_specs:
        profiles = getattr(inp, "profiles", None)
        if not profiles:
            continue
        if num_profiles == 0:
            num_profiles = len(profiles)
        elif len(profiles) != num_profiles:
            raise ValueError(
                "All inputs declaring optimization profiles must declare the same "
                f"number of profiles, found both {num_profiles} and {len(profiles)}."
            )
    return num_profiles


def extract_var_range_info_for_profile(
    symbolic_integer: torch.SymInt,
    symbol_values: Dict[str, int],
) -> int:
    """Evaluate a symbolic dimension at a single profile corner.

    ``symbol_values`` maps source symbol name (e.g. ``"s0"``) to the concrete
    integer value of that symbol at one corner (min / opt / max) of a profile.
    The intermediate ``SymInt`` expression (e.g. ``s0/4``) is evaluated by
    substituting those values. Shape ops in export produce affine expressions
    that are monotonic in each source symbol, so per-corner substitution is
    exact.
    """
    node = symbolic_integer.node
    expr = node.expr
    # A fully-static expression may already be a Python/sympy integer.
    if isinstance(expr, int):
        return int(expr)
    free_symbols: Any = getattr(expr, "free_symbols", set())
    substitution = {
        sym: symbol_values[sym.name]
        for sym in free_symbols
        if getattr(sym, "name", None) in symbol_values
    }
    evaluated = expr.xreplace(substitution) if substitution else expr
    if getattr(evaluated, "free_symbols", set()):
        # A free symbol was missing from symbol_values; caller should fall back.
        raise KeyError(
            f"Could not fully evaluate symbolic dim {expr} with symbol values "
            f"{symbol_values} (unresolved free symbols remain)."
        )
    return int(evaluated)


def unwrap_tensor_shape(
    tensor: Union[torch.Tensor, FakeTensor, torch.SymInt], mode: Optional[str] = ""
) -> Sequence[Union[Optional[int], Tuple[Optional[int], Optional[int]]]]:
    """
    This is a helper function used to print/return the shape of the tensor.
    For regular torch.tensor's, it returns the static shape.
    For symbolic tensors, eg:(1, s0, 4), this function returns [1, [min, max], 4]. The min
    and max correspond to the lower and upper values of s0 symbolic dimension.
    """
    tensor_shape: List[Union[Optional[int], Tuple[Optional[int], Optional[int]]]] = []
    # for dimension in tensor.shape:
    if isinstance(tensor, int):
        tensor_shape.append(tensor)
    elif isinstance(tensor, torch.SymInt):
        min_max_opt = extract_var_range_info(tensor)
        if mode:
            tensor_shape.append(min_max_opt[mode])
        else:
            tensor_shape.append((min_max_opt["min"], min_max_opt["max"]))
    elif isinstance(tensor, torch.SymFloat):
        # SymFloats can be an input to graph sometimes. Although SymFloat is scalar value, we treat it as a 1D tensor throughout Torch-TRT codebase.
        tensor_shape.append(1)
    elif isinstance(tensor, (torch.Tensor, FakeTensor)):
        for dimension in tensor.shape:
            tensor_shape.extend(unwrap_tensor_shape(dimension, mode=mode))

    return tuple(tensor_shape)


def unwrap_tensor_dtype(tensor: Union[torch.Tensor, FakeTensor, torch.SymInt]) -> Any:
    """
    Returns the dtype of torch.tensor or FakeTensor. For symbolic integers, we return int64
    """
    if isinstance(tensor, (torch.Tensor, FakeTensor)):
        return tensor.dtype
    elif isinstance(tensor, (int, float, bool)):
        return torch.tensor(tensor).dtype
    elif isinstance(tensor, torch.SymInt):
        return torch.int64
    elif isinstance(tensor, torch.SymFloat):
        return torch.float32
    elif isinstance(tensor, FakeScriptObject) or tensor is None:
        # In torch.compile distributed cases, the DeviceMesh for each tensor is passed as an argument to the graph as a FakeScriptObject
        # Case where we explicitly pass one of the inputs to be None (eg: FLUX.1-dev)
        return None
    else:
        raise ValueError(f"Found invalid tensor type {type(tensor)}")


def get_graph_io_attrs(
    io_nodes: Sequence[torch.fx.Node], attr_type: str
) -> Sequence[Any]:
    """
    Returns a list of attributes (shapes or dtypes) of the I/O nodes
    """
    assert attr_type in ["shape", "dtype"]
    attr_fn = unwrap_tensor_shape if attr_type == "shape" else unwrap_tensor_dtype
    graph_io_attrs = []
    for node in io_nodes:
        if "val" in node.meta:
            metadata = node.meta["val"]
            if isinstance(metadata, (tuple, list)):
                for tensor in metadata:
                    graph_io_attrs.append(attr_fn(tensor))
            else:
                graph_io_attrs.append(attr_fn(metadata))

    return graph_io_attrs


def parse_graph_io(module: torch.fx.GraphModule, dryrun_tracker: Any) -> None:
    """
    Parse the graph I/O shape/dtype info for the whole graph and store in the dryrun tracker
    """
    # Parse inputs of the graph
    input_nodes = [node for node in module.graph.nodes if node.op == "placeholder"]
    input_shapes = get_graph_io_attrs(input_nodes, "shape")
    input_dtypes = get_graph_io_attrs(input_nodes, "dtype")
    dryrun_tracker.input_shapes = input_shapes
    dryrun_tracker.input_dtypes = input_dtypes

    # Parse outputs of the graph
    mark_output_nodes = [node for node in module.graph.nodes if node.op == "output"]
    output_nodes = []
    for node in mark_output_nodes:
        output_nodes.extend(node.all_input_nodes)
    output_shapes = get_graph_io_attrs(output_nodes, "shape")
    output_dtypes = get_graph_io_attrs(output_nodes, "dtype")
    dryrun_tracker.output_shapes = output_shapes
    dryrun_tracker.output_dtypes = output_dtypes


def to_torch_device(device: Optional[Union[Device, torch.device, str]]) -> torch.device:
    """Cast a device-type to torch.device

    Returns the corresponding torch.device
    """
    if isinstance(device, Device):
        return device.to(torch.device)

    elif isinstance(device, torch.device):
        return device

    elif device is None:
        return torch.device(torch.cuda.current_device())

    else:
        return torch.device(device)


def to_torch_tensorrt_device(
    device: Optional[Union[Device, torch.device, str]],
) -> Device:
    """Cast a device-type to torch_tensorrt.Device

    Returns the corresponding torch_tensorrt.Device
    """
    return Device._from(device)


def parse_dynamo_kwargs(
    kwargs: Any,
) -> Tuple[CompilationSettings, Optional[BaseEngineCache]]:
    """Parses the kwargs field of a Dynamo backend

    Args:
        kwargs: Keyword arguments dictionary provided to the backend
    Returns:
        CompilationSettings object with relevant kwargs
    """
    # Initialize an empty CompilationSettings object
    settings = CompilationSettings()

    # If the user specifies keyword args, overwrite those fields in settings
    # Validate all specified kwargs to ensure they are true fields of the dataclass
    #
    # Note: kwargs provided by torch.compile are wrapped in the "options" key
    if kwargs:
        if "options" in kwargs and len(kwargs) == 1:
            kwargs = kwargs["options"]

        valid_attrs = {attr.name for attr in fields(settings)}
        validate_compiler_options(kwargs, valid_attrs, "torch.compile")
        settings = replace(settings, **kwargs)

    # Ensure device is a torch_tensorrt Device
    settings.device = to_torch_tensorrt_device(settings.device)

    # Check and update device settings
    if "device" not in kwargs:
        logger.info(
            f"Device not specified, using Torch default current device - cuda:{settings.device.gpu_id}. "
            "If this is incorrect, please specify an input device, via the device keyword."
        )

    # Ignore and warn about require_full_compilation flag
    if settings.require_full_compilation:
        logger.warning(
            "Detected require_full_compilation=True for a torch.compile run. "
            "This option has no effect in torch.compile."
        )
        settings.require_full_compilation = False

    # If cache_built_engines and reuse_cached_engines are True but custom_engine_cache is not provided,
    # then create a default disk engine cache

    engine_cache = None
    if kwargs.get("cache_built_engines") or kwargs.get("reuse_cached_engines"):
        if kwargs.get("custom_engine_cache") is not None:
            engine_cache = kwargs.get("custom_engine_cache")
        else:
            from torch_tensorrt.dynamo._engine_cache import DiskEngineCache

            engine_cache_dir = kwargs.get(
                "engine_cache_dir", _defaults.ENGINE_CACHE_DIR
            )
            engine_cache_size = kwargs.get(
                "engine_cache_size", _defaults.ENGINE_CACHE_SIZE
            )
            engine_cache = DiskEngineCache(engine_cache_dir, engine_cache_size)

    if kwargs.get("torch_executed_ops"):
        settings.torch_executed_ops = kwargs.get("torch_executed_ops")

    logger.info("Compilation Settings: %s\n", settings)

    return settings, engine_cache


def req_torch_version(min_torch_version: str = "2.dev") -> Callable[..., Any]:
    """
    Create a decorator which verifies the Torch version installed
    against a specified version range

    Args:
        min_torch_version (str): The minimum required Torch version
        for the decorated function to work properly

    Returns:
        A decorator which raises a descriptive error message if
        an unsupported Torch version is used
    """

    def nested_decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        def function_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Parse minimum and current Torch versions
            min_version = version.parse(min_torch_version)
            current_version = version.parse(torch.__version__)

            if current_version < min_version:
                raise AssertionError(
                    f"Expected Torch version {min_torch_version} or greater, "
                    + f"when calling {f}. Detected version {torch.__version__}"
                )
            else:
                return f(*args, **kwargs)

        return function_wrapper

    return nested_decorator


def _copy_inputs(inputs: Any) -> Any:
    """Copy the tensors in ``inputs`` so that inputs sharing memory still share it.

    Inputs whose bytes overlap are copied once, over the bytes they use, and each
    tensor becomes a view of its copy. The copies are new tensors, so inputs with
    autograd history and Parameters copy like any other tensor.
    """

    def by_storage(t: torch.Tensor) -> bool:
        # A sparse or wrapper subclass tensor has no storage to copy, and a quantized
        # one loses its scale when rebuilt from its storage, so these are cloned.
        return (
            t.layout == torch.strided
            and not t.is_quantized
            and not is_traceable_wrapper_subclass(t)
        )

    def start(t: torch.Tensor) -> int:
        address = t.untyped_storage().data_ptr() + t.storage_offset() * t.element_size()
        return int(address)

    tensors = [
        t
        for t in torch.utils._pytree.tree_leaves(inputs)
        if isinstance(t, torch.Tensor) and t.numel() > 0 and by_storage(t)
    ]
    # Spans are addresses, not offsets, because two storages can hold the same
    # memory, as after torch.from_dlpack of a view.
    groups: List[List[Any]] = []
    for t in sorted(tensors, key=lambda t: (str(t.device), start(t))):
        lo = start(t)
        span = 1 + sum((n - 1) * s for n, s in zip(t.shape, t.stride()))
        hi = lo + span * t.element_size()
        if groups and groups[-1][0] == t.device and lo < groups[-1][2]:
            groups[-1][2] = max(groups[-1][2], hi)
            groups[-1][3].append(t)
        else:
            groups.append([t.device, lo, hi, [t]])
    # Each tensor's copy and the address its first byte stands for.
    copies: Dict[int, Tuple[torch.UntypedStorage, int]] = {}
    for device, lo, hi, members in groups:
        holders = [
            m.untyped_storage()
            for m in members
            if m.untyped_storage().data_ptr() <= lo
            and m.untyped_storage().data_ptr() + m.untyped_storage().nbytes() >= hi
        ]
        # Without a storage that holds the whole group, its tensors are cloned.
        if not holders:
            continue
        storage = holders[0]
        # Starting at a multiple of 16 bytes keeps every element size's offset whole.
        first = (lo - storage.data_ptr()) // 16 * 16
        whole = torch.empty(0, dtype=torch.uint8, device=device)
        whole.set_(storage, 0, (storage.nbytes(),), (1,))
        copy = whole[first : hi - storage.data_ptr()].clone().untyped_storage()
        for m in members:
            copies[id(m)] = (copy, storage.data_ptr() + first)

    def copy_one(t: torch.Tensor) -> torch.Tensor:
        if id(t) not in copies:
            return t.detach().clone()
        copy, first = copies[id(t)]
        view = torch.empty(0, dtype=t.dtype, device=t.device)
        view = view.set_(
            copy, (start(t) - first) // t.element_size(), t.size(), t.stride()
        )
        # The storage holds the values before a conjugate or negative view's flag.
        if t.is_conj():
            view = view.conj()
        if t.is_neg():
            view = torch._neg_view(view)
        return view

    return torch.utils._pytree.tree_map_only(torch.Tensor, copy_one, inputs)


def check_module_output(
    new_module: torch.fx.GraphModule,
    refitted_module: torch.fx.GraphModule,
    arg_inputs: Any,
    kwarg_inputs: Any = None,
) -> bool:
    # Each module gets its own copy of the inputs: a module that writes an input,
    # such as a caller's cache, must not change the caller's tensor or what the
    # other module sees.
    old_args = _copy_inputs(arg_inputs)
    new_args, new_kwargs = _copy_inputs((arg_inputs, kwarg_inputs))
    old_outputs, new_outputs = (
        refitted_module(*old_args),
        new_module(*new_args, **new_kwargs),
    )

    # A compiled module can return a tuple where the model returns one tensor or a
    # dictionary, so the outputs are compared flattened.
    # Dictionary keys must match when both outputs have them, on a leaf's path and
    # for each dictionary, also one whose entries hold no leaf. A leaf that is not a
    # tensor, such as a size, is compared as it is.
    def leaves(outputs: Any) -> List[Tuple[List[Any], Any]]:
        try:
            flat = torch.utils._pytree.tree_flatten_with_path(outputs)[0]
        except ValueError:
            # A type registered without key paths, as diffusers registers its
            # outputs, flattens without them.
            return [([], leaf) for leaf in torch.utils._pytree.tree_leaves(outputs)]
        return [
            (
                [k.key for k in path if isinstance(k, torch.utils._pytree.MappingKey)],
                leaf,
            )
            for path, leaf in flat
        ]

    def key_sets(outputs: Any) -> Dict[Any, int]:
        found: Dict[Any, int] = {}
        pending = [outputs]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                found[frozenset(value)] = found.get(frozenset(value), 0) + 1
                pending.extend(value.values())
            elif isinstance(value, (list, tuple)):
                pending.extend(value)
        return found

    def equal(old: Any, new: Any) -> bool:
        if isinstance(old, torch.Tensor) and isinstance(new, torch.Tensor):
            return old.dtype == new.dtype and check_output_equal(old, new)
        if isinstance(old, torch.Tensor) or isinstance(new, torch.Tensor):
            return False
        return type(old) == type(new) and bool(old == new)

    old_dicts, new_dicts = key_sets(old_outputs), key_sets(new_outputs)
    if old_dicts and new_dicts and old_dicts != new_dicts:
        return False
    old_leaves, new_leaves = leaves(old_outputs), leaves(new_outputs)
    return len(old_leaves) == len(new_leaves) and all(
        (not old_keys or not new_keys or old_keys == new_keys) and equal(old, new)
        for (old_keys, old), (new_keys, new) in zip(old_leaves, new_leaves)
    )


def check_output_equal(
    output1: Any,
    output2: Any,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> bool:
    if type(output1) != type(output2):
        logger.warning(
            "The output types are different. Check_output_equal will always return false."
        )
        return False

    if isinstance(output1, torch.Tensor):
        if output1.shape != output2.shape:
            return False
        return torch.allclose(output1, output2, rtol, atol)  # type: ignore

    elif isinstance(output1, (tuple, list)):
        if len(output1) != len(output2):
            return False
        for a, b in zip(output1, output2):
            if not check_output_equal(a, b):
                return False
            return True

    elif isinstance(output1, dict):
        if output1.keys() != output2.keys():
            return False
        for a, b in zip(output1.values(), output2.values()):
            if not check_output_equal(a, b):
                return False
        return True

    logger.warning(
        "The output type is not supported to be checked. Check_output_equal will always return false."
    )
    return False


def get_flat_args_with_check(
    exported_program: torch.export.ExportedProgram,
    args: list[Any],
    kwargs: dict[str, Any],
) -> tuple[Any, Any]:
    """Flatten args, kwargs using pytree, then, check specs.

    Args:
        args: List[Any] original args passed to __call__
        kwargs: Dict[str, Any] original kwargs passed to __call

    Returns:
        A tuple of (flat_args, received_spec)
        flat_args is flattend args / kwargs
        received_spec is the pytree spec produced while flattening the
        tuple (args, kwargs)
    """
    import torch.utils._pytree as pytree
    from torch.export._tree_utils import reorder_kwargs

    in_spec = exported_program.call_spec.in_spec
    if in_spec is not None:
        kwargs = reorder_kwargs(kwargs, in_spec)
    flat_args_with_path, received_spec = pytree.tree_flatten_with_path((args, kwargs))
    flat_args = tuple(x[1] for x in flat_args_with_path)
    return flat_args, received_spec


def get_metadata(
    gm: torch.fx.GraphModule, target_op: torch._ops.OpOverload
) -> List[Any]:
    """
    Return the list which has the metadata of all the target_op nodes present in the graph.
    """
    return [node.meta for node in gm.graph.nodes if node.target == target_op]


def set_metadata(
    gm: torch.fx.GraphModule, target_op: torch._ops.OpOverload, metadata: List[Any]
) -> None:
    """
    Return the list which has the metadata of all the target_op nodes present in the graph.
    """
    target_nodes = [node for node in gm.graph.nodes if node.target == target_op]
    assert len(target_nodes) == len(metadata)
    for idx, node in enumerate(target_nodes):
        node.meta = metadata[idx]


def copy_metadata(match_and_replacements: List[Any]) -> None:
    """
    Copy the metadata from anchor node to the replacement node. This should be used
    if the anchor node is replaced with only a single replacement node i.e one-one replacement.
    """
    for match_and_replacement in match_and_replacements:
        anchor_node = match_and_replacement.nodes_map[match_and_replacement.anchor]
        assert (
            len(match_and_replacement.replacements) == 1
        ), "Found more than 1 replacements for the anchor node."
        replacement_node = match_and_replacement.replacements[0]
        replacement_node.meta = anchor_node.meta


def flatten_nodes(nodes: Any) -> List[torch.fx.node.Node]:
    ret = []
    if isinstance(nodes, torch.fx.node.Node):
        ret.append(nodes)
    elif isinstance(nodes, (tuple, list)):
        for node in nodes:
            ret.extend(flatten_nodes(node))
    else:
        raise ValueError(
            f"expect torch.fx.node.Node or a tuple/list of torch.fx.node.Node type, got unexpected types: {type(nodes)=}"
        )
    return ret


def get_output_metadata(
    gm: torch.fx.GraphModule,
) -> List[Any]:
    outputs = [node for node in gm.graph.nodes if node.op == "output"]
    assert len(outputs) > 0
    outputs = outputs[0].args
    nodes = flatten_nodes(outputs)
    return [node.meta for node in nodes]


def get_output_dtypes(output: Any, truncate_double: bool = False) -> List[dtype]:
    output_dtypes = []
    if isinstance(output, torch.fx.node.Node):
        if "val" in output.meta:
            output_meta = output.meta["val"]
            if output_meta is None:
                # Placeholder output (e.g. unused slot in flash attention return tuple)
                pass
            elif isinstance(output_meta, (FakeTensor, torch.Tensor)):
                out_dtype = output_meta.dtype
                if out_dtype in COMPLEX_TO_REAL_DTYPE:
                    out_dtype = COMPLEX_TO_REAL_DTYPE[out_dtype]
                if truncate_double and out_dtype == torch.float64:
                    output_dtypes.append(dtype.float32)
                else:
                    output_dtypes.append(dtype._from(out_dtype))
            elif isinstance(output_meta, torch.SymInt):
                output_dtypes.append(dtype.int64)
        elif "tensor_meta" in output.meta:
            output_meta = output.meta["tensor_meta"]
            output_dtypes.append(dtype._from(output_meta.dtype))
        else:
            raise ValueError(
                f"node.name={output.name}: metadata does not exist, expect metadata exists for each output node"
            )
    elif isinstance(output, (tuple, list)):
        for ele in output:
            output_dtypes.extend(get_output_dtypes(ele))
    else:
        raise ValueError(
            f"got unexpected type {type(output)}, expected type is a torch.fx.node.Node or a tuple/list of torch.fx.node.Node"
        )
    return output_dtypes


def get_cpu_memory_usage() -> Any:
    return psutil.Process().memory_info().rss / 1024 / 1024


def release_host_and_device_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        # One sync so builder work has finished before empty_cache; a second
        # sync after ipc_collect does not free extra blocks.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    trim_host_heap()


@functools.lru_cache(maxsize=1)
def _glibc_malloc_trim() -> Optional[Callable[[int], int]]:
    """glibc's ``malloc_trim``, or ``None`` where the C library doesn't provide it."""
    if platform.system() != "Linux":
        return None
    try:
        malloc_trim = ctypes.CDLL("libc.so.6").malloc_trim
    except (OSError, AttributeError):
        return None
    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int
    return malloc_trim


def trim_host_heap() -> None:
    """Return freed heap memory to the operating system.

    Building a TensorRT engine frees on the order of gigabytes of host allocations that
    glibc keeps in its arenas instead of returning to the OS, so the process holds them
    for the rest of its life. On unified memory systems that memory is also unavailable
    to the GPU. ``malloc_trim`` hands it back. Set
    ``TORCHTRT_ENABLE_BUILDER_MALLOC_TRIM=0`` to skip this.
    """
    if os.environ.get("TORCHTRT_ENABLE_BUILDER_MALLOC_TRIM", "1") == "0":
        return
    malloc_trim = _glibc_malloc_trim()
    if malloc_trim is not None:
        # The return value only says whether there was anything to release.
        malloc_trim(0)


def is_quantized_by_modelopt(model: torch.nn.Module) -> bool:
    """
    Check if the model has been quantized by NVIDIA ModelOpt library.
    If ModelOpt is installed, use its function to check if the model is quantized.
    Otherwise, do name/module-string checks. A caveat is that it can break if ModelOpt renames things.
    """
    if importlib.util.find_spec("modelopt") is not None:
        from modelopt.torch.quantization.utils import is_quantized

        return bool(is_quantized(model))

    for m in model.modules():
        cls = type(m)
        if cls.__name__ == "TensorQuantizer" and cls.__module__.startswith(
            "modelopt.torch.quantization"
        ):
            return True
    return False
