# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Signature, tensor metadata, launch, and PTX helpers shared by kernel frontends."""

from __future__ import annotations

import logging
import re
from math import prod
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

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
    *,
    target_validator: Optional[Callable[[Any], None]] = None,
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
        if target_validator is not None:
            device = getattr(getattr(settings, "device", None), "gpu_id", None)
            target_validator(device if device is not None and device >= 0 else None)
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


# Only accept pointer dtypes covered by the TensorRT ABI checks.
_POINTER_TO_TORCH_DTYPE: Dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "i1": torch.bool,
    "i8": torch.int8,
    "i32": torch.int32,
    "i64": torch.int64,
    "u8": torch.uint8,
    "fp8e4nv": torch.float8_e4m3fn,
}


class SignatureParam(NamedTuple):
    """One runtime pointer or scalar parameter, decoded from a signature."""

    name: str
    is_pointer: bool
    dtype: torch.dtype


class SignatureLayout(NamedTuple):
    """A ``signature`` split along the ``(inputs, extras, outputs)`` convention."""

    inputs: Tuple[SignatureParam, ...]
    scalars: Tuple[SignatureParam, ...]
    outputs: Tuple[SignatureParam, ...]


def parse_signature(
    signature: Dict[str, str], *, frontend: str = "triton_op"
) -> List[SignatureParam]:
    """Decode ``*<dtype>`` pointers and ``i32`` extras in declaration order."""
    if not isinstance(signature, dict):
        raise ValueError(
            f"{frontend} signature must be an insertion-ordered dict mapping "
            "parameter names to pointer or scalar type strings."
        )

    params: List[SignatureParam] = []
    for name, type_str in signature.items():
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"{frontend} signature keys must be non-empty strings; got {name!r}."
            )
        if not isinstance(type_str, str):
            raise ValueError(
                f"{frontend} signature parameter '{name}' must use a string type; "
                f"got {type(type_str).__name__}."
            )
        text = type_str.strip()
        if text.startswith("*"):
            match = re.fullmatch(r"\*([A-Za-z0-9_]+)", text)
            if match is None:
                raise ValueError(
                    f"{frontend} signature parameter '{name}' has malformed pointer "
                    f"type {type_str!r}; expected exactly '*<dtype>'."
                )
            element = match.group(1)
            dtype = _POINTER_TO_TORCH_DTYPE.get(element)
            if dtype is None:
                supported = ", ".join(sorted(_POINTER_TO_TORCH_DTYPE))
                raise ValueError(
                    f"{frontend} signature parameter '{name}' has unsupported pointer "
                    f"element type '{element}'. TensorRT-tested types: {supported}."
                )
            params.append(SignatureParam(name, True, dtype))
            continue

        if text != "i32":
            raise ValueError(
                f"{frontend} signature parameter '{name}' has unsupported scalar "
                f"type '{text}'; TensorRT AOT extras require trtp.SymInt32 (i32)."
            )
        params.append(SignatureParam(name, False, torch.int32))
    return params


def analyze_signature(
    signature: Dict[str, str],
    arity: Tuple[int, int],
    *,
    frontend: str = "triton_op",
) -> SignatureLayout:
    """Split runtime parameters using the Torch schema's input/output counts.

    TensorRT passes ``(input_ptrs..., extra_scalars..., output_ptrs...)``.
    Reject a different layout before it can bind scalars as device pointers.
    """
    if (
        not isinstance(arity, tuple)
        or len(arity) != 2
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in arity
        )
    ):
        raise ValueError(
            f"{frontend} tensor arity must be a pair of non-negative integers."
        )

    params = parse_signature(signature, frontend=frontend)
    if not params:
        raise ValueError(
            f"{frontend} signature is empty; it must declare the kernel's "
            "non-constexpr parameters in declaration order."
        )

    def convention() -> str:
        order = ", ".join(
            f"{p.name}={'ptr' if p.is_pointer else 'scalar'}" for p in params
        )
        return (
            f"Expected (input_ptrs..., extra_scalars..., output_ptrs...); got {order}."
        )

    if not params[0].is_pointer or not params[-1].is_pointer:
        raise ValueError(
            f"{frontend} signature must begin and end with pointer parameters. "
            f"{convention()}"
        )

    num_inputs, num_outputs = arity
    output_start = len(params) - num_outputs
    layout = SignatureLayout(
        inputs=tuple(params[:num_inputs]),
        scalars=tuple(params[num_inputs:output_start]),
        outputs=tuple(params[output_start:]),
    )
    if (
        output_start < num_inputs
        or not all(p.is_pointer for p in layout.inputs + layout.outputs)
        or any(p.is_pointer for p in layout.scalars)
    ):
        raise ValueError(
            f"{frontend} signature must match {num_inputs} tensor input(s), "
            f"followed by scalar extras and {num_outputs} tensor output(s). "
            f"{convention()}"
        )
    return layout


def make_symint32_args(
    op_name: str,
    scalar_params: Tuple[SignatureParam, ...],
    values: Optional[Iterable[Any]],
    *,
    frontend: str = "triton_op",
) -> Any:
    """Validate and materialize the exact AOT scalar argument list."""
    import tensorrt.plugin as trtp

    try:
        materialized = [] if values is None else list(values)
    except TypeError as exc:
        raise TypeError(
            f"{frontend} '{op_name}' extra_args_fn must return an iterable of "
            "scalar values."
        ) from exc
    if len(materialized) != len(scalar_params):
        names = [param.name for param in scalar_params]
        raise ValueError(
            f"{frontend} '{op_name}' extra_args_fn returned {len(materialized)} "
            f"value(s), but the signature declares {len(scalar_params)} scalar "
            f"parameter(s) {names}."
        )

    for index, (param, value) in enumerate(zip(scalar_params, materialized)):
        if isinstance(value, bool) or not isinstance(value, (int, trtp.SymInt32)):
            raise TypeError(
                f"{frontend} '{op_name}' extra_args_fn value {index} for "
                f"'{param.name}' must be an int or trtp.SymInt32; got "
                f"{type(value).__name__}."
            )
        constant = constant_int(value)
        if constant is not None and not -(2**31) <= constant < 2**31:
            raise ValueError(
                f"{frontend} '{op_name}' extra_args_fn value {index} for "
                f"'{param.name}' is outside the signed i32 range: {constant}."
            )
    return pack_symint32_args(materialized, trtp)


def validate_launch_grid(
    op_name: str,
    value: Any,
    *,
    frontend: str = "triton_op",
    trtp: Any = None,
) -> Tuple[Any, ...]:
    """Validate and preserve one to three concrete or symbolic launch dimensions."""
    if trtp is None:
        import tensorrt.plugin as plugin

        trtp = plugin
    import tensorrt as trt

    dims = tuple(value) if isinstance(value, (tuple, list)) else (value,)
    if not 1 <= len(dims) <= 3:
        raise ValueError(
            f"{frontend} '{op_name}' grid returned {len(dims)} dimension(s); "
            "TensorRT launches accept 1 to 3 (grid_x, grid_y, grid_z)."
        )

    symint32 = getattr(trtp, "SymInt32", None)
    symbolic_types = tuple(
        cls
        for cls in (
            symint32,
            getattr(symint32, "__base__", None),
            getattr(trtp, "ShapeExpr", None),
            getattr(trt, "IDimensionExpr", None),
        )
        if isinstance(cls, type) and cls is not object
    )
    for index, dim in enumerate(dims):
        if isinstance(dim, bool) or not isinstance(dim, (int, *symbolic_types)):
            raise TypeError(
                f"{frontend} '{op_name}' grid dimension {index} must be an int or "
                f"TensorRT symbolic integer; got {type(dim).__name__}."
            )
        constant = constant_int(dim)
        limit = 2**31 - 1 if index == 0 else 65535
        if constant is not None and not 1 <= constant <= limit:
            raise ValueError(
                f"{frontend} '{op_name}' grid dimension {index} must be between "
                f"1 and {limit}; got {constant}."
            )
    return dims


def make_pointer_aot_fn(
    op_name: str,
    layout: SignatureLayout,
    grid: Callable[..., Any],
    block: Tuple[int, int, int],
    shared_mem: int,
    extra_args_fn: Optional[Callable[..., Any]],
    check_target: Callable[[], None],
    *,
    frontend: str,
) -> Callable[..., Any]:
    """Build a checked launch for the pointer / i32 / pointer AOT convention."""
    import tensorrt.plugin as trtp

    def aot_fn(inputs: Any, outputs: Any, _tactic: int) -> Any:
        check_target()
        dims = validate_launch_grid(op_name, grid(inputs, outputs), frontend=frontend)
        dims += (1,) * (3 - len(dims))
        launch = trtp.KernelLaunchParams(
            grid_x=dims[0],
            grid_y=dims[1],
            grid_z=dims[2],
            block_x=block[0],
            block_y=block[1],
            block_z=block[2],
            shared_mem=shared_mem,
        )
        values = [] if extra_args_fn is None else extra_args_fn(inputs, outputs)
        extras = make_symint32_args(op_name, layout.scalars, values, frontend=frontend)
        return launch, extras

    return check_aot_dtypes(
        op_name,
        aot_fn,
        [param.dtype for param in layout.inputs],
        [param.dtype for param in layout.outputs],
    )


def normalize_block_dims(
    op_name: str, block_size: Any, *, frontend: str = "cutile_op"
) -> Tuple[int, int, int]:
    """Validate a static CUDA block size and normalize it to three dimensions."""
    if isinstance(block_size, Sequence) and not isinstance(
        block_size, (str, bytes, bytearray)
    ):
        dims = tuple(block_size)
    else:
        dims = (block_size,)

    if not 1 <= len(dims) <= 3 or any(
        isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in dims
    ):
        raise ValueError(
            f"{frontend} '{op_name}' block_size must contain 1 to 3 positive "
            f"integers; got {block_size!r}."
        )
    padded = tuple(dims) + (1,) * (3 - len(dims))
    limits = (1024, 1024, 64)
    if any(dim > limit for dim, limit in zip(padded, limits)) or prod(padded) > 1024:
        raise ValueError(
            f"{frontend} '{op_name}' block_size {block_size!r} exceeds CUDA's "
            "per-dimension or 1024 threads-per-block limit."
        )
    return int(padded[0]), int(padded[1]), int(padded[2])


_ENTRY_RE = re.compile(
    r"((?:(?:\.(?:visible|weak))\s+)?\.entry\s+([\w$%]+)\s*\()([^)]*)(\))",
    re.DOTALL,
)
_PTX_INT = r"(?:0[xX][0-9a-fA-F]+|0[bB][01]+|0[0-7]*|[1-9]\d*)"
_REQNTID_RE = re.compile(
    rf"\.reqntid\s+({_PTX_INT})(?:\s*,\s*({_PTX_INT}))?"
    rf"(?:\s*,\s*({_PTX_INT}))?(?![A-Za-z0-9_])"
)
_PTX_COMMENT_RE = re.compile(r"//[^\r\n]*|/\*.*?\*/", re.DOTALL)


ParsedEntry = Tuple[Optional["re.Match[str]"], str, List[str]]


def mask_ptx_comments(ptx: str) -> str:
    """Hide comments without changing offsets used to rewrite the source."""
    return _PTX_COMMENT_RE.sub(lambda match: " " * len(match.group()), ptx)


def parse_entry(ptx: str) -> ParsedEntry:
    """``(match, kernel name, params)`` for one unambiguous PTX entry."""
    matches = list(_ENTRY_RE.finditer(mask_ptx_comments(ptx)))
    if len(matches) != 1:
        return None, "", []
    match = matches[0]
    params = [p.strip() for p in match.group(3).split(",") if p.strip()]
    return match, match.group(2), params


def parse_reqntid(ptx: str) -> Optional[Tuple[int, int, int]]:
    """Read the mandatory CUDA thread block dimensions from a single PTX entry."""
    uncommented = mask_ptx_comments(ptx)
    entry, _, _ = parse_entry(uncommented)
    if entry is None:
        return None
    body_start = uncommented.find("{", entry.end())
    if body_start < 0:
        return None
    matches = list(_REQNTID_RE.finditer(uncommented, entry.end(), body_start))
    if len(matches) != 1:
        return None
    match = matches[0]
    x, y, z = match.groups()

    def _ptx_int(value: Optional[str]) -> int:
        text = value or "1"
        if text.lower().startswith("0x"):
            base = 16
        elif text.lower().startswith("0b"):
            base = 2
        elif len(text) > 1 and text.startswith("0"):
            base = 8
        else:
            base = 10
        return int(text, base)

    return _ptx_int(x), _ptx_int(y), _ptx_int(z)


def load_ptx(ptx: str, kernel_name: str) -> Any:
    """Ask the driver to JIT the module and resolve its expected entry."""
    try:
        from cuda.bindings import driver as cuda
    except ImportError:  # cuda-python < 12.8
        from cuda import cuda

    # cuModuleLoadData needs a current context. Asking PyTorch for its current
    # stream establishes the primary context without allocating a tensor.
    torch.cuda.init()
    torch.cuda.current_stream()

    err, module = cuda.cuModuleLoadData(ptx.encode("utf-8"))
    if err != cuda.CUresult.CUDA_SUCCESS:
        return err
    try:
        err, _function = cuda.cuModuleGetFunction(module, kernel_name.encode())
    finally:
        cuda.cuModuleUnload(module)
    return err
