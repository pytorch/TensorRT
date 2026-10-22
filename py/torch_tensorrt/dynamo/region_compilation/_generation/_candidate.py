# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Validate generated device code and reuse the existing kernel frontends."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import keyword
import logging
import math
import sys
from functools import partial
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)
_FIELDS = {"source", "kernel_name", "constants", "grid", "num_warps", "num_stages"}
_BUILTINS = {"abs", "bool", "float", "int", "len", "max", "min", "range", "tuple"}
_METHODS = {
    "astype",
    "broadcast_to",
    "expand_dims",
    "max",
    "min",
    "permute",
    "reshape",
    "split",
    "sum",
    "to",
    "trans",
}


def _identifier(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.isidentifier()
        and not keyword.iskeyword(value)
        and "__" not in value
        and value not in {"triton", "tl", "ct", "libdevice"}
    )


def _attribute(node: ast.AST | None, namespace: str, name: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == namespace
        and node.attr == name
    )


def _validate_source(
    source: str,
    kernel_name: str,
    constants: list[dict[str, Any]],
    backend: str,
    runtime_names: list[str],
) -> None:
    try:
        tree = ast.parse(source)
    except (SyntaxError, RecursionError) as error:
        raise ValueError(
            f"Generated kernel has invalid Python syntax: {error}"
        ) from error
    allowed_imports = (
        {
            ("triton", None),
            ("triton", "triton"),
            ("triton.language", "tl"),
            ("triton.language.extra.cuda.libdevice", "libdevice"),
        }
        if backend == "triton"
        else {("cuda.tile", "ct")}
    )
    imported = set()
    functions = []
    for index, node in enumerate(tree.body):
        if (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                if (alias.name, alias.asname) not in allowed_imports:
                    raise ValueError("Generated source may import only its kernel DSL")
                imported.add(alias.asname or alias.name)
        elif isinstance(node, ast.FunctionDef):
            functions.append(node)
        else:
            raise ValueError(
                "Generated source permits imports and one kernel definition only"
            )
    namespace, decorator = (
        ("triton", "jit") if backend == "triton" else ("ct", "kernel")
    )
    if len(functions) != 1 or functions[0].name != kernel_name:
        raise ValueError("Generated source must define exactly the named kernel")
    function = functions[0]
    if namespace not in imported or (
        backend == "triton" and constants and "tl" not in imported
    ):
        raise ValueError("Generated source is missing its kernel DSL imports")
    if len(function.decorator_list) != 1 or not _attribute(
        function.decorator_list[0], namespace, decorator
    ):
        raise ValueError(f"Generated kernel requires exactly @{namespace}.{decorator}")
    args = function.args
    if (
        args.posonlyargs
        or args.vararg
        or args.kwonlyargs
        or args.kwarg
        or args.defaults
        or function.returns is not None
        or getattr(function, "type_params", [])
    ):
        raise ValueError(
            "Generated kernel requires plain positional parameters without defaults"
        )
    expected_names = runtime_names + [item["name"] for item in constants]
    if [arg.arg for arg in args.args] != expected_names:
        raise ValueError(f"Generated kernel parameters must be {expected_names}")
    if any(arg.annotation is not None for arg in args.args[: len(runtime_names)]):
        raise ValueError("Generated tensor parameters must not have Python annotations")
    for argument, constant in zip(args.args[len(runtime_names) :], constants):
        annotation = argument.annotation
        valid = _attribute(annotation, "tl", "constexpr")
        if backend == "cutile":
            valid = (
                isinstance(annotation, ast.Subscript)
                and _attribute(annotation.value, "ct", "Constant")
                and isinstance(annotation.slice, ast.Name)
                and annotation.slice.id == type(constant["value"]).__name__
            )
        if not valid:
            raise ValueError(
                f"Generated constant {argument.arg!r} has an invalid annotation"
            )
    forbidden = (
        ast.AsyncFunctionDef,
        ast.Await,
        ast.ClassDef,
        ast.Global,
        ast.Import,
        ast.ImportFrom,
        ast.Lambda,
        ast.Nonlocal,
        ast.Try,
        ast.With,
        ast.Yield,
        ast.YieldFrom,
    )
    for statement in function.body:
        for inner in ast.walk(statement):
            if isinstance(inner, forbidden + (ast.FunctionDef,)):
                raise ValueError(
                    "Generated kernel must contain device computation only"
                )
            if isinstance(inner, ast.Name) and "__" in inner.id:
                raise ValueError("Generated kernel cannot access private Python names")
            if isinstance(inner, ast.Attribute) and (
                inner.attr.startswith("_") or "__" in inner.attr
            ):
                raise ValueError(
                    "Generated kernel cannot access private Python attributes"
                )
            if isinstance(inner, ast.Call):
                target = inner.func
                if isinstance(target, ast.Name) and target.id in _BUILTINS:
                    continue
                if isinstance(target, ast.Attribute):
                    if isinstance(target.value, ast.Name):
                        if _attribute(target, "tl", "inline_asm_elementwise"):
                            raise ValueError(
                                "Generated kernels cannot contain inline assembly"
                            )
                        if target.value.id in {"tl", "ct", "libdevice"}:
                            continue
                        if target.value.id == "triton" and target.attr in {
                            "cdiv",
                            "next_power_of_2",
                        }:
                            continue
                    if target.attr in _METHODS:
                        continue
                raise ValueError(
                    "Generated kernel calls must use the supported kernel DSL"
                )


def validate_candidate(
    candidate: dict[str, Any], backend: str, num_inputs: int, num_outputs: int
) -> dict[str, Any]:
    """Enforce the source and ABI contract; this is not a hostile-code sandbox."""
    if backend not in {"triton", "cutile"}:
        raise ValueError("Generated kernel backend must be 'triton' or 'cutile'")
    if any(type(count) is not int or count < 1 for count in (num_inputs, num_outputs)):
        raise ValueError("Generated kernels need at least one tensor input and output")
    if not isinstance(candidate, dict) or set(candidate) != _FIELDS:
        raise ValueError(f"Generated candidate must contain exactly {sorted(_FIELDS)}")
    source = candidate["source"]
    if not isinstance(source, str) or not source.strip() or len(source) > 131072:
        raise ValueError("Generated source must be nonempty Python text under 128 KiB")
    kernel_name = candidate["kernel_name"]
    if not _identifier(kernel_name):
        raise ValueError(
            "Generated kernel_name must be a nonreserved Python identifier"
        )
    grid = candidate["grid"]
    if (
        not isinstance(grid, list)
        or not 1 <= len(grid) <= 3
        or any(
            type(value) is not int or not 1 <= value <= limit
            for value, limit in zip(grid, (2147483647, 65535, 65535))
        )
    ):
        raise ValueError(
            "Generated grid must contain one to three valid positive CUDA dimensions"
        )
    constants = candidate["constants"]
    if not isinstance(constants, list):
        raise ValueError("Generated constants must be a list of name/value objects")
    runtime_names = [f"arg{i}" for i in range(num_inputs)] + [
        f"out{i}" for i in range(num_outputs)
    ]
    names = set(runtime_names)
    normalized_constants = []
    for item in constants:
        if (
            not isinstance(item, dict)
            or set(item) != {"name", "value"}
            or not _identifier(item["name"])
            or item["name"] in names
        ):
            raise ValueError("Generated constants need unique, nonreserved names")
        value = item["value"]
        if type(value) not in {bool, int, float} or (
            isinstance(value, float) and not math.isfinite(value)
        ):
            raise ValueError(
                "Generated constant values must be finite bool/int/float scalars"
            )
        names.add(item["name"])
        normalized_constants.append(dict(item))
    from torch_tensorrt.kernels._triton import validate_compile_options

    validate_compile_options(candidate["num_warps"], candidate["num_stages"])
    if backend == "cutile" and any(
        candidate[name] is not None for name in ("num_warps", "num_stages")
    ):
        raise ValueError("cuTile candidates must leave Triton compile options null")
    _validate_source(source, kernel_name, normalized_constants, backend, runtime_names)
    return dict(
        source=source,
        kernel_name=kernel_name,
        constants=normalized_constants,
        grid=list(grid),
        num_warps=candidate["num_warps"],
        num_stages=candidate["num_stages"],
    )


def make_registration(
    candidate: dict[str, Any],
    backend: str,
    input_specs: list[Any],
    output_specs: list[Any],
    directory: Path,
) -> partial[None]:
    """Load validated source from disk and derive trusted registration callbacks."""
    import torch

    from torch_tensorrt import kernels
    from torch_tensorrt.kernels import _triton

    candidate = validate_candidate(
        candidate, backend, len(input_specs), len(output_specs)
    )
    names = [f"arg{i}" for i in range(len(input_specs))] + [
        f"out{i}" for i in range(len(output_specs))
    ]
    dtypes = []
    for spec in input_specs + output_specs:
        dtype = spec["dtype"] if isinstance(spec, dict) else spec.dtype
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype.removeprefix("torch."), None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(
                f"Generated kernel has an unsupported boundary dtype: {dtype}"
            )
        dtypes.append(dtype)
    digest = hashlib.sha256(candidate["source"].encode()).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.py"
    path.write_text(candidate["source"], encoding="utf-8")
    _LOGGER.info("Generated %s kernel source: %s", backend, path)
    module_digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()
    module_name = f"_torch_tensorrt_generated_{module_digest}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError("Could not load generated kernel source")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    kernel = getattr(module, candidate["kernel_name"])
    dimensions = tuple(candidate["grid"])

    def grid(inputs: Any, outputs: Any) -> tuple[int, ...]:
        return dimensions

    constants = {item["name"]: item["value"] for item in candidate["constants"]}
    if backend == "cutile":
        return partial(
            kernels.cutile_op,
            kernel=kernel,
            signature=dict(zip(names, dtypes)),
            constants=constants,
            ndim=1,
            grid=grid,
        )
    dtype_names = {
        dtype: name for name, dtype in _triton._TRITON_TO_TORCH_DTYPE.items()
    }
    try:
        signature = {
            name: f"*{dtype_names[dtype]}" for name, dtype in zip(names, dtypes)
        }
    except KeyError as error:
        raise ValueError(
            f"Generated Triton kernel cannot use dtype {error.args[0]}"
        ) from error
    return partial(
        kernels.triton_op,
        kernel=kernel,
        signature=signature,
        constexprs=constants,
        grid=grid,
        num_warps=candidate["num_warps"],
        num_stages=candidate["num_stages"],
    )
