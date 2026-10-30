# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile pointer-based CuTe DSL kernels for the shared QDP AOT backend."""

from __future__ import annotations

import inspect
import logging
import re
import tempfile
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    get_type_hints,
)

import torch

from packaging.version import InvalidVersion, Version
from torch_tensorrt.kernels import _common

_LOGGER = logging.getLogger(__name__)


class CompiledCuteDSLArtifact(NamedTuple):
    ptx: bytes
    kernel_name: str
    block: Tuple[int, int, int]
    shared_mem: int
    arch: Tuple[int, int]


def _cutedsl_import() -> Tuple[Any, Any]:
    requirement = "nvidia-cutlass-dsl>=4.8.1,<5"
    try:
        import cutlass
        import cutlass.cute as cute

        version = Version(cutlass.__version__)
    except (ImportError, AttributeError, InvalidVersion, TypeError) as exc:
        raise ImportError(
            f"cutedsl_op requires {requirement}. Install it with: pip install '{requirement}'"
        ) from exc
    if not Version("4.8.1") <= version < Version("5"):
        raise ImportError(f"cutedsl_op requires {requirement}; found {version}.")
    return cutlass, cute


def validate_config(
    op_name: str,
    kernel: Any,
    signature: Dict[str, str],
    constants: Mapping[str, Any],
    arity: Tuple[int, int],
    extra_args_fn: Optional[Callable[..., Any]],
) -> _common.SignatureLayout:
    layout = _common.analyze_signature(signature, arity, frontend="cutedsl_op")
    for param in layout.inputs + layout.outputs:
        _common.to_trt_dtype(param.dtype)
    cutlass, cute = _cutedsl_import()
    if (
        getattr(kernel, "_decorator_kind", None) != "kernel"
        or getattr(kernel, "_dsl_cls", None) is not cute.kernel.__self__
    ):
        raise ValueError(
            "cutedsl_op requires a non-experimental @cute.kernel function."
        )
    try:
        params = inspect.signature(kernel).parameters
        annotations = get_type_hints(inspect.unwrap(kernel))
    except (NameError, TypeError, ValueError) as exc:
        raise ValueError(
            f"cutedsl_op '{op_name}' requires inspectable kernel parameters and "
            "resolvable annotations. Import annotation types in the kernel's module."
        ) from exc
    constexpr_names = {
        name for name in params if annotations.get(name) is cutlass.Constexpr
    }
    runtime_names = [name for name in params if name not in constexpr_names]
    if (
        list(signature) != runtime_names
        or not isinstance(constants, Mapping)
        or set(constants) != constexpr_names
        or any(
            param.kind
            not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            for param in params.values()
        )
    ):
        raise ValueError(
            "cutedsl_op signature must match the kernel's runtime parameters in "
            "declaration order, and constants must supply every cutlass.Constexpr parameter."
        )
    if extra_args_fn is not None and not callable(extra_args_fn):
        raise ValueError(f"cutedsl_op '{op_name}' extra_args_fn must be callable.")
    if bool(layout.scalars) != (extra_args_fn is not None):
        raise ValueError(
            "cutedsl_op requires extra_args_fn exactly when the signature declares i32 extras."
        )
    return layout


def _validate_ptx(
    ptx: str,
    layout: _common.SignatureLayout,
    block: Tuple[int, int, int],
    shared_mem: int,
) -> str:
    entry, name, params = _common.parse_entry(ptx)
    expected = layout.inputs + layout.scalars + layout.outputs
    if entry is None or len(params) != len(expected):
        raise RuntimeError(
            "CuTe DSL must emit one PTX entry with exactly the declared arguments."
        )
    for param, declared in zip(params, expected):
        pattern = (
            r"\.param\s+\.u64\s+\.ptr\s+\.global\s+\.align\s+1\s+[\w$%]+"
            if declared.is_pointer
            else r"\.param\s+\.[usb]32\s+[\w$%]+"
        )
        if re.fullmatch(pattern, param) is None:
            raise RuntimeError(
                f"CuTe DSL parameter '{declared.name}' has unsupported PTX ABI: {param}."
            )
    if _common.parse_reqntid(ptx) != block:
        raise RuntimeError("CuTe DSL PTX thread dimensions do not match block_size.")
    uncommented = _common.mask_ptx_comments(ptx)
    if re.search(
        r"\.(?:explicitcluster|reqnctapercluster|maxclusterrank)\b", uncommented
    ):
        raise RuntimeError("cutedsl_op does not support clustered kernel launches.")
    if shared_mem == 0 and re.search(r"\.extern\s+\.shared\b", uncommented):
        raise ValueError(
            "This CuTe DSL kernel requires dynamic shared memory; "
            "set shared_mem in bytes."
        )
    return name


def _target_arch(arch: Tuple[int, int]) -> str:
    """Keep architecture-specific instructions enabled on the current device."""
    from cutlass.base_dsl import Arch

    base = f"sm_{arch[0]}{arch[1]}"
    target = Arch.__members__.get(base + "a") or Arch.__members__.get(base)
    if target is None:
        raise RuntimeError(f"Installed CuTe DSL does not support CUDA target {base}.")
    return str(target.to_string())


def _driver_ptx_version() -> int:
    """PTX ISA supported by the loaded driver, independent of the CUDA toolkit."""
    from cuda.bindings import driver

    status, raw_version = driver.cuDriverGetVersion()
    version = int(raw_version)
    if int(status) != 0 or version < 12000:
        raise RuntimeError(
            "cutedsl_op requires a CUDA 12+ driver with queryable PTX support; "
            f"cuDriverGetVersion returned {status}, version {version}."
        )
    major, minor = version // 1000, (version % 1000) // 10
    # CUDA 12.6 still uses PTX 8.5; CUDA 12.8/12.9 use 8.7/8.8.
    if major == 12:
        return 80 + minor - (minor >= 6)
    return 90 + (major - 13) * 10 + minor


def _ptx_version(ptx: str) -> int:
    match = re.search(r"(?m)^\s*\.version\s+(\d+)\.(\d+)\b", ptx)
    if match is None:
        raise RuntimeError("CuTe DSL PTX is missing its ISA version.")
    return int(match[1]) * 10 + int(match[2])


def _compile_within_driver_isa(compile_fn: Callable[[Optional[int]], str]) -> str:
    """Recompile newer PTX for the driver; never relabel an existing artifact."""
    ceiling = _driver_ptx_version()
    ptx = compile_fn(None)
    if _ptx_version(ptx) <= ceiling:
        return ptx
    requirement = (
        f"driver PTX ISA {ceiling // 10}.{ceiling % 10}. "
        "Upgrade the CUDA driver or use a compatible CuTe DSL toolchain."
    )
    _LOGGER.info("Recompiling CuTe DSL for driver PTX ISA %d", ceiling)
    try:
        ptx = compile_fn(ceiling)
    except Exception as exc:
        raise RuntimeError(
            f"CuTe DSL could not compile this kernel for {requirement}"
        ) from exc
    if _ptx_version(ptx) > ceiling:
        raise RuntimeError(f"CuTe DSL did not honor the {requirement}")
    return ptx


def compile_to_ptx(
    kernel: Any,
    constants: Mapping[str, Any],
    layout: _common.SignatureLayout,
    block: Tuple[int, int, int],
    shared_mem: int,
) -> CompiledCuteDSLArtifact:
    cutlass, cute = _cutedsl_import()
    from cutlass.base_dsl.compiler import DumpDir, ExtraCompilerOpts, GPUArch, KeepPTX
    from cutlass.cute.runtime import make_ptr

    type_names = {
        torch.float16: "Float16",
        torch.bfloat16: "BFloat16",
        torch.float32: "Float32",
        torch.bool: "Boolean",
        torch.int8: "Int8",
        torch.int32: "Int32",
        torch.int64: "Int64",
        torch.uint8: "Uint8",
        torch.float8_e4m3fn: "Float8E4M3FN",
    }
    params = layout.inputs + layout.scalars + layout.outputs
    runtime_names = tuple(param.name for param in params)
    constants = dict(constants)

    @cute.jit  # type: ignore[misc]
    def launch(grid_x: Any, grid_y: Any, grid_z: Any, *args: Any) -> None:
        kernel(**dict(zip(runtime_names, args)), **constants).launch(
            grid=(grid_x, grid_y, grid_z), block=block, smem=shared_mem
        )

    args = [
        (
            make_ptr(
                getattr(cutlass, type_names[param.dtype]),
                0,
                cute.AddressSpace.gmem,
                assumed_align=1,
            )
            if param.is_pointer
            else cutlass.Int32(1)
        )
        for param in params
    ]
    arch = torch.cuda.get_device_capability()
    with tempfile.TemporaryDirectory(prefix="torch_tensorrt_cutedsl_") as directory:
        options: Tuple[Any, ...] = (
            GPUArch(_target_arch(arch)),
            KeepPTX(),
            DumpDir(directory),
        )

        def compile(ptx_version: Optional[int]) -> str:
            compile_options = options
            if ptx_version is not None:
                compile_options += (
                    ExtraCompilerOpts(f"ptx-version=+ptx{ptx_version}"),
                )
            # Host grid placeholders stay dynamic; TensorRT supplies the actual grid.
            grid = [cutlass.Int32(1) for _ in range(3)]
            compiled = cute.compile[compile_options](launch, *grid, *args)
            if compiled.total_added_arguments or compiled.kernel_extra_args:
                raise RuntimeError(
                    "cutedsl_op cannot supply compiler-added workspace arguments."
                )
            ptx = compiled.__ptx__
            if not isinstance(ptx, str):
                raise RuntimeError("CuTe DSL did not expose the compiled PTX text.")
            return ptx

        ptx = _compile_within_driver_isa(compile)
    name = _validate_ptx(ptx, layout, block, shared_mem)
    error = _common.load_ptx(ptx, name)
    if int(error) != 0:
        raise RuntimeError(
            f"The CUDA driver could not load CuTe DSL PTX for '{name}': {error}. "
            "Use a CUDA driver and CuTe DSL toolchain that support this kernel's "
            "PTX ISA and target architecture."
        )
    return CompiledCuteDSLArtifact(ptx.encode(), name, block, shared_mem, arch)


def validate_target(artifact: CompiledCuteDSLArtifact, device: Any = None) -> None:
    actual = torch.cuda.get_device_capability(device)
    if actual != artifact.arch:
        raise RuntimeError(
            f"CuTe DSL kernel '{artifact.kernel_name}' was compiled for {artifact.arch}, "
            f"but the TensorRT build targets {actual}. Register on the target CUDA device."
        )
