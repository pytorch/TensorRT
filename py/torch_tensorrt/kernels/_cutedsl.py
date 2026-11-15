# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile pointer-based CuTe DSL kernels for the shared QDP AOT backend."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import re
import tempfile
from pathlib import Path
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

_REQUIREMENT = "nvidia-cutlass-dsl>=4.8.1,<5"
_TESTED_SERIES = "4.8"

# Opt-in on-disk PTX cache. The key covers the kernel's defining module, so edits
# to helpers imported from other modules require clearing the directory.
CACHE_DIR_ENV = "TORCH_TENSORRT_CUTEDSL_CACHE_DIR"

# Highest PTX ISA each CUDA driver release loads, from the PTX ISA release notes.
# Releases between entries (e.g. 12.7) use the closest earlier entry; drivers
# newer than the table impose no ceiling and are still checked by a driver load.
_CUDA_TO_PTX_ISA: Tuple[Tuple[Tuple[int, int], int], ...] = (
    ((12, 0), 80),
    ((12, 1), 81),
    ((12, 2), 82),
    ((12, 3), 83),
    ((12, 4), 84),
    ((12, 5), 85),
    ((12, 6), 85),
    ((12, 8), 87),
    ((12, 9), 88),
    ((13, 0), 90),
    ((13, 1), 91),
    ((13, 2), 92),
)

# CuTe's default ISA depends only on its bundled toolchain, so once it exceeds a
# driver ceiling every later registration can compile at the ceiling directly.
_DEFAULT_ISA_EXCEEDS: Dict[int, bool] = {}


class CompiledCuteDSLArtifact(NamedTuple):
    ptx: bytes
    kernel_name: str
    block: Tuple[int, int, int]
    shared_mem: int
    arch: Tuple[int, int]


class _CuteAPI(NamedTuple):
    """CuTe DSL interfaces this frontend depends on, resolved once."""

    cutlass: Any
    cute: Any
    dsl_cls: Any
    make_ptr: Any
    DumpDir: Any
    ExtraCompilerOpts: Any
    GPUArch: Any
    KeepPTX: Any


def _api_changed(cutlass: Any, what: str) -> str:
    version = getattr(cutlass, "__version__", "unknown")
    return (
        f"The installed nvidia-cutlass-dsl {version} does not expose the {what} "
        f"cutedsl_op relies on; this frontend is tested with {_TESTED_SERIES}.x. "
        f"Install a tested release, e.g. pip install 'nvidia-cutlass-dsl~={_TESTED_SERIES}.1'."
    )


def _cutedsl_import() -> _CuteAPI:
    try:
        import cutlass
        import cutlass.cute as cute

        version = Version(cutlass.__version__)
    except (ImportError, AttributeError, InvalidVersion, TypeError) as exc:
        raise ImportError(
            f"cutedsl_op requires {_REQUIREMENT}. Install it with: pip install '{_REQUIREMENT}'"
        ) from exc
    if not Version("4.8.1") <= version < Version("5"):
        raise ImportError(f"cutedsl_op requires {_REQUIREMENT}; found {version}.")
    try:
        from cutlass.base_dsl.compiler import (
            DumpDir,
            ExtraCompilerOpts,
            GPUArch,
            KeepPTX,
        )
        from cutlass.cute.runtime import make_ptr

        dsl_cls = cute.kernel.__self__
    except (ImportError, AttributeError) as exc:
        raise ImportError(_api_changed(cutlass, "compiler interfaces")) from exc
    return _CuteAPI(
        cutlass, cute, dsl_cls, make_ptr, DumpDir, ExtraCompilerOpts, GPUArch, KeepPTX
    )


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
    api = _cutedsl_import()
    if (
        getattr(kernel, "_decorator_kind", None) != "kernel"
        or getattr(kernel, "_dsl_cls", None) is not api.dsl_cls
    ):
        raise ValueError(
            "cutedsl_op requires a non-experimental @cute.kernel function. If it "
            "is one, the installed nvidia-cutlass-dsl "
            f"{api.cutlass.__version__} changed its decorator internals; this "
            f"frontend is tested with {_TESTED_SERIES}.x."
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
        name for name in params if annotations.get(name) is api.cutlass.Constexpr
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


def make_capability_validator(
    op_name: str,
    layout: _common.SignatureLayout,
    artifact: CompiledCuteDSLArtifact,
    user_validator: Optional[Callable[..., bool]] = None,
) -> Callable[..., bool]:
    """Check dtypes, shared memory and target; declines hardware-compatible engines."""
    return _common.make_dtype_capability_validator(
        op_name,
        "CuTe DSL",
        [param.dtype for param in layout.inputs],
        [param.dtype for param in layout.outputs],
        user_validator,
        target_validator=lambda device: validate_target(artifact, device),
        shared_mem=artifact.shared_mem,
    )


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
            r"\.param\s+\.u64\s+\.ptr\s+\.global(?:\s+\.align\s+\d+)?\s+[\w$%]+"
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
    """Keep architecture-specific instructions enabled on the current device.

    The engine is already tied to this exact architecture by
    :func:`validate_target`, so the ``a`` variant costs no portability while
    enabling instructions such as Hopper ``wgmma``.
    """
    try:
        from cutlass.base_dsl import Arch
    except ImportError as exc:
        import cutlass

        raise ImportError(_api_changed(cutlass, "target architecture enum")) from exc

    base = f"sm_{arch[0]}{arch[1]}"
    target = Arch.__members__.get(base + "a") or Arch.__members__.get(base)
    if target is None:
        raise RuntimeError(f"Installed CuTe DSL does not support CUDA target {base}.")
    return str(target.to_string())


def _driver_ptx_version() -> Optional[int]:
    """PTX ISA the loaded driver accepts, or ``None`` when newer than the table."""
    from cuda.bindings import driver

    status, raw_version = driver.cuDriverGetVersion()
    version = int(raw_version)
    if int(status) != 0 or version < 12000:
        raise RuntimeError(
            "cutedsl_op requires a CUDA 12+ driver with queryable PTX support; "
            f"cuDriverGetVersion returned {status}, version {version}."
        )
    release = (version // 1000, (version % 1000) // 10)
    if release > _CUDA_TO_PTX_ISA[-1][0]:
        _LOGGER.debug(
            "CUDA driver %d.%d is newer than the known PTX ISA table; "
            "relying on the driver load check",
            *release,
        )
        return None
    return max(isa for known, isa in _CUDA_TO_PTX_ISA if known <= release)


def _ptx_version(ptx: str) -> int:
    match = re.search(r"(?m)^\s*\.version\s+(\d+)\.(\d+)\b", ptx)
    if match is None:
        raise RuntimeError("CuTe DSL PTX is missing its ISA version.")
    return int(match[1]) * 10 + int(match[2])


def _compile_within_driver_isa(compile_fn: Callable[[Optional[int]], str]) -> str:
    """Recompile newer PTX for the driver; never relabel an existing artifact."""
    ceiling = _driver_ptx_version()
    if ceiling is None:
        return compile_fn(None)
    if not _DEFAULT_ISA_EXCEEDS.get(ceiling):
        ptx = compile_fn(None)
        if _ptx_version(ptx) <= ceiling:
            return ptx
        _DEFAULT_ISA_EXCEEDS[ceiling] = True
    requirement = (
        f"driver PTX ISA {ceiling // 10}.{ceiling % 10}. "
        "Upgrade the CUDA driver or use a compatible CuTe DSL toolchain."
    )
    _LOGGER.info("Compiling CuTe DSL for driver PTX ISA %d", ceiling)
    try:
        ptx = compile_fn(ceiling)
    except Exception as exc:
        raise RuntimeError(
            f"CuTe DSL could not compile this kernel for {requirement}"
        ) from exc
    if _ptx_version(ptx) > ceiling:
        raise RuntimeError(f"CuTe DSL did not honor the {requirement}")
    return ptx


def _cache_path(
    kernel: Any,
    constants: Mapping[str, Any],
    layout: _common.SignatureLayout,
    block: Tuple[int, int, int],
    shared_mem: int,
    alignments: Mapping[str, int],
    target: str,
    cutlass_version: str,
) -> Optional[Path]:
    """Where to cache this compilation, or ``None`` when caching is off or unsafe."""
    directory = os.environ.get(CACHE_DIR_ENV)
    if not directory:
        return None
    try:
        function = inspect.unwrap(kernel)
        source_file = inspect.getsourcefile(function)
        if source_file is None:
            raise OSError("kernel has no source file")
        payload = json.dumps(
            {
                "frontend": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "cutlass": cutlass_version,
                "module": hashlib.sha256(Path(source_file).read_bytes()).hexdigest(),
                "kernel": function.__qualname__,
                "constants": dict(constants),
                "signature": [
                    (p.name, p.is_pointer, str(p.dtype))
                    for p in layout.inputs + layout.scalars + layout.outputs
                ],
                "block": block,
                "shared_mem": shared_mem,
                "alignments": dict(alignments),
                "target": target,
                "driver_ptx": _driver_ptx_version(),
            },
            sort_keys=True,
        )
    except (OSError, TypeError, ValueError, AttributeError) as exc:
        _LOGGER.debug("Not caching CuTe DSL PTX: %s", exc)
        return None
    return Path(directory) / f"{hashlib.sha256(payload.encode()).hexdigest()}.ptx"


def _read_cache(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _write_cache(path: Optional[Path], ptx: str) -> None:
    if path is None:
        return
    temporary: Optional[str] = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, suffix=".tmp", delete=False, encoding="utf-8"
        ) as handle:
            temporary = handle.name
            handle.write(ptx)
        os.replace(temporary, path)
    except OSError as exc:
        _LOGGER.debug("Could not write CuTe DSL PTX cache %s: %s", path, exc)
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def compile_to_ptx(
    kernel: Any,
    constants: Mapping[str, Any],
    layout: _common.SignatureLayout,
    block: Tuple[int, int, int],
    shared_mem: int,
    alignments: Optional[Mapping[str, int]] = None,
) -> CompiledCuteDSLArtifact:
    api = _cutedsl_import()
    cutlass, cute = api.cutlass, api.cute

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
    alignments = dict(alignments or {})
    params = layout.inputs + layout.scalars + layout.outputs
    runtime_names = tuple(param.name for param in params)
    constants = dict(constants)
    arch = torch.cuda.get_device_capability()
    target = _target_arch(arch)
    cache = _cache_path(
        kernel,
        constants,
        layout,
        block,
        shared_mem,
        alignments,
        target,
        str(cutlass.__version__),
    )
    cached = _read_cache(cache)
    fresh = cached is None
    if cached is not None:
        _LOGGER.debug("Loaded CuTe DSL PTX from cache %s", cache)
        ptx = cached
    else:

        @cute.jit  # type: ignore[misc]
        def launch(grid_x: Any, grid_y: Any, grid_z: Any, *args: Any) -> None:
            kernel(**dict(zip(runtime_names, args)), **constants).launch(
                grid=(grid_x, grid_y, grid_z), block=block, smem=shared_mem
            )

        args = [
            (
                api.make_ptr(
                    getattr(cutlass, type_names[param.dtype]),
                    0,
                    cute.AddressSpace.gmem,
                    assumed_align=alignments.get(param.name, 1),
                )
                if param.is_pointer
                else cutlass.Int32(1)
            )
            for param in params
        ]
        with tempfile.TemporaryDirectory(prefix="torch_tensorrt_cutedsl_") as directory:
            options: Tuple[Any, ...] = (
                api.GPUArch(target),
                api.KeepPTX(),
                api.DumpDir(directory),
            )

            def compile(ptx_version: Optional[int]) -> str:
                compile_options = options
                if ptx_version is not None:
                    compile_options += (
                        api.ExtraCompilerOpts(f"ptx-version=+ptx{ptx_version}"),
                    )
                # Host grid placeholders stay dynamic; TensorRT supplies the actual grid.
                grid = [cutlass.Int32(1) for _ in range(3)]
                compiled = cute.compile[compile_options](launch, *grid, *args)
                missing = object()
                added = getattr(compiled, "total_added_arguments", missing)
                extra = getattr(compiled, "kernel_extra_args", missing)
                text = getattr(compiled, "__ptx__", None)
                if added is missing or extra is missing or not isinstance(text, str):
                    raise RuntimeError(
                        _api_changed(cutlass, "compiled-kernel metadata")
                    )
                if added or extra:
                    raise RuntimeError(
                        "cutedsl_op cannot supply compiler-added workspace arguments."
                    )
                return text

            ptx = _compile_within_driver_isa(compile)
    name = _validate_ptx(ptx, layout, block, shared_mem)
    error = _common.load_ptx(ptx, name)
    if int(error) != 0:
        raise RuntimeError(
            f"The CUDA driver could not load CuTe DSL PTX for '{name}': {error}. "
            "Use a CUDA driver and CuTe DSL toolchain that support this kernel's "
            "PTX ISA and target architecture."
        )
    if fresh:
        _write_cache(cache, ptx)
    return CompiledCuteDSLArtifact(ptx.encode(), name, block, shared_mem, arch)


def validate_target(artifact: CompiledCuteDSLArtifact, device: Any = None) -> None:
    actual = torch.cuda.get_device_capability(device)
    if actual != artifact.arch:
        raise RuntimeError(
            f"CuTe DSL kernel '{artifact.kernel_name}' was compiled for {artifact.arch}, "
            f"but the TensorRT build targets {actual}. Register on the target CUDA device."
        )
