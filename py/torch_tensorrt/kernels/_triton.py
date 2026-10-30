# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools
import inspect
import logging
import operator
import re
from collections.abc import Mapping
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

import torch

from packaging.version import InvalidVersion, Version
from torch_tensorrt.kernels import _common

_LOGGER = logging.getLogger(__name__)

# Triton 3.5 is the first release whose NVIDIA compiler metadata contains the
# complete launch ABI that ``_validate_launch_metadata`` checks, including
# ``profile_scratch_size``. With an older release, an absent field cannot tell
# us that scratch is zero; it only tells us that Triton did not report it, so
# fail before compilation instead of weakening the launch validation.
_MIN_TRITON_VERSION = Version("3.5.0")


class CompiledTritonArtifact(NamedTuple):
    """Immutable compiler output; PTX is retained exactly as emitted by Triton."""

    ptx: bytes
    kernel_name: str
    num_warps: int
    shared_mem: int
    target: Any
    ptx_version: Optional[int]


def validate_compile_options(
    num_warps: Optional[int], num_stages: Optional[int]
) -> None:
    for name, value in (("num_warps", num_warps), ("num_stages", num_stages)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError(
                f"{name} must be a positive integer or None; got {value!r}."
            )
    if num_warps is not None and (num_warps > 32 or num_warps & (num_warps - 1)):
        raise ValueError("num_warps must be a power of two between 1 and 32.")


def _device_arch(device: Any = None) -> int:
    major, minor = torch.cuda.get_device_capability(device)
    return int(major * 10 + minor)


def validate_target(artifact: CompiledTritonArtifact, device: Any = None) -> None:
    arch = _device_arch(device)
    if arch != artifact.target.arch:
        raise RuntimeError(
            f"Triton kernel '{artifact.kernel_name}' was compiled for sm_{artifact.target.arch}, "
            f"but the TensorRT build targets sm_{arch}. Register a separate operator "
            "inside torch.cuda.device(target_device) for each GPU architecture."
        )


SignatureParam = _common.SignatureParam

SignatureLayout = _common.SignatureLayout


def _validate_kernel_parameter_order(
    kernel: Any,
    signature: Dict[str, str],
    constexprs: Dict[str, Any],
) -> None:
    """Require ``signature`` to match the Triton declaration's runtime order.

    ``ASTSource`` resolves named signature entries against ``kernel.arg_names``.
    Validating only dictionary insertion order can therefore approve one ABI
    while Triton compiles another. Inspect the JIT function itself and require
    an exact partition between runtime and constexpr parameters.
    """
    if not isinstance(constexprs, dict):
        raise ValueError("triton_op constexprs must be a dict keyed by parameter name.")
    invalid_constexpr_names = [
        name for name in constexprs if not isinstance(name, str) or not name
    ]
    if invalid_constexpr_names:
        raise ValueError(
            "triton_op constexpr keys must be non-empty parameter names; got "
            f"{invalid_constexpr_names!r}."
        )

    arg_names = getattr(kernel, "arg_names", None)
    params = getattr(kernel, "params", None)
    if not isinstance(arg_names, (list, tuple)) or not isinstance(
        params, (list, tuple)
    ):
        raise ValueError(
            "triton_op requires a supported @triton.jit function exposing complete "
            "arg_names and KernelParam metadata; refusing to infer an unchecked ABI."
        )
    arg_names = list(arg_names)
    params = list(params)

    if (
        not arg_names
        or not all(isinstance(name, str) and name for name in arg_names)
        or len(set(arg_names)) != len(arg_names)
    ):
        raise ValueError(
            "triton_op kernel has empty, non-string, or duplicate declared "
            "parameter names; pass a supported @triton.jit function."
        )

    if len(params) != len(arg_names):
        raise ValueError(
            "triton_op kernel exposes incomplete KernelParam metadata; refusing "
            "to infer constexpr or runtime parameters."
        )
    if any(
        not hasattr(param, "name") or not hasattr(param, "is_constexpr")
        for param in params
    ):
        raise ValueError(
            "triton_op kernel exposes incomplete KernelParam metadata; refusing "
            "to infer constexpr or runtime parameters."
        )
    param_names = [param.name for param in params]
    constexpr_flags = [param.is_constexpr for param in params]
    if param_names != arg_names or not all(
        isinstance(flag, bool) for flag in constexpr_flags
    ):
        raise ValueError(
            "triton_op kernel exposes inconsistent declaration metadata: "
            f"arg_names={arg_names}, KernelParam names={param_names}, and every "
            "is_constexpr marker must be bool."
        )

    declared_constexprs = {
        param.name
        for param, is_constexpr in zip(params, constexpr_flags)
        if is_constexpr
    }
    supplied_constexprs = set(constexprs)
    if supplied_constexprs != declared_constexprs:
        missing = sorted(declared_constexprs - supplied_constexprs)
        unknown = sorted(supplied_constexprs - declared_constexprs)
        details = []
        if missing:
            details.append(f"missing constexpr values for {missing}")
        if unknown:
            details.append(f"non-constexpr names supplied in constexprs: {unknown}")
        raise ValueError(
            "triton_op constexprs do not match the kernel declaration: "
            + "; ".join(details)
            + "."
        )

    declared_runtime = [name for name in arg_names if name not in declared_constexprs]
    supplied_runtime = list(signature)
    if supplied_runtime != declared_runtime:
        raise ValueError(
            "triton_op signature parameter order must exactly match the "
            f"kernel declaration after removing constexprs; declared runtime "
            f"parameters are {declared_runtime}, signature keys are "
            f"{supplied_runtime}."
        )


analyze_signature = _common.analyze_signature


def validate_triton_config(
    op_name: str,
    kernel: Any,
    signature: Dict[str, str],
    constexprs: Dict[str, Any],
    arity: Tuple[int, int],
    extra_args_fn: Optional[Callable[..., Any]],
) -> SignatureLayout:
    """Check a ``triton_op`` registration and return its signature layout.

    These checks bind the signature to the Torch schema and Triton declaration.
    Left unchecked, ABI mistakes yield wrong numbers rather than a useful error,
    so registration rejects them before compiling anything.
    """
    if extra_args_fn is not None and not callable(extra_args_fn):
        raise ValueError(
            f"triton_op '{op_name}' extra_args_fn must be callable or None."
        )

    layout = analyze_signature(signature, arity)
    _validate_kernel_parameter_order(kernel, signature, constexprs)

    if layout.scalars and extra_args_fn is None:
        names = ", ".join(p.name for p in layout.scalars)
        raise ValueError(
            f"triton_op '{op_name}' signature declares scalar parameter(s) "
            f"({names}) but no extra_args_fn was given, so TensorRT would launch "
            "the kernel with no extra arguments and those scalars would read as "
            "zero. Pass extra_args_fn=lambda inputs, outputs: [...] returning "
            "one trtp.SymInt32 per scalar."
        )
    if not layout.scalars and extra_args_fn is not None:
        raise ValueError(
            f"triton_op '{op_name}' was given an extra_args_fn but its signature "
            "declares no scalar parameters to receive the values. Add the "
            "scalars to the signature or drop extra_args_fn."
        )
    return layout


make_symint32_args = _common.make_symint32_args


validate_launch_grid = _common.validate_launch_grid


def make_dtype_capability_validator(
    op_name: str,
    layout: SignatureLayout,
    user_validator: Optional[Callable[..., bool]] = None,
    *,
    artifact: Optional[CompiledTritonArtifact] = None,
) -> Callable[..., bool]:
    """Check tensor dtypes before enforcing Triton's compiled device target."""
    return _common.make_dtype_capability_validator(
        op_name,
        "Triton",
        [param.dtype for param in layout.inputs],
        [param.dtype for param in layout.outputs],
        user_validator,
        target_validator=(
            None
            if artifact is None
            else lambda device: validate_target(artifact, device)
        ),
    )


@functools.lru_cache(maxsize=1)
def _driver_max_ptx_version() -> Optional[int]:
    """Highest PTX ISA the running CUDA driver can load, as a ``ptx_version`` int.

    A driver supports PTX up to the ISA of the CUDA toolkit it ships with, so we
    read the driver's CUDA version (``cuDriverGetVersion``) and map it with
    Triton's own ``ptx_get_version`` — the exact mapping Triton uses to decide
    which ISA to emit. This avoids both a hard-coded version table and trial
    module loads. Returns ``None`` if it can't be determined (e.g. Triton
    internals moved), in which case no capping is applied.
    """
    try:
        try:
            from cuda.bindings import driver as cuda
        except ImportError:  # cuda-python < 12.6
            from cuda import cuda
        from triton.backends.nvidia.compiler import ptx_get_version

        init_status = cuda.cuInit(0)[0]
        if int(init_status) != 0:
            raise RuntimeError(f"cuInit failed with CUDA error {int(init_status)}")
        version_status, raw = cuda.cuDriverGetVersion()
        if int(version_status) != 0:
            raise RuntimeError(
                f"cuDriverGetVersion failed with CUDA error {int(version_status)}"
            )
        # CUDA's integer encoding uses 13010 for CUDA 13.1.
        cuda_version = f"{raw // 1000}.{(raw % 1000) // 10}"
        return int(ptx_get_version(cuda_version))
    except Exception as exc:  # pragma: no cover - environment dependent
        _LOGGER.debug("Could not determine driver PTX version: %s", exc)
        return None


def _triton_default_ptx_version() -> Optional[int]:
    """The ISA Triton would emit for this GPU, without compiling anything.

    Triton picks its ISA from the version of the ``ptxas`` it bundles, so asking
    that binary directly predicts the ``.version`` header a compile would
    produce. Knowing it up front lets the caller compile once at the right ISA
    instead of compiling, noticing the emitted ISA is too new, and compiling
    again. Returns ``None`` if Triton's internals moved, which puts the caller
    back on the compile-then-check path.
    """
    try:
        from triton.backends.nvidia.compiler import get_ptxas, ptx_get_version

        # Triton 3.5 takes no arguments; newer versions select ptxas by SM.
        args = (_device_arch(),) if inspect.signature(get_ptxas).parameters else ()
        return int(ptx_get_version(get_ptxas(*args).version))
    except Exception as exc:  # pragma: no cover - environment dependent
        _LOGGER.debug("Could not determine Triton's default PTX version: %s", exc)
        return None


def _parse_ptx_version(ptx: str) -> Optional[int]:
    """``.version 9.3`` -> ``93``, the form Triton's ``ptx_version`` option uses."""
    match = re.search(r"(?m)^\s*\.version\s+(\d+)\.(\d+)(?:\s|$)", ptx)
    if match is None:
        return None
    return int(match.group(1)) * 10 + int(match.group(2))


def _require_supported_triton_version(triton: Any) -> None:
    """Reject Triton releases without the compiler metadata contract we use."""
    raw_version = getattr(triton, "__version__", None)
    try:
        if not isinstance(raw_version, str):
            raise TypeError("Triton version must be a string")
        installed_version = Version(raw_version)
    except (InvalidVersion, TypeError):
        raise ImportError(
            "torch_tensorrt.kernels.triton_op could not determine the installed "
            "Triton version. Install a supported release with: "
            f"pip install 'triton>={_MIN_TRITON_VERSION}'"
        ) from None

    if installed_version < _MIN_TRITON_VERSION:
        raise ImportError(
            "torch_tensorrt.kernels.triton_op requires Triton "
            f">={_MIN_TRITON_VERSION}; found {raw_version}. The required launch "
            "metadata is unavailable in older releases. Upgrade with: "
            f"pip install 'triton>={_MIN_TRITON_VERSION}'"
        )


def _triton_import() -> Any:
    """Import a supported Triton, raising an actionable dependency error."""
    try:
        import triton
    except ImportError:
        raise ImportError(
            "triton is required for triton_op plugins. "
            f"Install it with: pip install 'triton>={_MIN_TRITON_VERSION}'"
        ) from None

    _require_supported_triton_version(triton)
    try:
        import triton.compiler  # noqa: F401  (ensures ASTSource is importable)
    except ImportError:
        raise ImportError(
            "The installed Triton package does not expose triton.compiler. "
            "Install a supported release with: "
            f"pip install 'triton>={_MIN_TRITON_VERSION}'"
        ) from None
    return triton


def _compile_within_driver_isa(
    triton: Any,
    kernel: Any,
    signature: Dict[str, str],
    constexprs: Dict[str, Any],
    options: Dict[str, Any],
    target: Any,
) -> Any:
    """Compile ``kernel``, holding the PTX ISA to what the driver can load.

    Triton derives the ISA from its bundled ``ptxas``, which can be newer than
    the installed driver. The driver then rejects the PTX at load time with
    ``CUDA_ERROR_UNSUPPORTED_PTX_VERSION``, surfacing for AOT plugins as a TRT
    ``onShapeChange`` failure at engine *runtime* — long after registration, and
    with nothing in the message pointing at the ISA.

    The cap only ever lowers the ISA: a ``ptx_version`` above what the bundled
    ``ptxas`` can assemble fails the compile outright.
    """

    def _compile(ptx_version: Optional[int] = None) -> Any:
        src = triton.compiler.ASTSource(
            fn=kernel,
            signature=dict(signature),
            constexprs=dict(constexprs),
        )
        opts = dict(options)
        if ptx_version is not None:
            opts["ptx_version"] = ptx_version
        return triton.compile(src, target=target, options=opts or None)

    driver_max = _driver_max_ptx_version()
    if driver_max is None:
        # Nothing to hold the ISA to; take whatever Triton emits.
        return _compile()

    label = getattr(kernel, "__name__", "<kernel>")
    default_isa = _triton_default_ptx_version()
    if default_isa is not None:
        # The ISA is known before compiling, so one compile suffices.
        if default_isa <= driver_max:
            return _compile()
        _LOGGER.debug(
            "Triton would emit PTX ISA %d but the driver accepts at most %d; "
            "compiling '%s' with ptx_version=%d",
            default_isa,
            driver_max,
            label,
            driver_max,
        )
        return _compile(ptx_version=driver_max)

    # The ISA could not be predicted: compile, then pay for a second compile
    # only if what came out is actually too new.
    compiled = _compile()
    emitted = _parse_ptx_version(_compiled_ptx(compiled, label))
    if emitted is None or emitted <= driver_max:
        return compiled
    _LOGGER.debug(
        "PTX ISA %d exceeds driver max %d; recompiling '%s' with ptx_version=%d",
        emitted,
        driver_max,
        label,
        driver_max,
    )
    return _compile(ptx_version=driver_max)


def _compiled_ptx(compiled: Any, kernel_name: str) -> str:
    """Return non-empty PTX text from a Triton compilation result."""
    asm = getattr(compiled, "asm", None)
    ptx = asm.get("ptx") if isinstance(asm, Mapping) else None
    if not isinstance(ptx, str) or not ptx:
        raise RuntimeError(
            f"Triton kernel '{kernel_name}' did not produce non-empty PTX text."
        )
    return ptx


def _launch_metadata_int(metadata: Any, field: str, kernel_name: str) -> int:
    """Read one exact integer field without lossy ``int(...)`` coercion."""
    value = getattr(metadata, field)
    if isinstance(value, bool):
        raise RuntimeError(
            f"Triton kernel '{kernel_name}' launch metadata '{field}' must be "
            f"an integer; got bool."
        )
    try:
        return operator.index(value)
    except TypeError as exc:
        raise RuntimeError(
            f"Triton kernel '{kernel_name}' launch metadata '{field}' must be "
            f"an integer; got {type(value).__name__}."
        ) from exc


def _launch_metadata_bool(metadata: Any, field: str, kernel_name: str) -> bool:
    """Read one exact Boolean field without truthiness coercion."""
    value = getattr(metadata, field)
    if not isinstance(value, bool):
        raise RuntimeError(
            f"Triton kernel '{kernel_name}' launch metadata '{field}' must be "
            f"bool; got {type(value).__name__}."
        )
    return value


def _validate_launch_metadata(compiled: Any) -> Tuple[str, int, int]:
    """Refuse Triton launch features the QDP AOT path cannot reproduce.

    Triton appends ``global_scratch`` / ``profile_scratch`` pointer params to
    every kernel. TensorRT pads the kernel arguments it wasn't given with null,
    so zero-sized scratch needs no handling — null is exactly what Triton itself
    passes in that case. A kernel that *uses* its scratch would receive that
    same null and fault, so reject it while there is still a name to blame.
    """
    metadata = getattr(compiled, "metadata", None)
    if metadata is None:
        raise RuntimeError(
            "Triton compilation returned no launch metadata; refusing to assume "
            "its hidden launch ABI."
        )
    required_fields = (
        "name",
        "num_warps",
        "shared",
        "global_scratch_size",
        "profile_scratch_size",
        "num_ctas",
        "warp_size",
        "launch_cooperative_grid",
        "launch_pdl",
        "tmem_size",
        "tensordesc_meta",
    )
    missing = [field for field in required_fields if not hasattr(metadata, field)]
    if missing:
        raise RuntimeError(
            "Triton compilation uses an unsupported compiler metadata "
            f"layout missing {missing}; refusing to assume its hidden launch ABI."
        )

    kernel_name = metadata.name
    if not isinstance(kernel_name, str) or not kernel_name:
        raise RuntimeError("Triton compilation reports an invalid kernel name.")

    numeric_fields = (
        "num_warps",
        "shared",
        "global_scratch_size",
        "profile_scratch_size",
        "num_ctas",
        "warp_size",
        "tmem_size",
    )
    numeric = {
        field: _launch_metadata_int(metadata, field, kernel_name)
        for field in numeric_fields
    }
    cooperative = _launch_metadata_bool(
        metadata, "launch_cooperative_grid", kernel_name
    )
    pdl = _launch_metadata_bool(metadata, "launch_pdl", kernel_name)
    tensordesc_meta = metadata.tensordesc_meta
    if not isinstance(tensordesc_meta, (list, tuple)):
        raise RuntimeError(
            f"Triton kernel '{kernel_name}' launch metadata 'tensordesc_meta' "
            f"must be a list or tuple; got {type(tensordesc_meta).__name__}."
        )

    global_scratch = numeric["global_scratch_size"]
    profile_scratch = numeric["profile_scratch_size"]
    if global_scratch or profile_scratch:
        raise RuntimeError(
            f"Triton kernel '{kernel_name}' requires non-zero scratch memory "
            f"(global={global_scratch}, profile={profile_scratch}), which the "
            "TensorRT AOT QDP launch path cannot provide. Rewrite the kernel "
            "to avoid Triton scratch buffers to use triton_op."
        )

    unsupported = []
    if numeric["num_ctas"] != 1:
        unsupported.append(f"num_ctas={metadata.num_ctas}")
    if numeric["warp_size"] != 32:
        unsupported.append(f"warp_size={metadata.warp_size}")
    if cooperative:
        unsupported.append("launch_cooperative_grid=True")
    if pdl:
        unsupported.append("launch_pdl=True")
    if numeric["tmem_size"]:
        unsupported.append(f"tmem_size={metadata.tmem_size}")
    if tensordesc_meta:
        unsupported.append("tensordesc_meta is non-empty")
    if unsupported:
        raise RuntimeError(
            f"Triton kernel '{kernel_name}' requires unsupported launch feature(s): "
            + ", ".join(unsupported)
            + ". The TensorRT AOT QDP path currently supports only ordinary, "
            "non-clustered CUDA launches."
        )

    compiled_warps = numeric["num_warps"]
    shared_mem = numeric["shared"]
    if not 1 <= compiled_warps <= 32:
        raise RuntimeError(
            f"Triton kernel '{kernel_name}' reports invalid num_warps="
            f"{compiled_warps}."
        )
    if not 0 <= shared_mem < 2**31:
        raise RuntimeError(
            f"Triton kernel '{kernel_name}' reports invalid shared memory size "
            f"{shared_mem}."
        )
    return kernel_name, compiled_warps, shared_mem


def compile_triton_to_ptx(
    kernel: Any,
    signature: Dict[str, str],
    constexprs: Dict[str, Any],
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
) -> CompiledTritonArtifact:
    """Compile a ``@triton.jit`` kernel to PTX ahead of time.

    This mirrors what ``examples/dynamo/aot_plugin.py`` does by hand inside its
    ``@trtp.aot_impl`` body, but performs it once so callers don't have to.

    Args:
        kernel: the ``@triton.jit`` kernel function.
        signature: Triton signature mapping *non-constexpr* parameter names to
            Triton type strings, in kernel-declaration order — e.g.
            ``{"x_ptr": "*fp32", "n_elements": "i32", "y_ptr": "*fp32"}``.
            Pointer args use ``*<dtype>``; scalar args use the bare dtype.
        constexprs: compile-time ``tl.constexpr`` values baked into the PTX,
            e.g. ``{"BLOCK_SIZE": 256}``. These must NOT appear in ``signature``.
        num_warps: warps per block, forwarded to ``triton.compile``. Defaults to
            Triton's own choice. Also sets the launch's threads-per-block.
        num_stages: software pipelining depth, forwarded to ``triton.compile``.
            Defaults to Triton's own choice.

    Returns:
        An immutable artifact containing unmodified PTX, the entry symbol,
        launch metadata, GPU target, and PTX ISA version.
    """
    validate_compile_options(num_warps, num_stages)
    triton = _triton_import()
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "cuda" or target.warp_size != 32:
        raise ValueError(
            "triton_op requires an NVIDIA CUDA target with 32-thread warps."
        )

    # Bind compilation to the recorded device, including when an environment
    # override would otherwise select a different architecture.
    options: Dict[str, Any] = {"arch": f"sm{target.arch}"}
    if num_warps is not None:
        options["num_warps"] = num_warps
    if num_stages is not None:
        options["num_stages"] = num_stages

    compiled = _compile_within_driver_isa(
        triton, kernel, signature, constexprs, options, target
    )
    # Read back from metadata rather than trusting the request: Triton clamps
    # num_warps to what the kernel can actually use.
    kernel_name, compiled_warps, shared_mem = _validate_launch_metadata(compiled)

    # Embedded exactly as Triton emitted it. TensorRT sizes the kernel argument
    # buffer from the kernel's own declared ABI and pads the slots it wasn't
    # given with null -- libnvinfer logs "AOT Plugin: ... has N hidden args ...
    # Padding with null/zero values" on every registration -- so Triton's
    # trailing scratch params need no rewriting.
    ptx = _compiled_ptx(compiled, kernel_name)
    ptx_bytes = ptx.encode("utf-8")

    _LOGGER.debug(
        "Compiled triton kernel '%s' -> PTX (%d bytes, num_warps=%d, shared=%d)",
        kernel_name,
        len(ptx_bytes),
        compiled_warps,
        shared_mem,
    )
    return CompiledTritonArtifact(
        ptx_bytes,
        kernel_name,
        compiled_warps,
        shared_mem,
        target,
        _parse_ptx_version(ptx),
    )
