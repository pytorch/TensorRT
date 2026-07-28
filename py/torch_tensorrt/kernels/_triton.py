from __future__ import annotations

import functools
import logging
import re
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import torch

_LOGGER = logging.getLogger(__name__)

# Triton element-type spellings -> torch dtypes. Spellings absent here (rarer fp8
# variants, future additions) resolve to None via ``.get``, which disables dtype
# checking for that parameter rather than rejecting a kernel we don't recognize.
_TRITON_TO_TORCH_DTYPE: Dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
    "i1": torch.bool,
    "i8": torch.int8,
    "i16": torch.int16,
    "i32": torch.int32,
    "i64": torch.int64,
    "u8": torch.uint8,
    "fp8e4nv": torch.float8_e4m3fn,
    "fp8e5": torch.float8_e5m2,
}


class SignatureParam(NamedTuple):
    """One entry of a Triton ``signature``, decoded."""

    name: str
    is_pointer: bool
    dtype: Optional[torch.dtype]


class SignatureLayout(NamedTuple):
    """A ``signature`` split along the ``(inputs, extras, outputs)`` convention."""

    inputs: List[SignatureParam]
    scalars: List[SignatureParam]
    outputs: List[SignatureParam]


def _parse_signature(signature: Dict[str, str]) -> List[SignatureParam]:
    """Decode each ``signature`` value into pointer-ness and a torch dtype.

    Accepts the ``*fp32`` / ``fp32`` spellings, tolerating Triton's optional
    ``:<alignment>`` specializer suffix (``*fp32:16``).
    """
    params = []
    for name, type_str in signature.items():
        text = str(type_str).strip()
        is_pointer = text.startswith("*")
        element = text.lstrip("*").split(":", 1)[0].strip()
        params.append(
            SignatureParam(name, is_pointer, _TRITON_TO_TORCH_DTYPE.get(element))
        )
    return params


def analyze_signature(
    signature: Dict[str, str],
    arity: Optional[Tuple[int, int]] = None,
) -> SignatureLayout:
    """Split a ``signature`` into input pointers, extra scalars, output pointers.

    ``triton_op`` requires the kernel's runtime parameters to be declared as
    ``(input_ptrs..., extra_scalars..., output_ptrs...)`` because that is the
    order TensorRT passes tensor pointers and AOT extra arguments. This checks
    the declaration actually has that shape and, when ``arity`` gives the op's
    ``(tensor inputs, outputs)`` counts, that the pointer runs are that long.

    Getting this wrong does not fail loudly at runtime — the kernel reads
    whatever TensorRT happened to place in those slots — so every deviation is
    rejected here, at registration time.

    Raises:
        ValueError: if the signature is empty, starts or ends with a scalar,
            interleaves scalars between pointers, or disagrees with ``arity``.
    """
    params = _parse_signature(signature)
    if not params:
        raise ValueError(
            "triton_op signature is empty; it must declare the kernel's "
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
            f"triton_op signature must begin and end with pointer parameters. "
            f"{convention()}"
        )

    scalar_positions = [i for i, p in enumerate(params) if not p.is_pointer]
    if scalar_positions and scalar_positions != list(
        range(scalar_positions[0], scalar_positions[-1] + 1)
    ):
        raise ValueError(
            f"triton_op signature interleaves scalar parameters with pointers. "
            f"Scalars must form one contiguous run between the input and output "
            f"pointers. {convention()}"
        )

    if scalar_positions:
        # The scalar run pins the boundary, so the split is known either way.
        num_leading = scalar_positions[0]
        num_trailing = len(params) - scalar_positions[-1] - 1
        mismatched = arity is not None and (num_leading, num_trailing) != arity
    elif arity is not None:
        # All pointers: only the op's arity can say where inputs end.
        num_leading, num_trailing = arity
        mismatched = num_leading + num_trailing != len(params)
    else:
        num_leading, num_trailing, mismatched = len(params), 0, False

    if mismatched:
        assert arity is not None
        raise ValueError(
            f"triton_op signature declares {num_leading} leading and "
            f"{num_trailing} trailing pointer parameter(s) but the op takes "
            f"{arity[0]} tensor input(s) and returns {arity[1]} output(s). "
            f"{convention()}"
        )

    return SignatureLayout(
        inputs=params[:num_leading],
        scalars=params[num_leading : len(params) - num_trailing],
        outputs=params[len(params) - num_trailing :],
    )


def validate_triton_config(
    op_name: str,
    signature: Dict[str, str],
    arity: Optional[Tuple[int, int]],
    extra_args_fn: Optional[Callable[..., Any]],
    derived_launch: bool,
) -> SignatureLayout:
    """Check a ``triton_op`` registration and return its signature layout.

    Every rule here is answerable from the signature and the op's arity alone,
    and every one of them, left unchecked, yields wrong numbers rather than an
    error — so they are enforced before anything is compiled.

    ``derived_launch`` is False when the caller supplied its own ``aot_fn``,
    which owns the extra arguments and so is exempt from the pairing rules.
    """
    layout = analyze_signature(signature, arity)
    if not derived_launch:
        return layout

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


def make_dtype_capability_validator(
    op_name: str,
    layout: SignatureLayout,
    user_validator: Optional[Callable[..., bool]] = None,
) -> Callable[..., bool]:
    """Build a converter capability validator enforcing the compiled dtypes.

    The PTX is compiled once for the dtypes named in ``signature``. Feeding the
    op tensors of any other dtype reinterprets their bytes and silently returns
    wrong numbers, so decline the conversion instead: TensorRT then leaves the
    op to PyTorch rather than embedding a kernel that cannot read its inputs.
    """
    expected_inputs = [p.dtype for p in layout.inputs]
    expected_outputs = [p.dtype for p in layout.outputs]

    def _tensor_meta(value: Any) -> Optional[torch.Tensor]:
        meta = getattr(value, "meta", None)
        if not isinstance(meta, dict):
            return None
        val = meta.get("val")
        return val if isinstance(val, torch.Tensor) else None

    def _mismatch(kind: str, index: int, got: torch.Tensor, want: torch.dtype) -> bool:
        # Warn, not debug: the op silently leaves the engine and runs in
        # PyTorch, and if no eager_fn was registered the eventual failure is an
        # opaque "not implemented for the CUDA backend" from the dispatcher.
        _LOGGER.warning(
            "Not lowering '%s' to its Triton plugin: %s %d is %s but the kernel "
            "was compiled for %s. Re-register with a matching signature to run "
            "it inside TensorRT; it will fall back to PyTorch for now.",
            op_name,
            kind,
            index,
            got.dtype,
            want,
        )
        return False

    def _validator(node: Any, settings: Any = None) -> bool:
        if user_validator is not None and not user_validator(node, settings):
            return False

        actual_inputs = [t for t in map(_tensor_meta, node.args) if t is not None]
        for index, (got, want) in enumerate(zip(actual_inputs, expected_inputs)):
            if want is not None and got.dtype != want:
                return _mismatch("input", index, got, want)

        produced = node.meta.get("val") if isinstance(node.meta, dict) else None
        actual_outputs = (
            list(produced) if isinstance(produced, (tuple, list)) else [produced]
        )
        for index, (got, want) in enumerate(zip(actual_outputs, expected_outputs)):
            if want is not None and isinstance(got, torch.Tensor) and got.dtype != want:
                return _mismatch("output", index, got, want)

        return True

    return _validator


def _parse_entry_params(
    ptx: str, kernel_name: str
) -> Tuple[Optional[re.Match[str]], List[str]]:
    """Locate the ``.visible .entry <kernel_name>( ... )`` param list.

    Returns the regex match (groups: prefix, param-block, close-paren) and the
    list of individual ``.param`` declaration strings, in order.
    """
    pattern = re.compile(
        r"(\.visible\s+\.entry\s+" + re.escape(kernel_name) + r"\s*\()([^)]*)(\))",
        re.DOTALL,
    )
    match = pattern.search(ptx)
    if match is None:
        return match, []
    params = [p.strip() for p in match.group(2).split(",") if p.strip()]
    return match, params


def _strip_trailing_scratch_params(
    ptx: str,
    match: Optional[re.Match[str]],
    params: List[str],
    keep: int,
) -> str:
    """Drop Triton's trailing scratch params so the entry matches TRT's launch.

    Triton (3.x) unconditionally appends ``global_scratch`` and
    ``profile_scratch`` pointer parameters after the user's kernel arguments.
    TensorRT's AOT QDP launcher only passes the declared tensor + extra-scalar
    arguments, so the extra params make the kernel's parameter count exceed what
    TRT supplies — the launch is misaligned and fails at ``onShapeChange``.

    When those scratch buffers are zero-sized (the common case) the params are
    unused in the kernel body, so removing them from the ``.entry`` signature is
    a safe, purely-syntactic fix. ``match`` and ``params`` come from
    :func:`_parse_entry_params`; ``keep`` is the number of real runtime
    parameters (i.e. ``len(signature)``).
    """
    if match is None or len(params) <= keep:
        return ptx
    kept = params[:keep]
    new_block = "\n\t" + ",\n\t".join(kept) + "\n"
    return (
        ptx[: match.start()]
        + match.group(1)
        + new_block
        + match.group(3)
        + ptx[match.end() :]
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
        from cuda.bindings import driver as cuda
        from triton.backends.nvidia.compiler import ptx_get_version

        cuda.cuInit(0)
        raw = cuda.cuDriverGetVersion()[1]  # e.g. 13010 -> CUDA 13.1
        cuda_version = f"{raw // 1000}.{(raw % 1000) // 10}"
        return int(ptx_get_version(cuda_version))
    except Exception as exc:  # pragma: no cover - environment dependent
        _LOGGER.debug("Could not determine driver PTX version: %s", exc)
        return None


@functools.lru_cache(maxsize=1)
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

        major, minor = torch.cuda.get_device_capability()
        return int(ptx_get_version(get_ptxas(major * 10 + minor).version))
    except Exception as exc:  # pragma: no cover - environment dependent
        _LOGGER.debug("Could not determine Triton's default PTX version: %s", exc)
        return None


def _parse_ptx_version(ptx: str) -> Optional[int]:
    """``.version 9.3`` -> ``93``, the form Triton's ``ptx_version`` option uses."""
    match = re.search(r"\.version (\d+)\.(\d+)", ptx)
    if match is None:
        return None
    return int(match.group(1)) * 10 + int(match.group(2))


def _triton_import() -> Any:
    """Import triton, raising an actionable error if it is not installed."""
    try:
        import triton
        import triton.compiler  # noqa: F401  (ensures ASTSource is importable)

        return triton
    except ImportError:
        raise ImportError(
            "triton is required for triton_op plugins. "
            "Install it with: pip install triton"
        )


def compile_triton_to_ptx(
    kernel: Any,
    signature: Dict[str, str],
    constexprs: Dict[str, Any],
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
) -> Tuple[bytes, str, int, int]:
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
        ``(ptx_bytes, kernel_name, num_warps, shared_mem_bytes)`` — the PTX to
        embed in the TRT engine, the entry symbol inside it, and the launch
        metadata needed to build ``trtp.KernelLaunchParams``.
    """
    triton = _triton_import()

    base_options: Dict[str, Any] = {}
    if num_warps is not None:
        base_options["num_warps"] = num_warps
    if num_stages is not None:
        base_options["num_stages"] = num_stages

    def _compile(ptx_version: Optional[int] = None) -> Any:
        src = triton.compiler.ASTSource(
            fn=kernel,
            signature=dict(signature),
            constexprs=dict(constexprs),
        )
        options = dict(base_options)
        if ptx_version is not None:
            options["ptx_version"] = ptx_version
        return triton.compile(src, options=options or None)

    # Triton derives the ISA from its bundled ptxas, which can be newer than the
    # installed driver; the driver then rejects the PTX at load time with
    # CUDA_ERROR_UNSUPPORTED_PTX_VERSION, surfacing for AOT plugins as a TRT
    # ``onShapeChange`` failure at engine runtime. Only lower, never raise — a
    # ptx_version above what the bundled ptxas can assemble fails the compile.
    kernel_label = getattr(kernel, "__name__", "<kernel>")
    driver_max = _driver_max_ptx_version()
    default_isa = _triton_default_ptx_version()
    cap = (
        driver_max
        if driver_max is not None
        and default_isa is not None
        and default_isa > driver_max
        else None
    )
    if cap is not None:
        _LOGGER.debug(
            "Triton would emit PTX ISA %d but the driver accepts at most %d; "
            "compiling '%s' with ptx_version=%d",
            default_isa,
            cap,
            kernel_label,
            cap,
        )

    compiled = _compile(ptx_version=cap)
    ptx_text: str = compiled.asm["ptx"]

    # Fallback for when the ISA couldn't be predicted up front: check what was
    # actually emitted and pay for a second compile only if it is too new.
    if cap is None and driver_max is not None:
        emitted = _parse_ptx_version(ptx_text)
        if emitted is not None and emitted > driver_max:
            _LOGGER.debug(
                "PTX ISA %d exceeds driver max %d; recompiling '%s' with "
                "ptx_version=%d",
                emitted,
                driver_max,
                kernel_label,
                driver_max,
            )
            compiled = _compile(ptx_version=driver_max)
            ptx_text = compiled.asm["ptx"]

    kernel_name = compiled.metadata.name
    # Read back from metadata rather than trusting the request: Triton clamps
    # num_warps to what the kernel can actually use.
    compiled_warps = int(compiled.metadata.num_warps)
    shared_mem = int(compiled.metadata.shared)

    # Zero-sized scratch params are unused and safe to strip; non-zero ones mean
    # the kernel needs scratch the AOT QDP path cannot provide.
    global_scratch = int(getattr(compiled.metadata, "global_scratch_size", 0) or 0)
    profile_scratch = int(getattr(compiled.metadata, "profile_scratch_size", 0) or 0)
    entry, params = _parse_entry_params(ptx_text, kernel_name)
    num_runtime_params = len(signature)
    if len(params) > num_runtime_params:
        if global_scratch != 0 or profile_scratch != 0:
            raise RuntimeError(
                f"Triton kernel '{kernel_name}' requires non-zero scratch memory "
                f"(global={global_scratch}, profile={profile_scratch}), which the "
                "TensorRT AOT QDP launch path cannot provide. Rewrite the kernel "
                "to avoid Triton scratch buffers to use triton_op."
            )
        ptx_text = _strip_trailing_scratch_params(
            ptx_text, entry, params, num_runtime_params
        )
        _LOGGER.debug(
            "Stripped %d trailing scratch param(s) from triton kernel '%s'",
            len(params) - num_runtime_params,
            kernel_name,
        )

    ptx_bytes = ptx_text.encode("utf-8")

    _LOGGER.debug(
        "Compiled triton kernel '%s' -> PTX (%d bytes, num_warps=%d, shared=%d)",
        kernel_name,
        len(ptx_bytes),
        compiled_warps,
        shared_mem,
    )
    return ptx_bytes, kernel_name, compiled_warps, shared_mem
