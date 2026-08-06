"""cuTile backend for ``torch_tensorrt.kernels.cutile_op``.

Compiles a ``@ct.kernel`` cuTile program ahead of time and reshapes the result
into what TensorRT's AOT Quick Deployable Plugin launcher expects.

Pipeline
--------
1. **Signature validation** — the ``signature`` names the kernel's array
   parameters (inputs then outputs) and ``constants`` the ``ct.Constant[...]``
   parameters that follow them. Both are checked against the op's arity before
   anything is compiled.
2. **CUBIN compilation** — ``cuda.tile.compilation.export_kernel`` builds a
   CUBIN for a ``KernelSignature`` of ``ArrayConstraint`` / ``ConstantConstraint``
   entries derived from the signature.
3. **PTX extraction** — the CUBIN embeds its PTX in a debug section; it is
   recovered from the raw ELF bytes.
4. **Parameter reordering** — the cuTile kernel ABI groups parameters per array
   as ``(ptr, extents..., strides...)``, in declaration order. TRT's AOT
   launcher passes ``(input_ptrs..., extra_args..., output_ptrs...)``. The
   ``.entry`` parameter list is permuted so the two agree, and
   :func:`build_extra_args` produces the matching extents / strides.

Every step that could silently bind the wrong argument raises instead. A
misordered launch does not fail — the kernel reads whatever TensorRT happened
to place in those slots and returns plausible-looking garbage.
"""

from __future__ import annotations

import io
import logging
import re
import shutil
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch

_LOGGER = logging.getLogger(__name__)

# Short element-type spellings accepted in a ``signature``, mapped to the torch
# dtype name. Any torch dtype name is also accepted directly, so this only has
# to cover the abbreviations; both cuTile and torch spell the canonical names
# identically ("float32", "bfloat16", ...), which is what lets a single name
# serve for the dtype lookup and the ``cuda.tile`` attribute lookup.
_DTYPE_ALIASES = {
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp32": "float32",
    "fp64": "float64",
    "i1": "bool",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "u8": "uint8",
}


def _dtype_name(dtype: torch.dtype) -> str:
    """``torch.bfloat16`` -> ``"bfloat16"`` — also the ``cuda.tile`` attribute."""
    return str(dtype).rsplit(".", 1)[-1]


class ArrayParam(NamedTuple):
    """One array parameter of a cuTile kernel, decoded from ``signature``."""

    name: str
    dtype: torch.dtype
    ndim: int

    @property
    def num_slots(self) -> int:
        """cuTile array ABI: ``ptr`` + one extent and one stride per dimension.

        The single statement of that shape. :func:`cutile_param_order` places
        the slots and :func:`build_extra_args` fills the non-pointer ones, so
        the two must agree on this count.
        """
        return 1 + 2 * self.ndim


class SignatureLayout(NamedTuple):
    """A ``signature`` split into the op's input and output arrays."""

    inputs: List[ArrayParam]
    outputs: List[ArrayParam]

    @property
    def arrays(self) -> List[ArrayParam]:
        """All arrays in kernel-declaration order: inputs, then outputs."""
        return self.inputs + self.outputs

    @property
    def num_slots(self) -> int:
        return sum(p.num_slots for p in self.arrays)


_NDIM_SUFFIX_RE = re.compile(r"^(?P<element>[^\[\]]+)\[(?P<ndim>\d+)\]$")


def _parse_array_type(name: str, spelling: Any, default_ndim: int) -> ArrayParam:
    """Decode one ``signature`` entry into a dtype and a rank.

    Accepts a :class:`torch.dtype`, a dtype name (``"float32"``) or one of the
    :data:`_DTYPE_ALIASES` abbreviations (``"fp32"``), each optionally carrying
    an explicit rank as ``"fp32[2]"`` for kernels whose arrays differ in rank.
    """
    ndim = default_ndim
    if isinstance(spelling, torch.dtype):
        dtype: Optional[torch.dtype] = spelling
    else:
        text = str(spelling).strip()
        suffix = _NDIM_SUFFIX_RE.match(text)
        if suffix is not None:
            text, ndim = suffix.group("element").strip(), int(suffix.group("ndim"))
        canonical = _DTYPE_ALIASES.get(text.lower(), text.lower())
        candidate = getattr(torch, canonical, None)
        dtype = candidate if isinstance(candidate, torch.dtype) else None

    if dtype is None:
        raise ValueError(
            f"cutile_op signature entry '{name}' has unknown element type "
            f"{spelling!r}. cuTile must be told the exact dtype to compile for; "
            f"pass a torch.dtype, a dtype name such as 'float32', or one of: "
            f"{', '.join(sorted(_DTYPE_ALIASES))}."
        )
    if ndim < 1:
        raise ValueError(
            f"cutile_op signature entry '{name}' declares rank {ndim}; "
            "cuTile arrays have rank >= 1."
        )
    return ArrayParam(name, dtype, ndim)


def validate_cutile_config(
    op_name: str,
    signature: Dict[str, Any],
    constants: Dict[str, Any],
    arity: Optional[Tuple[int, int]],
    default_ndim: int = 1,
    derived_launch: bool = True,
    has_grid: bool = True,
) -> SignatureLayout:
    """Check a ``cutile_op`` registration and return its signature layout.

    ``signature`` lists the kernel's array parameters in declaration order,
    inputs first then outputs; ``ct.Constant`` parameters are not part of it.
    ``arity`` is the ``(tensor inputs, outputs)`` of the op being registered and
    is what decides where the inputs end. ``derived_launch`` is False when the
    caller supplied its own ``aot_fn``, which replaces ``grid``.

    Everything here is answerable before anything is compiled, and every rule,
    left unchecked, produces wrong numbers rather than an error.
    """
    if default_ndim < 1:
        raise ValueError(
            f"cutile_op '{op_name}' was given ndim={default_ndim}; "
            "cuTile arrays have rank >= 1."
        )

    if derived_launch and not has_grid:
        raise ValueError(
            f"cutile_op '{op_name}' needs a grid= to build the launch from, or "
            "an aot_fn= to replace it."
        )
    if not derived_launch and has_grid:
        raise ValueError(
            f"cutile_op '{op_name}' was given both grid= and aot_fn=; an aot_fn "
            "builds the whole launch, so grid= would be ignored. Drop one."
        )

    overlap = sorted(set(signature) & set(constants))
    if overlap:
        raise ValueError(
            f"cutile_op '{op_name}' declares {overlap} in both signature and "
            "constants. Array parameters belong in signature; ct.Constant "
            "parameters belong in constants."
        )

    for name, value in constants.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"cutile_op '{op_name}' constant '{name}' is {value!r}; cuTile "
                "bakes ct.Constant parameters into the compiled symbol as "
                "integers, so only int values are supported."
            )

    params = [
        _parse_array_type(name, spelling, default_ndim)
        for name, spelling in signature.items()
    ]
    if not params:
        raise ValueError(
            f"cutile_op '{op_name}' signature is empty; it must declare the "
            "kernel's array parameters (inputs then outputs) in declaration "
            "order. Scalars belong in constants=, not signature=."
        )

    if arity is None:
        # Guessing the split would feed cutile_param_order a wrong permutation
        # and silently misbind the launch, so ask instead. ``schema`` exists
        # precisely to state this when meta_fn's annotations can't be read.
        raise ValueError(
            f"cutile_op '{op_name}' could not determine how many tensors the op "
            "takes and returns from meta_fn's type hints, so the signature "
            "cannot be split into inputs and outputs. Pass schema= (e.g. "
            '"(Tensor x) -> Tensor").'
        )

    num_inputs, num_outputs = arity
    if num_inputs + num_outputs != len(params):
        raise ValueError(
            f"cutile_op '{op_name}' signature declares {len(params)} array "
            f"parameter(s) ({', '.join(p.name for p in params)}) but the op "
            f"takes {num_inputs} tensor input(s) and returns {num_outputs} "
            "output(s). The signature must list every input array followed "
            "by every output array."
        )

    return SignatureLayout(inputs=params[:num_inputs], outputs=params[num_inputs:])


def make_dtype_capability_validator(
    op_name: str,
    layout: SignatureLayout,
    user_validator: Optional[Callable[..., bool]] = None,
) -> Callable[..., bool]:
    """Build a converter capability validator enforcing the compiled dtypes.

    The kernel is compiled once for the dtypes named in ``signature``. Feeding
    the op tensors of any other dtype reinterprets their bytes and silently
    returns wrong numbers, so decline the conversion instead: TensorRT then
    leaves the op to PyTorch rather than embedding a kernel that cannot read
    its inputs.
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
            "Not lowering '%s' to its cuTile plugin: %s %d is %s but the kernel "
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
            if got.dtype != want:
                return _mismatch("input", index, got, want)

        produced = node.meta.get("val") if isinstance(node.meta, dict) else None
        actual_outputs = (
            list(produced) if isinstance(produced, (tuple, list)) else [produced]
        )
        for index, (got, want) in enumerate(zip(actual_outputs, expected_outputs)):
            if isinstance(got, torch.Tensor) and got.dtype != want:
                return _mismatch("output", index, got, want)

        return True

    return _validator


def cutile_param_order(layout: SignatureLayout) -> Tuple[int, ...]:
    """The permutation mapping TensorRT's slot order onto cuTile's.

    cuTile declares, for each array in kernel order, ``(ptr, extents...,
    strides...)``. TensorRT's AOT launcher fills the parameter slots with
    ``(input_ptrs..., extra_args..., output_ptrs...)``. ``permutation[i]`` is
    the cuTile parameter index that must be moved into physical slot ``i``, so
    the extras land exactly where :func:`build_extra_args` puts them: every
    input's extents and strides, then every output's.
    """
    offsets: List[int] = []
    total = 0
    for param in layout.arrays:
        offsets.append(total)
        total += param.num_slots

    def pointer(index: int) -> int:
        return offsets[index]

    def extents_strides(index: int) -> range:
        start = offsets[index] + 1
        return range(start, start + 2 * layout.arrays[index].ndim)

    inputs = range(len(layout.inputs))
    outputs = range(len(layout.inputs), len(layout.arrays))
    return tuple(
        [pointer(i) for i in inputs]
        + [slot for i in inputs for slot in extents_strides(i)]
        + [slot for i in outputs for slot in extents_strides(i)]
        + [pointer(i) for i in outputs]
    )


# ---------------------------------------------------------------------------
# PTX post-processing
# ---------------------------------------------------------------------------

_ELF_MAGIC = b"\x7fELF"

_ENTRY_RE = re.compile(
    r"(\.(?:visible|weak)\s+\.entry\s+([\w$]+)\s*\()([^)]*)(\))", re.DOTALL
)
_REQNTID_RE = re.compile(r"\.reqntid\s+(\d+)")
_PTX_VERSION_RE = re.compile(r"\.version\s+(\d+)\.(\d+)")
_BRACE_RE = re.compile(rb"[{}]")


def extract_ptx_from_cubin(cubin: bytes) -> Optional[str]:
    """Recover the PTX cuTile embeds in a CUBIN's debug section.

    TODO(upstream-cuda-tile): scraping a debug section is a workaround.
    ``export_kernel`` currently offers only ``output_format="cubin"`` /
    ``"tileir_bytecode"``; ask for a ``"ptx"`` format so both this and
    :func:`reorder_entry_params` can work on supported output instead of on
    bytes whose layout is an implementation detail.
    Tracking issue: <ADD CUDA-TILE ISSUE URL>.

    The compiler stores it as null-separated strings, so the text is located by
    its ``.version`` header and terminated at the brace matching the entry
    body's opening one. Brace *matching* rather than a plain search for ``}``
    matters: PTX vector-register syntax (``mov.b64 {%r1, %r2}, %rd0``) puts
    braces inside the body. Returns ``None`` if the section isn't there.
    """
    if len(cubin) < 64 or cubin[:4] != _ELF_MAGIC:
        return None
    start = cubin.find(b".version")
    if start < 0:
        return None
    open_brace = cubin.find(b"{", start)
    if open_brace < 0:
        return None

    # Walk only the braces, in C, rather than every byte of a multi-hundred-KB
    # CUBIN from Python: the latter costs an interpreter iteration per byte.
    depth = 0
    end = -1
    for brace in _BRACE_RE.finditer(cubin, open_brace):
        depth += 1 if brace.group() == b"{" else -1
        if depth == 0:
            end = brace.start()
            break
    if end < 0:
        return None

    text = cubin[start : end + 1].replace(b"\x00", b"\n").decode("utf-8", "replace")
    return "\n".join(line for line in text.splitlines() if line.strip()) + "\n"


ParsedEntry = Tuple[Optional["re.Match[str]"], str, List[str]]


def parse_entry(ptx: str) -> ParsedEntry:
    """``(match, kernel name, params)`` for the PTX ``.entry`` declaration."""
    match = _ENTRY_RE.search(ptx)
    if match is None:
        return None, "", []
    params = [p.strip() for p in match.group(3).split(",") if p.strip()]
    return match, match.group(2), params


def reorder_entry_params(
    ptx: str, order: Sequence[int], parsed: Optional[ParsedEntry] = None
) -> str:
    """Permute the ``.entry`` parameter declarations so slot ``i`` holds ``order[i]``.

    Only the declaration list is rewritten; the body keeps referring to each
    parameter by its own name, so moving the declarations is what changes which
    incoming argument each name binds to. ``parsed`` reuses an earlier
    :func:`parse_entry` result rather than re-scanning the whole module.
    """
    if parsed is None:
        parsed = parse_entry(ptx)
    match, _name, params = parsed
    if match is None:
        raise RuntimeError(
            "cuTile PTX has no '.entry' declaration to reorder; the compiled "
            "kernel cannot be wired to TensorRT's AOT launch."
        )
    if len(params) != len(order):
        raise RuntimeError(
            f"cuTile PTX entry declares {len(params)} parameter(s) but the "
            f"reorder expects {len(order)}."
        )
    reordered = ",\n\t".join(params[i] for i in order)
    return (
        ptx[: match.start()]
        + match.group(1)
        + "\n\t"
        + reordered
        + "\n"
        + match.group(4)
        + ptx[match.end() :]
    )


def parse_reqntid(ptx: str) -> Optional[int]:
    """The ``.reqntid`` (required threads per CTA) a cuTile kernel declares.

    cuTile vectorizes (e.g. ``f32x2``), so the thread count is often smaller
    than the tile size; the kernel must be launched with exactly this many
    threads or it traps.
    """
    match = _REQNTID_RE.search(ptx)
    return int(match.group(1)) if match is not None else None


def parse_ptx_version(ptx: str) -> Optional[int]:
    """``.version 9.3`` -> ``93``, the encoding used for ISA comparisons."""
    match = _PTX_VERSION_RE.search(ptx)
    if match is None:
        return None
    return int(match.group(1)) * 10 + int(match.group(2))


def cap_ptx_version(ptx: str, max_version: int) -> str:
    """Lower the ``.version`` header to ``max_version`` if it exceeds it.

    Only reached when a caller passes ``max_ptx_version=``; nothing lowers a
    header on its own. See :func:`verify_driver_accepts_ptx` for why.
    """
    emitted = parse_ptx_version(ptx)
    if emitted is None or emitted <= max_version:
        return ptx
    return set_ptx_version(ptx, max_version)


def set_ptx_version(ptx: str, version: int) -> str:
    """Rewrite the ``.version`` header, e.g. ``90`` -> ``.version 9.0``."""
    match = _PTX_VERSION_RE.search(ptx)
    if match is None:
        return ptx
    replacement = f".version {version // 10}.{version % 10}"
    return ptx[: match.start()] + replacement + ptx[match.end() :]


# A minimal well-formed module used only to ask the driver whether it accepts a
# given ISA. It declares no parameters and does nothing, so a rejection can only
# come from the ``.version`` header.
# A minimal valid module, used only to ask which ISA the driver would accept
# when reporting a mismatch.
_PROBE_PTX = (
    "//\n.version {major}.{minor}\n.target sm_50\n.address_size 64\n"
    ".visible .entry _ttk_ptx_probe()\n{{\n\tret;\n}}\n"
)


def _load_ptx(ptx: str) -> Any:
    """Ask the driver to JIT the module; returns the ``CUresult``."""
    from cuda.bindings import driver as cuda

    # cuModuleLoadData needs a current context; touching the device makes
    # PyTorch create the primary one for us.
    torch.cuda.init()
    torch.zeros(1, device="cuda")

    err, module = cuda.cuModuleLoadData(ptx.encode("utf-8"))
    if err == cuda.CUresult.CUDA_SUCCESS:
        cuda.cuModuleUnload(module)
    return err


# Bounds for the ISA search below. The ceiling only has to stay ahead of what
# any toolchain emits; reaching the floor means the driver could not be asked
# rather than that every ISA is too new.
_PTX_VERSION_CEILING = 129  # 12.9
_PTX_VERSION_FLOOR = 70  # 7.0


def driver_loads_ptx(ptx: str) -> bool:
    """True if the running driver JITs this module."""
    from cuda.bindings import driver as cuda

    return bool(_load_ptx(ptx) == cuda.CUresult.CUDA_SUCCESS)


def driver_max_ptx_version(below: int = _PTX_VERSION_CEILING) -> Optional[int]:
    """The newest ISA at or below ``below`` this driver loads.

    Walks down one step at a time rather than bisecting: the numbering has gaps
    (7.8 then 8.0, 8.8 then 9.0), so a rejected probe can mean "no such version"
    rather than "too new". Nothing on the success path calls this -- it exists
    to name a concrete remedy when a kernel's ISA is refused.
    """
    for version in range(below, _PTX_VERSION_FLOOR - 1, -1):
        if driver_loads_ptx(_PROBE_PTX.format(major=version // 10, minor=version % 10)):
            return version
    return None


def verify_driver_accepts_ptx(op_name: str, kernel_name: str, ptx: str) -> None:
    """Raise unless the running driver will load the PTX about to be embedded.

    ``tileiras`` emits the ISA of the toolkit it was built against, which can be
    newer than the installed driver accepts. TensorRT loads embedded PTX lazily,
    so a module the driver refuses does not surface at build time -- it appears
    much later as an opaque ``onShapeChange status -1`` from the engine. Loading
    it here turns that into an error at registration, next to its cause.

    The mismatch is reported rather than patched around. Lowering the
    ``.version`` header would sometimes work, but it is a text substitution over
    a body the compiler emitted for a different ISA: whether it survives depends
    on which instructions that body happens to contain, so it silently turns a
    clear environment problem into a kernel that may or may not assemble. The
    real fix is to align the driver with the cuda-tile toolchain, which the
    error says. ``max_ptx_version=`` remains for callers who have established
    that a lower header is safe for their kernel.
    """
    from cuda.bindings import driver as cuda

    try:
        err = _load_ptx(ptx)
    except Exception as exc:  # pragma: no cover - environment dependent
        _LOGGER.warning(
            "Could not verify that the driver accepts the PTX for '%s' (%s); "
            "embedding it unchecked. If the engine later fails at onShapeChange, "
            "this is why.",
            op_name,
            exc,
        )
        return

    if err == cuda.CUresult.CUDA_SUCCESS:
        return

    reason = str(err).split(".")[-1].split(":")[0]
    emitted = parse_ptx_version(ptx)
    if err == cuda.CUresult.CUDA_ERROR_UNSUPPORTED_PTX_VERSION and emitted is not None:
        accepted = driver_max_ptx_version(emitted - 1)
        remedy = (
            "Update the CUDA driver, or install a cuda-tile built against a "
            "toolkit it supports."
        )
        if accepted is not None:
            remedy += (
                f" This driver loads at most PTX ISA {accepted // 10}."
                f"{accepted % 10}; if you have established that ISA is safe for "
                f"this kernel, pass max_ptx_version={accepted} to cutile_op to "
                "set the header explicitly."
            )
        detail = (
            f"cuda-tile compiled it to PTX ISA {emitted // 10}.{emitted % 10}, "
            f"which this CUDA driver is too old to load. {remedy}"
        )
    else:
        detail = (
            "The PTX itself was rejected, so this is not merely a version gap; "
            "the compiled kernel is unusable as emitted."
        )

    raise RuntimeError(
        f"cutile_op '{op_name}': the CUDA driver refuses the PTX compiled for "
        f"kernel '{kernel_name}' ({reason}). {detail} Embedding it anyway would "
        "fail later inside the engine as an opaque 'onShapeChange status -1'."
    )


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def _cutile_import() -> Any:
    """Import ``cuda.tile``, raising an actionable error if it is unavailable."""
    try:
        import cuda.tile as ct

        return ct
    except ImportError as exc:
        raise ImportError(
            "cuda-tile is required for cutile_op plugins. "
            "Install it with: pip install cuda-tile"
        ) from exc


def _cutile_dtype(ct: Any, dtype: torch.dtype) -> Any:
    """The ``cuda.tile`` dtype object matching a torch dtype.

    Both spell the canonical names identically, so the torch name is also the
    attribute name; looking it up with ``getattr`` means a dtype this cuda-tile
    build does not expose raises here rather than deep inside the compiler.
    """
    value = getattr(ct, _dtype_name(dtype), None)
    if value is None:
        raise ValueError(f"cuTile has no dtype corresponding to {dtype}.")
    return value


def _default_arch() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


def compile_cutile_to_ptx(
    op_name: str,
    kernel: Any,
    layout: SignatureLayout,
    constants: Dict[str, Any],
    arch_override: Optional[str] = None,
    max_ptx_version: Optional[int] = None,
) -> Tuple[bytes, str, Optional[int]]:
    """Compile a cuTile kernel to TRT-ready PTX.

    Args:
        op_name: the op being registered, used only in error messages.
        kernel: the ``@ct.kernel`` program object.
        layout: the validated signature split into input and output arrays.
        constants: ``ct.Constant`` parameter values, in declaration order,
            baked into the compiled symbol.
        arch_override: target architecture (e.g. ``"sm_90"``). Defaults to the
            current device's compute capability.
        max_ptx_version: ISA ceiling as a ``93``-style int, lowering the
            ``.version`` header when the compiler emits something newer. Omit
            it: by default the emitted ISA is left alone and a driver that
            cannot load it is reported as an error.

    Returns:
        ``(ptx_bytes, kernel_name, reqntid)`` — the reordered PTX to embed in
        the engine, the entry symbol inside it, and the thread count the kernel
        requires (``None`` if it declares none).
    """
    ct = _cutile_import()
    try:
        from cuda.tile.compilation import (
            ArrayConstraint,
            CallingConvention,
            ConstantConstraint,
            KernelSignature,
            export_kernel,
        )
    except ImportError as exc:
        raise ImportError(
            f"cutile_op '{op_name}' needs the cuda.tile.compilation API to "
            "compile ahead of time; this cuda-tile build does not expose it."
        ) from exc

    # tileiras ships with cuda-tile but lives in the package's bin directory,
    # which is not on PATH by default. Say so here rather than let export_kernel
    # fail with a bare FileNotFoundError.
    if shutil.which("tileiras") is None:
        raise RuntimeError(
            f"cutile_op '{op_name}': the 'tileiras' compiler was not found on "
            "PATH. It ships with cuda-tile; add the package's bin directory "
            "(e.g. <site-packages>/nvidia/cu13/bin) to PATH before registering "
            "cuTile kernels."
        )

    parameters: List[Any] = [
        ArrayConstraint(
            dtype=_cutile_dtype(ct, param.dtype),
            ndim=param.ndim,
            index_dtype=ct.int32,
            # cuTile rejects negative strides by default; TRT only ever hands
            # the plugin non-negative ones.
            stride_lower_bound_incl=0,
            alias_groups=(),
            may_alias_internally=False,
        )
        for param in layout.arrays
    ]
    parameters.extend(ConstantConstraint(int(v)) for v in constants.values())

    signature = KernelSignature(
        parameters=tuple(parameters),
        calling_convention=CallingConvention.cutile_python_v1(),
        symbol=None,
    )

    buffer = io.BytesIO()
    export_kernel(
        kernel,
        [signature],
        buffer,
        gpu_code=arch_override or _default_arch(),
        output_format="cubin",
    )
    cubin = buffer.getvalue()

    ptx = extract_ptx_from_cubin(cubin)
    if ptx is None:
        raise RuntimeError(
            f"cutile_op '{op_name}': could not recover PTX from the compiled "
            "CUBIN. The AOT plugin path needs PTX text to reorder the kernel's "
            "parameters into TensorRT's launch order."
        )

    parsed = parse_entry(ptx)
    _, kernel_name, params = parsed
    if not kernel_name:
        raise RuntimeError(
            f"cutile_op '{op_name}': the compiled PTX has no '.entry' "
            "declaration, so its parameters cannot be matched to TensorRT's "
            "launch order."
        )

    order = cutile_param_order(layout)
    if len(params) != len(order):
        described = ", ".join(f"{p.name} (rank {p.ndim})" for p in layout.arrays)
        raise RuntimeError(
            f"cutile_op '{op_name}': kernel '{kernel_name}' compiled to "
            f"{len(params)} PTX parameter(s) but the signature describes "
            f"{len(order)} — {described}, each contributing one pointer plus one "
            f"extent and one stride per dimension. "
            f"{_diagnose_param_count(layout, len(params))}"
        )

    ptx = reorder_entry_params(ptx, order, parsed)
    if max_ptx_version is not None:
        ptx = cap_ptx_version(ptx, max_ptx_version)
    if arch_override is None:
        # Only meaningful when the PTX targets the device we can load it on;
        # a deliberate cross-compile is the caller's to verify.
        verify_driver_accepts_ptx(op_name, kernel_name, ptx)

    reqntid = parse_reqntid(ptx)
    _LOGGER.debug(
        "Compiled cuTile kernel '%s' -> PTX (%d bytes, reqntid=%s)",
        kernel_name,
        len(ptx),
        reqntid,
    )
    return ptx.encode("utf-8"), kernel_name, reqntid


def _diagnose_param_count(layout: SignatureLayout, actual: int) -> str:
    """Suggest what a mismatched PTX parameter count most likely means."""
    num_arrays = len(layout.arrays)
    if num_arrays and actual % num_arrays == 0:
        per_array = actual // num_arrays
        if per_array >= 3 and per_array % 2 == 1:
            return (
                f"The kernel looks like it was compiled for rank "
                f"{(per_array - 1) // 2} arrays; pass ndim="
                f"{(per_array - 1) // 2} (or a '<dtype>[rank]' signature entry)."
            )
    if actual > layout.num_slots:
        return (
            "The extra parameters are most likely runtime scalars, which the "
            "AOT QDP launch path cannot supply. Annotate them as "
            "ct.Constant[int] and pass their values in constants=."
        )
    return "Check the kernel's array parameters against the signature."


# ---------------------------------------------------------------------------
# AOT launch
# ---------------------------------------------------------------------------


def _trtp() -> Any:
    """The ``tensorrt.plugin`` module, resolved lazily.

    Indirected through a function rather than imported at module scope so the
    PTX and signature helpers above stay importable without a QDP-capable
    TensorRT, and so tests can substitute a stub for the symbolic-expression
    types, which only work inside a live plugin's expression builder.
    """
    import tensorrt.plugin as trtp

    return trtp


def _as_symint32(value: Any) -> Any:
    trtp = _trtp()
    if isinstance(value, trtp.SymInt32):
        return value
    return trtp.SymInt32(value)


def _extents_and_strides(desc: Any, param: ArrayParam) -> List[Any]:
    """The ``(extents..., strides...)`` a cuTile array parameter expects.

    Rank 1 is the flattened view a 1-D cuTile kernel is written against, so its
    single extent is the tensor's element count regardless of how many
    dimensions the tensor has. Higher ranks map dimension for dimension onto
    the tensor's own shape, with row-major strides.
    """
    trtp = _trtp()

    shape = desc.shape_expr
    if param.ndim == 1:
        return [_as_symint32(shape.numel()), trtp.SymInt32(1)]

    dims = list(shape)
    if len(dims) != param.ndim:
        raise ValueError(
            f"cuTile array '{param.name}' is compiled for rank {param.ndim} but "
            f"received a rank-{len(dims)} tensor. Register with "
            f"ndim={len(dims)}, or reshape the tensor before the op."
        )

    # Row-major strides are the suffix products of the shape, so accumulate
    # once from the right instead of rebuilding each product from scratch.
    strides = [trtp.SymInt32(1)]
    for dim in reversed(dims[1:]):
        strides.append(_as_symint32(strides[-1] * _as_symint32(dim)))
    return [_as_symint32(d) for d in dims] + strides[::-1]


def build_extra_args(
    inputs: Sequence[Any], outputs: Sequence[Any], layout: SignatureLayout
) -> Any:
    """Build the ``SymIntExprs`` TensorRT passes between the in and out pointers.

    The order is every input array's extents and strides, then every output
    array's — exactly the slots :func:`cutile_param_order` routes them into.

    Raises:
        RuntimeError: if TensorRT hands over a different number of tensors than
            the signature describes. Registration validates the two agree, so
            this is a last line of defense — but it is the one place where a
            disagreement would go undetected: zipping the shorter of the two
            would quietly emit too few extra arguments and leave the kernel
            reading whatever occupied the unfilled parameter slots.
    """
    trtp = _trtp()

    for kind, descs, params in (
        ("input", inputs, layout.inputs),
        ("output", outputs, layout.outputs),
    ):
        if len(descs) != len(params):
            raise RuntimeError(
                f"cuTile launch received {len(descs)} {kind} tensor(s) but the "
                f"signature describes {len(params)} "
                f"({', '.join(p.name for p in params)}). The extra arguments "
                "would not line up with the kernel's parameters."
            )

    values: List[Any] = []
    for desc, param in zip(inputs, layout.inputs):
        values.extend(_extents_and_strides(desc, param))
    for desc, param in zip(outputs, layout.outputs):
        values.extend(_extents_and_strides(desc, param))

    extra_args = trtp.SymIntExprs(len(values))
    for index, value in enumerate(values):
        extra_args[index] = value
    return extra_args


def resolve_block_threads(
    op_name: str,
    kernel_name: str,
    reqntid: Optional[int],
    block_size: Optional[int],
) -> int:
    """The threads-per-block the compiled kernel must be launched with.

    ``.reqntid`` is a requirement, not a hint: cuTile vectorizes, so the thread
    count is usually below the tile size, and any other count traps.
    """
    if reqntid is None:
        if block_size is None:
            raise ValueError(
                f"cutile_op '{op_name}': kernel '{kernel_name}' declares no "
                ".reqntid, so the threads-per-block cannot be derived. Pass "
                "block_size= explicitly."
            )
        return block_size
    if block_size is not None and block_size != reqntid:
        raise ValueError(
            f"cutile_op '{op_name}' was given block_size={block_size} but kernel "
            f"'{kernel_name}' declares .reqntid {reqntid}, which must be the "
            "launch's threads-per-block. Drop block_size."
        )
    return reqntid


def make_aot_fn(
    op_name: str,
    layout: SignatureLayout,
    grid: Callable[..., Any],
    block_threads: int,
) -> Callable[..., Any]:
    """Derive the AOT launch function from the user's ``grid`` and the layout."""

    def _aot_fn(inputs: Any, outputs: Any, tactic: int) -> Any:
        trtp = _trtp()

        dims = grid(inputs, outputs)
        if not isinstance(dims, (tuple, list)):
            dims = (dims,)
        if not 1 <= len(dims) <= 3:
            raise ValueError(
                f"cutile_op '{op_name}' grid returned {len(dims)} dimension(s); "
                "TensorRT launches accept 1 to 3 (grid_x, grid_y, grid_z)."
            )

        launch_params = trtp.KernelLaunchParams()
        launch_params.grid_x = dims[0]
        if len(dims) > 1:
            launch_params.grid_y = dims[1]
        if len(dims) > 2:
            launch_params.grid_z = dims[2]
        launch_params.block_x = block_threads
        launch_params.shared_mem = 0

        return launch_params, build_extra_args(inputs, outputs, layout)

    return _aot_fn
