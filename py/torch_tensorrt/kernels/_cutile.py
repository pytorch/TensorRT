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

import inspect
import io
import keyword
import logging
import re
import struct
from math import prod
from typing import Any, Callable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

import torch

_LOGGER = logging.getLogger(__name__)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_SCHEMA_NAMES = frozenset({"outputs", "stream", "tactic"})

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


class TensorSchema(NamedTuple):
    """Validated Tensor-only Torch schema used by the cuTile launch path."""

    schema: str
    input_names: Tuple[str, ...]
    num_inputs: int
    num_outputs: int


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


def analyze_tensor_schema(
    op_name: str, meta_fn: Callable[..., Any], schema: Optional[str]
) -> TensorSchema:
    """Infer and validate the Tensor-only schema supported by ``cutile_op``.

    cuTile scalar parameters are compile-time constants. TensorRT's AOT wrapper
    cannot forward runtime Torch scalar arguments to the kernel, so accepting
    one would register an operator whose runtime value is silently ignored.
    """
    parts = op_name.split("::") if isinstance(op_name, str) else []
    if len(parts) != 2 or any(
        not _IDENTIFIER_RE.fullmatch(part) or keyword.iskeyword(part) for part in parts
    ):
        raise ValueError(
            "cutile_op op_name must be a qualified pair of Python identifiers "
            f"such as 'my_ops::add_one'; got {op_name!r}."
        )
    if not callable(meta_fn):
        raise ValueError(f"cutile_op '{op_name}' meta_fn must be callable.")

    inferred_schema = None
    try:
        inferred_schema = torch.library.infer_schema(meta_fn, mutates_args=())
    except Exception:
        pass
    if schema is None:
        if inferred_schema is None:
            raise ValueError(
                f"cutile_op '{op_name}' could not infer a complete schema from "
                "meta_fn. Add Tensor argument and return annotations, or pass "
                'schema= explicitly (for example, "(Tensor x) -> Tensor").'
            )
        schema = inferred_schema

    if not isinstance(schema, str) or not schema.strip().startswith("("):
        raise ValueError(
            f"cutile_op '{op_name}' schema must be an operator schema suffix "
            f"such as '(Tensor x) -> Tensor'; got {schema!r}."
        )
    schema = schema.strip()
    try:
        parsed = torch._C.parse_schema(f"{op_name}{schema}")
    except Exception as exc:
        raise ValueError(
            f"cutile_op '{op_name}' received an invalid schema {schema!r}."
        ) from exc

    tensor_type = torch._C.TensorType.get()
    non_tensor_args = [
        f"{arg.name}: {arg.type}"
        for arg in parsed.arguments
        if not arg.type.isSubtypeOf(tensor_type)
    ]
    non_tensor_returns = [
        str(ret.type) for ret in parsed.returns if not ret.type.isSubtypeOf(tensor_type)
    ]
    if non_tensor_args:
        raise ValueError(
            f"cutile_op '{op_name}' supports Tensor inputs only; runtime scalar "
            f"arguments are not forwarded to the compiled kernel "
            f"({', '.join(non_tensor_args)}). Put scalar values in constants=."
        )
    if any(arg.kwarg_only for arg in parsed.arguments):
        raise ValueError(
            f"cutile_op '{op_name}' does not support keyword-only Tensor inputs."
        )
    reserved = [
        arg.name for arg in parsed.arguments if arg.name in _RESERVED_SCHEMA_NAMES
    ]
    if reserved:
        raise ValueError(
            f"cutile_op '{op_name}' schema uses reserved parameter names "
            f"{reserved}; choose names other than outputs, stream, or tactic."
        )
    if any(
        arg.alias_info is not None or arg.has_default_value()
        for arg in parsed.arguments
    ) or any(ret.alias_info is not None for ret in parsed.returns):
        raise ValueError(
            f"cutile_op '{op_name}' does not support aliased, mutable, or "
            "defaulted Tensor schemas."
        )
    if not parsed.arguments:
        raise ValueError(f"cutile_op '{op_name}' requires at least one Tensor input.")
    if not parsed.returns or non_tensor_returns:
        detail = ", ".join(non_tensor_returns) or "no outputs"
        raise ValueError(
            f"cutile_op '{op_name}' must return one or more Tensors; got {detail}."
        )

    meta_params = list(inspect.signature(meta_fn).parameters.values())
    schema_names = tuple(arg.name for arg in parsed.arguments)
    positional = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    if tuple(param.name for param in meta_params) != schema_names or any(
        param.kind not in positional or param.default is not inspect.Parameter.empty
        for param in meta_params
    ):
        raise ValueError(
            f"cutile_op '{op_name}' meta_fn parameters must exactly match the "
            f"schema's positional inputs {list(schema_names)}."
        )
    if inferred_schema is not None:
        inferred = torch._C.parse_schema(f"{op_name}{inferred_schema}")
        if len(inferred.returns) != len(parsed.returns):
            raise ValueError(
                f"cutile_op '{op_name}' schema declares {len(parsed.returns)} "
                f"outputs but meta_fn is annotated for {len(inferred.returns)}."
            )
    return TensorSchema(
        schema, schema_names, len(parsed.arguments), len(parsed.returns)
    )


def validate_kernel_parameters(
    op_name: str,
    kernel: Any,
    layout: SignatureLayout,
    constants: Mapping[str, Any],
) -> None:
    """Require the declarative parameter order to match the ``@ct.kernel``."""
    pyfunc = getattr(kernel, "_pyfunc", None)
    if not callable(pyfunc):
        raise ValueError(
            f"cutile_op '{op_name}' kernel must be a cuda.tile @ct.kernel object "
            "with inspectable parameter metadata."
        )
    try:
        declared = inspect.signature(pyfunc).parameters.values()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"cutile_op '{op_name}' kernel must be a cuda.tile @ct.kernel object "
            "with inspectable parameter metadata."
        ) from exc

    params = list(declared)
    expected = [param.name for param in layout.arrays] + list(constants)
    actual = [param.name for param in params]
    positional = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    if actual != expected or any(param.kind not in positional for param in params):
        raise ValueError(
            f"cutile_op '{op_name}' declares parameters {expected}, but the "
            f"@ct.kernel declares {actual}. signature= arrays must come first "
            "(inputs then outputs), followed by constants=, all in exact "
            "declaration order."
        )


def validate_cutile_config(
    op_name: str,
    signature: Mapping[str, Any],
    constants: Mapping[str, Any],
    arity: Tuple[int, int],
    default_ndim: int = 1,
    input_names: Optional[Sequence[str]] = None,
) -> SignatureLayout:
    """Check a ``cutile_op`` registration and return its signature layout.

    ``signature`` lists the kernel's array parameters in declaration order,
    inputs first then outputs; ``ct.Constant`` parameters are not part of it.
    ``arity`` is the ``(tensor inputs, outputs)`` of the op being registered and
    is what decides where the inputs end.

    Everything here is answerable before anything is compiled, and every rule,
    left unchecked, produces wrong numbers rather than an error.
    """
    if isinstance(default_ndim, bool) or not isinstance(default_ndim, int):
        raise ValueError(
            f"cutile_op '{op_name}' ndim must be an integer; got {default_ndim!r}."
        )
    if default_ndim < 1:
        raise ValueError(
            f"cutile_op '{op_name}' was given ndim={default_ndim}; "
            "cuTile arrays have rank >= 1."
        )

    if not isinstance(signature, Mapping) or not isinstance(constants, Mapping):
        raise ValueError(
            f"cutile_op '{op_name}' signature and constants must be mappings."
        )

    for kind, values in (("signature", signature), ("constants", constants)):
        invalid = [name for name in values if not isinstance(name, str) or not name]
        if invalid:
            raise ValueError(
                f"cutile_op '{op_name}' {kind} has invalid parameter names: {invalid}."
            )

    overlap = sorted(set(signature) & set(constants))
    if overlap:
        raise ValueError(
            f"cutile_op '{op_name}' declares {overlap} in both signature and "
            "constants. Array parameters belong in signature; ct.Constant "
            "parameters belong in constants."
        )

    for name, value in constants.items():
        if type(value) not in (bool, int, float):
            raise ValueError(
                f"cutile_op '{op_name}' constant '{name}' is {value!r}; cuTile "
                "ct.Constant values must be bool, int, or float."
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

    num_inputs, num_outputs = arity
    if num_inputs + num_outputs != len(params):
        raise ValueError(
            f"cutile_op '{op_name}' signature declares {len(params)} array "
            f"parameter(s) ({', '.join(p.name for p in params)}) but the op "
            f"takes {num_inputs} tensor input(s) and returns {num_outputs} "
            "output(s). The signature must list every input array followed "
            "by every output array."
        )

    layout = SignatureLayout(inputs=params[:num_inputs], outputs=params[num_inputs:])
    declared_inputs = tuple(param.name for param in layout.inputs)
    if input_names is not None and declared_inputs != tuple(input_names):
        raise ValueError(
            f"cutile_op '{op_name}' signature input order {list(declared_inputs)} "
            f"does not match the Torch schema input order {list(input_names)}. "
            "These names and their order must match so TensorRT binds each "
            "pointer to the intended kernel parameter."
        )
    return layout


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

    def _missing(kind: str, expected: int, actual: int) -> bool:
        _LOGGER.warning(
            "Not lowering '%s' to its cuTile plugin: expected metadata for %d "
            "%s tensor(s), got %d. Refusing an unchecked pointer binding.",
            op_name,
            expected,
            kind,
            actual,
        )
        return False

    def _validator(node: Any, settings: Any = None) -> bool:
        if user_validator is not None and not user_validator(node, settings):
            return False

        node_args = getattr(node, "args", ())
        if len(node_args) != len(expected_inputs):
            return _missing("input", len(expected_inputs), len(node_args))
        actual_inputs = [_tensor_meta(value) for value in node_args]
        if any(value is None for value in actual_inputs):
            return _missing(
                "input",
                len(expected_inputs),
                sum(value is not None for value in actual_inputs),
            )
        for index, (got, want) in enumerate(zip(actual_inputs, expected_inputs)):
            assert got is not None
            if got.dtype != want:
                return _mismatch("input", index, got, want)

        node_meta = getattr(node, "meta", None)
        produced = node_meta.get("val") if isinstance(node_meta, dict) else None
        actual_outputs = (
            list(produced) if isinstance(produced, (tuple, list)) else [produced]
        )
        if len(actual_outputs) != len(expected_outputs) or any(
            not isinstance(value, torch.Tensor) for value in actual_outputs
        ):
            return _missing(
                "output",
                len(expected_outputs),
                sum(isinstance(value, torch.Tensor) for value in actual_outputs),
            )
        for index, (got, want) in enumerate(zip(actual_outputs, expected_outputs)):
            assert isinstance(got, torch.Tensor)
            if got.dtype != want:
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
    r"((?:(?:\.(?:visible|weak))\s+)?\.entry\s+([\w$%]+)\s*\()([^)]*)(\))",
    re.DOTALL,
)
_PTX_INT = r"(?:0[xX][0-9a-fA-F]+|0[bB][01]+|0[0-7]*|[1-9]\d*)"
_REQNTID_RE = re.compile(
    rf"\.reqntid\s+({_PTX_INT})(?:\s*,\s*({_PTX_INT}))?"
    rf"(?:\s*,\s*({_PTX_INT}))?(?![A-Za-z0-9_])"
)
_UNSUPPORTED_LAUNCH_RE = re.compile(
    r"\.(?:maxntid|explicitcluster|reqnctapercluster|maxclusterrank)\b"
)
_PTX_VERSION_RE = re.compile(r"\.version\s+(\d+)\.(\d+)")
_PTX_COMMENT_RE = re.compile(r"//[^\r\n]*|/\*.*?\*/", re.DOTALL)


def _elf_section(cubin: bytes, wanted: bytes) -> Optional[bytes]:
    """Return one named ELF64 section, or ``None`` for malformed/ambiguous data."""
    if len(cubin) < 64 or cubin[:4] != _ELF_MAGIC or cubin[4] != 2:
        return None
    if cubin[5] == 1:
        byte_order = "<"
    elif cubin[5] == 2:
        byte_order = ">"
    else:
        return None

    try:
        section_offset = struct.unpack_from(f"{byte_order}Q", cubin, 40)[0]
        entry_size, count, names_index = struct.unpack_from(
            f"{byte_order}HHH", cubin, 58
        )
    except struct.error:
        return None
    if (
        entry_size < 64
        or count == 0
        or names_index >= count
        or section_offset > len(cubin)
        or count > (len(cubin) - section_offset) // entry_size
    ):
        return None

    def _header(index: int) -> Optional[Tuple[int, int, int]]:
        offset = section_offset + index * entry_size
        try:
            name = struct.unpack_from(f"{byte_order}I", cubin, offset)[0]
            data_offset, size = struct.unpack_from(
                f"{byte_order}QQ", cubin, offset + 24
            )
        except struct.error:
            return None
        if data_offset > len(cubin) or size > len(cubin) - data_offset:
            return None
        return name, data_offset, size

    names_header = _header(names_index)
    if names_header is None:
        return None
    _, names_offset, names_size = names_header
    names = cubin[names_offset : names_offset + names_size]

    matches = []
    for index in range(count):
        header = _header(index)
        if header is None:
            return None
        name_offset, data_offset, size = header
        if name_offset >= len(names):
            return None
        name_end = names.find(b"\x00", name_offset)
        if name_end < 0:
            return None
        if names[name_offset:name_end] == wanted:
            matches.append(cubin[data_offset : data_offset + size])
    return matches[0] if len(matches) == 1 else None


def extract_ptx_from_cubin(cubin: bytes) -> Optional[str]:
    """Recover the PTX cuTile embeds in a CUBIN's debug section.

    Reading the embedded debug text is a compatibility workaround.
    ``export_kernel`` currently offers only ``output_format="cubin"`` /
    ``"tileir_bytecode"``. A public ``"ptx"`` output would remove this need.
    Until then, reading the complete ``.nv_debug_ptx_txt`` ELF section avoids
    truncating module initializers or helper functions around the entry point.
    """
    section = _elf_section(cubin, b".nv_debug_ptx_txt")
    if section is None:
        return None
    start = section.find(b".version")
    if start < 0 or section.find(b".version", start + 1) >= 0:
        return None
    try:
        text = section[start:].rstrip(b"\x00").replace(b"\x00", b"\n").decode()
    except UnicodeDecodeError:
        return None
    return "\n".join(line for line in text.splitlines() if line.strip()) + "\n"


ParsedEntry = Tuple[Optional["re.Match[str]"], str, List[str]]


def _mask_ptx_comments(ptx: str) -> str:
    """Hide comments without changing offsets used to rewrite the source."""
    return _PTX_COMMENT_RE.sub(lambda match: " " * len(match.group()), ptx)


def parse_entry(ptx: str) -> ParsedEntry:
    """``(match, kernel name, params)`` for one unambiguous PTX entry."""
    matches = list(_ENTRY_RE.finditer(_mask_ptx_comments(ptx)))
    if len(matches) != 1:
        return None, "", []
    match = matches[0]
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


def parse_reqntid(ptx: str) -> Optional[Tuple[int, int, int]]:
    """The ``.reqntid`` (required threads per CTA) a cuTile kernel declares.

    cuTile vectorizes (e.g. ``f32x2``), so the thread count is often smaller
    than the tile size; the kernel must be launched with exactly this many
    threads or it traps.
    """
    uncommented = _mask_ptx_comments(ptx)
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


def unsupported_launch_directive(ptx: str) -> Optional[str]:
    """Return a PTX launch contract this frontend cannot represent, if any."""
    uncommented = _mask_ptx_comments(ptx)
    entry, _, _ = parse_entry(uncommented)
    if entry is None:
        return None
    body_start = uncommented.find("{", entry.end())
    if body_start < 0:
        return None
    match = _UNSUPPORTED_LAUNCH_RE.search(uncommented, entry.end(), body_start)
    return match.group() if match is not None else None


def parse_ptx_version(ptx: str) -> Optional[int]:
    """``.version 9.3`` -> ``93``, the encoding used for ISA comparisons."""
    match = _PTX_VERSION_RE.search(_mask_ptx_comments(ptx))
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
    match = _PTX_VERSION_RE.search(_mask_ptx_comments(ptx))
    if match is None:
        return ptx
    replacement = f".version {version // 10}.{version % 10}"
    return ptx[: match.start()] + replacement + ptx[match.end() :]


def _load_ptx(ptx: str, kernel_name: str) -> Any:
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


def verify_driver_accepts_ptx(op_name: str, kernel_name: str, ptx: str) -> None:
    """Fail unless the running driver JITs the exact PTX that will be embedded.

    ``tileiras`` emits the ISA of the toolkit it was built against, which can be
    newer than the installed driver loads. Nothing catches that on its own:
    TensorRT builds the engine happily, and at inference the plugin fails with
    ``onShapeChange status -1`` on stderr while ``enqueue`` still returns -- so
    the model silently produces garbage. Verifying here, against the same
    ``cuModuleLoadData`` the driver will use later, is what makes the mismatch
    visible at registration.

    Lowering a PTX header does not prove semantic compatibility, so this helper
    never rewrites it automatically. A caller may explicitly opt into a header
    ceiling with ``max_ptx_version``; that candidate is still verified here.
    """
    try:
        from cuda.bindings import driver as cuda
    except ImportError:  # cuda-python < 12.8
        from cuda import cuda

    try:
        err = _load_ptx(ptx, kernel_name)
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"cutile_op '{op_name}' could not verify the PTX for kernel "
            f"'{kernel_name}' with the CUDA driver; refusing to embed unchecked "
            "code. Ensure CUDA is initialized and cuda-python is installed."
        ) from exc

    if err == cuda.CUresult.CUDA_SUCCESS:
        return

    error_name = str(err).split(".")[-1].split(":")[0]
    emitted = parse_ptx_version(ptx)
    version = (
        f" PTX ISA {emitted // 10}.{emitted % 10}" if emitted is not None else " PTX"
    )
    raise RuntimeError(
        f"cutile_op '{op_name}': the CUDA driver could not load{version} and "
        f"resolve kernel '{kernel_name}' ({error_name}). Align the CUDA driver "
        "with the cuda-tile toolchain and ensure the compiled entry is present. "
        "If you have independently verified compatibility, max_ptx_version= can "
        "explicitly cap the header before this check."
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
            "Install it with: pip install 'cuda-tile>=1.3,<2'"
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


def _should_verify_ptx(arch_override: Optional[str]) -> bool:
    """Whether the compiled target can be meaningfully loaded on this device."""
    if arch_override is None:
        return True
    return torch.cuda.is_available() and arch_override == _default_arch()


def compile_cutile_to_ptx(
    op_name: str,
    kernel: Any,
    layout: SignatureLayout,
    constants: Mapping[str, Any],
    arch_override: Optional[str] = None,
    max_ptx_version: Optional[int] = None,
) -> Tuple[bytes, str, Optional[Tuple[int, int, int]]]:
    """Compile a cuTile kernel to TRT-ready PTX.

    Args:
        op_name: the op being registered, used only in error messages.
        kernel: the ``@ct.kernel`` program object.
        layout: the validated signature split into input and output arrays.
        constants: ``ct.Constant`` parameter values, in declaration order,
            baked into the compiled symbol.
        arch_override: target architecture (e.g. ``"sm_90"``). Defaults to the
            current device's compute capability.
        max_ptx_version: Explicit ISA ceiling as a ``90``-style int, pinning the
            ``.version`` header. By default the emitted PTX is never rewritten.

    Returns:
        ``(ptx_bytes, kernel_name, reqntid)`` — the reordered PTX to embed in
        the engine, the entry symbol inside it, and the three-dimensional block
        size the kernel requires (``None`` if it declares none).
    """
    validate_kernel_parameters(op_name, kernel, layout, constants)
    if max_ptx_version is not None and (
        isinstance(max_ptx_version, bool)
        or not isinstance(max_ptx_version, int)
        or max_ptx_version < 10
        or max_ptx_version > 999
    ):
        raise ValueError(
            f"cutile_op '{op_name}' max_ptx_version must be an integer encoded "
            f"like 90 for PTX 9.0; got {max_ptx_version!r}."
        )

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
            "compile ahead of time. Install 'cuda-tile>=1.3,<2'."
        ) from exc

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
    parameters.extend(ConstantConstraint(value) for value in constants.values())

    signature = KernelSignature(
        parameters=tuple(parameters),
        calling_convention=CallingConvention.cutile_python_v1(),
        symbol=None,
    )

    buffer = io.BytesIO()
    try:
        export_kernel(
            kernel,
            [signature],
            buffer,
            gpu_code=arch_override or _default_arch(),
            output_format="cubin",
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"cutile_op '{op_name}' could not find the 'tileiras' compiler. "
            "Expose tileiras from a CUDA Toolkit compatible with the installed "
            "cuda-tile through "
            "PATH or CUDA_HOME. In an environment without conflicting pinned "
            "CUDA packages, 'cuda-tile[tileiras]>=1.3,<2' is an alternative."
        ) from exc
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
        entry_count = len(list(_ENTRY_RE.finditer(ptx)))
        raise RuntimeError(
            f"cutile_op '{op_name}': expected exactly one compiled PTX '.entry' "
            f"declaration, found {entry_count}; its parameters cannot be matched "
            "unambiguously to TensorRT's launch order."
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
    if _should_verify_ptx(arch_override):
        # Only meaningful when the PTX targets the device we can load it on;
        # a deliberate cross-compile is the caller's to verify on its target.
        verify_driver_accepts_ptx(op_name, kernel_name, ptx)

    unsupported = unsupported_launch_directive(ptx)
    if unsupported is not None:
        raise RuntimeError(
            f"cutile_op '{op_name}': kernel '{kernel_name}' declares PTX launch "
            f"directive {unsupported}, which this frontend cannot represent. "
            "Refusing a launch whose requirements would be silently ignored."
        )
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


def _normalize_block_dims(op_name: str, block_size: Any) -> Tuple[int, int, int]:
    """Validate a static CUDA block size and normalize it to three dimensions."""
    if isinstance(block_size, bool):
        dims: Tuple[Any, ...] = (block_size,)
    elif isinstance(block_size, int):
        dims = (block_size,)
    elif isinstance(block_size, Sequence) and not isinstance(
        block_size, (str, bytes, bytearray)
    ):
        dims = tuple(block_size)
    else:
        dims = (block_size,)

    if not 1 <= len(dims) <= 3 or any(
        isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in dims
    ):
        raise ValueError(
            f"cutile_op '{op_name}' block_size must contain 1 to 3 positive "
            f"integers; got {block_size!r}."
        )
    padded = tuple(dims) + (1,) * (3 - len(dims))
    limits = (1024, 1024, 64)
    if any(dim > limit for dim, limit in zip(padded, limits)) or prod(padded) > 1024:
        raise ValueError(
            f"cutile_op '{op_name}' block_size {block_size!r} exceeds CUDA's "
            "per-dimension or 1024 threads-per-block limit."
        )
    return int(padded[0]), int(padded[1]), int(padded[2])


def resolve_block_dims(
    op_name: str,
    kernel_name: str,
    reqntid: Optional[Tuple[int, int, int]],
    block_size: Any,
) -> Tuple[int, int, int]:
    """The block dimensions the compiled kernel must be launched with.

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
        return _normalize_block_dims(op_name, block_size)

    required = _normalize_block_dims(op_name, reqntid)
    requested = (
        None if block_size is None else _normalize_block_dims(op_name, block_size)
    )
    if requested is not None and requested != required:
        raise ValueError(
            f"cutile_op '{op_name}' was given block_size={block_size} but kernel "
            f"'{kernel_name}' declares .reqntid {reqntid}, which must match the "
            "launch block exactly. Drop block_size."
        )
    return required


def _launch_dim_constant(value: Any) -> Optional[int]:
    """Return a concrete launch value, or ``None`` for a symbolic expression."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        is_constant = value.is_constant
        if callable(is_constant):  # raw ``trt.IDimensionExpr`` API
            is_constant = is_constant()
        if is_constant:
            getter = getattr(value, "constant_value", None)
            getter = getter if getter is not None else value.get_constant_value
            return int(getter() if callable(getter) else getter)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        # Some TensorRT QDP builds have a public accessor that references an
        # uninitialized ``_is_dummy`` field. The wrapped IDimensionExpr is still
        # populated inside the AOT expression-builder context, so use its
        # public constant query as a compatibility fallback.
        try:
            expr = value._expr
            if expr is not None and expr.is_constant():
                return int(expr.get_constant_value())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    return None


def validate_launch_grid(op_name: str, value: Any) -> Tuple[Any, ...]:
    """Validate a concrete or symbolic CUDA launch grid."""
    dims = tuple(value) if isinstance(value, (tuple, list)) else (value,)
    if not 1 <= len(dims) <= 3:
        raise ValueError(
            f"cutile_op '{op_name}' grid returned {len(dims)} dimension(s); "
            "TensorRT launches accept 1 to 3."
        )

    trtp = _trtp()
    import tensorrt as trt

    symint32 = getattr(trtp, "SymInt32", None)
    symint_expr = getattr(symint32, "__base__", None)
    symbolic_types = tuple(
        cls
        for cls in (symint32, symint_expr, getattr(trt, "IDimensionExpr", None))
        if isinstance(cls, type) and cls is not object
    )
    limits = (2**31 - 1, 65535, 65535)
    for index, (dim, limit) in enumerate(zip(dims, limits)):
        constant = _launch_dim_constant(dim)
        if isinstance(dim, bool):
            valid = False
        elif constant is not None:
            valid = 1 <= constant <= limit
        else:
            valid = bool(symbolic_types) and isinstance(dim, symbolic_types)
        if not valid:
            try:
                shown = repr(dim) if constant is None else repr(constant)
            except Exception:  # pragma: no cover - defensive for TRT wrappers
                shown = type(dim).__name__
            raise ValueError(
                f"cutile_op '{op_name}' grid dimension {index} must be a "
                f"positive integer <= {limit} or symbolic TensorRT dimension; "
                f"got {shown}."
            )
    return dims


def _static_launch_dim(op_name: str, field: str, value: Any) -> int:
    """Read a build-time-constant launch dimension from TRT's symbolic type."""
    constant = _launch_dim_constant(value)
    if constant is None:
        raise ValueError(
            f"cutile_op '{op_name}' custom aot_fn must set {field} to a "
            "build-time integer."
        )
    return constant


def make_checked_aot_fn(
    op_name: str,
    kernel_name: str,
    layout: SignatureLayout,
    reqntid: Optional[Tuple[int, int, int]],
    aot_fn: Callable[..., Any],
) -> Callable[..., Any]:
    """Wrap a custom AOT launch with the invariants the derived path enforces."""
    expected_extras = layout.num_slots - len(layout.arrays)

    def _checked_aot_fn(inputs: Any, outputs: Any, tactic: int) -> Any:
        result = aot_fn(inputs, outputs, tactic)
        if not isinstance(result, tuple) or len(result) != 2:
            raise ValueError(
                f"cutile_op '{op_name}' custom aot_fn must return "
                "(KernelLaunchParams, extra_args)."
            )
        launch_params, extra_args = result
        trtp = _trtp()
        if not isinstance(launch_params, trtp.KernelLaunchParams):
            raise ValueError(
                f"cutile_op '{op_name}' custom aot_fn must return a "
                "TensorRT KernelLaunchParams as its first value."
            )

        try:
            grid = tuple(
                getattr(launch_params, field)
                for field in ("grid_x", "grid_y", "grid_z")
            )
            block = tuple(
                _static_launch_dim(op_name, field, getattr(launch_params, field))
                for field in ("block_x", "block_y", "block_z")
            )
            shared_mem = _static_launch_dim(
                op_name, "shared_mem", launch_params.shared_mem
            )
        except AttributeError as exc:
            raise ValueError(
                f"cutile_op '{op_name}' custom aot_fn returned incomplete "
                "KernelLaunchParams."
            ) from exc

        validate_launch_grid(op_name, grid)
        actual_block = _normalize_block_dims(op_name, block)
        if shared_mem != 0:
            raise ValueError(
                f"cutile_op '{op_name}' custom aot_fn set shared_mem to "
                f"{shared_mem}; cutile_op does not support caller-supplied "
                "dynamic shared memory, so it must be 0."
            )
        if reqntid is not None and actual_block != reqntid:
            raise ValueError(
                f"cutile_op '{op_name}' custom aot_fn launches block "
                f"{actual_block}, but kernel '{kernel_name}' declares .reqntid "
                f"{reqntid}, which must match exactly."
            )

        if not isinstance(extra_args, trtp.SymIntExprs):
            raise ValueError(
                f"cutile_op '{op_name}' custom aot_fn must return TensorRT "
                "SymIntExprs as its second value."
            )
        actual_extras = len(extra_args)
        if actual_extras != expected_extras:
            raise ValueError(
                f"cutile_op '{op_name}' custom aot_fn returned {actual_extras} "
                f"extra argument(s), but the signature requires {expected_extras} "
                "extents and strides."
            )
        if any(not isinstance(value, trtp.SymInt32) for value in extra_args):
            raise ValueError(
                f"cutile_op '{op_name}' custom aot_fn extra_args must contain "
                "only TensorRT SymInt32 values."
            )
        return result

    return _checked_aot_fn


def make_aot_fn(
    op_name: str,
    layout: SignatureLayout,
    grid: Callable[..., Any],
    block_dims: Tuple[int, int, int],
) -> Callable[..., Any]:
    """Derive the AOT launch function from the user's ``grid`` and the layout."""

    def _aot_fn(inputs: Any, outputs: Any, tactic: int) -> Any:
        trtp = _trtp()

        dims = validate_launch_grid(op_name, grid(inputs, outputs))

        launch_params = trtp.KernelLaunchParams()
        launch_params.grid_x = dims[0]
        launch_params.grid_y = dims[1] if len(dims) > 1 else 1
        launch_params.grid_z = dims[2] if len(dims) > 2 else 1
        launch_params.block_x, launch_params.block_y, launch_params.block_z = block_dims
        launch_params.shared_mem = 0

        return launch_params, build_extra_args(inputs, outputs, layout)

    return _aot_fn
