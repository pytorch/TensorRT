# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile from a checkpoint's weights in place: zero-copy, memory-mapped model weights.

``map_weights`` points a module's parameters and buffers (or an ``ExportedProgram``'s state) at views of a
safetensors file. The file is mapped copy-on-write and never written; a weight's pages become resident
only when something reads them. During compilation that is TensorRT, while it builds the engine of the
partition that uses the weight: TensorRT takes weights as host pointers, and the converter passes a CPU
tensor's own data pointer, so a mapped model compiles without a host copy of its weights.

A mapped weight can be released without breaking its tensor: ``release_mapped_storage`` drops the pages
with ``madvise(MADV_DONTNEED)``; they are re-read from the file if the tensor is used again (so writes
made to a released weight are lost).

``save_weights_for_mapping`` writes a module's weights as they are after loading (dtypes included), for
loaders that cast while loading: a view must already have the model's dtype.
"""

import ctypes
import json
import logging
import mmap
import os
import struct
import threading
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import torch
from torch.export import ExportedProgram

logger = logging.getLogger(__name__)

_MAX_HEADER_BYTES = 100 * 1024 * 1024  # the safetensors format's limit

_SAFETENSORS_DTYPES: Dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}
for _name, _attr in (
    ("U16", "uint16"),
    ("U32", "uint32"),
    ("U64", "uint64"),
    ("F8_E4M3", "float8_e4m3fn"),
    ("F8_E5M2", "float8_e5m2"),
):
    if hasattr(torch, _attr):
        _SAFETENSORS_DTYPES[_name] = getattr(torch, _attr)
_DTYPE_NAMES = {dtype: name for name, dtype in _SAFETENSORS_DTYPES.items()}

KeyMap = Union[Callable[[str], str], Mapping[str, str], None]

# Live mappings as (base address, size, weak reference to the mmap): an entry lives exactly as long as
# something (a view, or the MappedWeights handle) keeps its mmap, so an address is never reused under it.
_live_mappings: List[Tuple[int, int, "weakref.ReferenceType[mmap.mmap]"]] = []
_lock = threading.Lock()


@dataclass
class MappedWeights:
    """A safetensors file mapped onto a module or ``ExportedProgram`` by ``map_weights``."""

    path: str
    mapped_bytes: int = 0
    copied: List[str] = field(default_factory=list)
    """Names whose checkpoint dtype differed from the model's: cast into new memory (``strict=False``)."""
    missing: List[str] = field(default_factory=list)
    """Names not in the checkpoint; the model's tensor is left as it was (``strict=False``)."""
    still_meta: List[str] = field(default_factory=list)
    """Missing names whose tensor is on the ``meta`` device: the caller must materialize them."""
    _mm: Optional[mmap.mmap] = field(default=None, repr=False)
    _base: int = field(default=0, repr=False)

    def contains(self, data_ptr: int) -> bool:
        mm = self._mm
        return mm is not None and self._base <= data_ptr < self._base + len(mm)

    def close(self) -> None:
        """Drop this handle's reference to the mapping. The file stays mapped (and its pages
        releasable) while tensors still view it, and is unmapped once they are gone."""
        self._mm = None

    def __enter__(self) -> "MappedWeights":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def release_mapped_storage(storage: torch.UntypedStorage) -> bool:
    """Drop the resident pages of a storage that views a file mapped by ``map_weights``. True if the
    storage is in a mapping; False for any other storage. ``release_weights_after_build`` calls this
    for each weight whose last engine is built."""
    ptr, nbytes = storage.data_ptr(), storage.nbytes()
    if not nbytes:
        return False
    with _lock:
        _live_mappings[:] = [e for e in _live_mappings if e[2]() is not None]
        entries = list(_live_mappings)
    for base, size, ref in entries:
        mm = ref()
        if mm is None or not (base <= ptr and ptr + nbytes <= base + size):
            continue
        if hasattr(
            mmap, "MADV_DONTNEED"
        ):  # POSIX; elsewhere the pages stay until the file is unmapped
            # Only pages entirely inside the storage: a page at either end may hold a neighbour's data,
            # and MADV_DONTNEED would discard writes made to it.
            page = mmap.PAGESIZE
            lo = -(-(ptr - base) // page) * page
            hi = (ptr - base + nbytes) // page * page
            if hi > lo:
                mm.madvise(mmap.MADV_DONTNEED, lo, hi - lo)
        return True
    return False


def _read_header(path: str) -> Tuple[Dict[str, Dict[str, Any]], int]:
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path} is not a safetensors file (too short)")
        (n,) = struct.unpack("<Q", prefix)
        if n > min(_MAX_HEADER_BYTES, size - 8):
            raise ValueError(f"{path}: invalid safetensors header length {n}")
        header = json.loads(f.read(n))
    if not isinstance(header, dict):
        raise ValueError(f"{path}: the safetensors header is not a JSON object")
    header.pop("__metadata__", None)
    return header, 8 + n


def _key_mapper(key_map: KeyMap) -> Callable[[str], str]:
    if key_map is None:
        return lambda name: name
    if callable(key_map):
        return key_map
    return lambda name: key_map.get(name, name)


# Where a tensor is registered: (the dict holding it, its key there, its full name).
_Ref = Tuple[Any, str, str]


def _registrations(
    model: Union[torch.nn.Module, ExportedProgram],
) -> List[Tuple[torch.Tensor, List[_Ref]]]:
    """Each distinct tensor object once, with every place it is registered (tied weights share one)."""
    groups: Dict[int, Tuple[torch.Tensor, List[_Ref]]] = {}

    def add(table: Any, key: str, name: str) -> None:
        tensor = table[key]
        if isinstance(tensor, torch.Tensor):
            groups.setdefault(id(tensor), (tensor, []))[1].append((table, key, name))

    if isinstance(model, ExportedProgram):
        for table in (model.state_dict, model.constants):
            for key in list(table):
                add(table, key, key)
    else:
        for kind in ("_parameters", "_buffers"):
            for prefix, owner in model.named_modules(remove_duplicate=False):
                table = getattr(owner, kind)
                for key in list(table):
                    add(table, key, f"{prefix}.{key}" if prefix else key)
    return list(groups.values())


def map_weights(
    model: Union[torch.nn.Module, ExportedProgram],
    path: Union[str, "os.PathLike[str]"],
    *,
    key_map: KeyMap = None,
    strict: bool = False,
) -> MappedWeights:
    """Point ``model``'s weights at zero-copy views of the safetensors file ``path``.

    Args:
        model: A module or an ``ExportedProgram`` (its ``state_dict`` and ``constants`` are remapped). Build
            the module on the ``meta`` device to never allocate its weights. Real tensors, on any device,
            are re-pointed at the CPU views: the model ends up on the CPU, ready for CPU compilation.
        path: The safetensors file. Each tensor's dtype must equal the model's for a zero-copy view;
            ``save_weights_for_mapping`` writes a model's weights as they are after loading.
        key_map: Checkpoint name for a model name: a callable or a mapping (names not in it unchanged).
        strict: Raise unless every tensor in the model's ``state_dict`` is mapped without a copy: no
            name missing from the checkpoint (as ``load_state_dict(strict=True)``; non-persistent
            buffers are not required) and no dtype mismatch. Otherwise a dtype mismatch is cast into
            new memory and listed in ``copied``, and missing names are listed in ``missing``. A shape
            mismatch always raises. Nothing is changed if this raises.

    Returns:
        A ``MappedWeights`` handle: mapped and copied tensors, names missing from the checkpoint, and
        ``close()``. A mapped tensor is resident only while it is read; ``release_mapped_storage`` (used by
        ``release_weights_after_build``) drops its pages again.
    """
    path = os.fspath(path)
    header, start = _read_header(path)
    to_key = _key_mapper(key_map)
    with open(path, "rb") as f:
        mm = mmap.mmap(
            f.fileno(), 0, access=mmap.ACCESS_COPY
        )  # private: the file is never written
    data_size = len(mm) - start
    anchor = ctypes.c_char.from_buffer(mm)
    base = ctypes.addressof(anchor)
    del anchor
    result = MappedWeights(path=path, _mm=mm, _base=base)

    def view_of(name: str, like: torch.Tensor) -> Optional[torch.Tensor]:
        meta = header.get(to_key(name))
        if meta is None:
            return None
        if not isinstance(meta, dict):
            raise ValueError(f"{name}: invalid safetensors header entry {meta!r}")
        dtype = _SAFETENSORS_DTYPES.get(str(meta.get("dtype")))
        if dtype is None:
            raise ValueError(
                f"{name}: unsupported safetensors dtype {meta.get('dtype')}"
            )
        if list(meta.get("shape", ())) != list(like.shape):
            raise ValueError(
                f"{name}: checkpoint shape {meta.get('shape')} != model shape {list(like.shape)}"
            )
        offsets = meta.get("data_offsets")
        begin, end = (
            offsets if isinstance(offsets, list) and len(offsets) == 2 else (None, None)
        )
        if not (
            isinstance(begin, int)
            and isinstance(end, int)
            and 0 <= begin <= end <= data_size
            and end - begin == like.numel() * dtype.itemsize
        ):
            raise ValueError(
                f"{name}: invalid data_offsets {offsets} for {meta.get('dtype')} {meta.get('shape')}"
            )
        if dtype != like.dtype and strict:
            raise ValueError(
                f"{name}: checkpoint dtype {dtype} != model dtype {like.dtype}"
            )
        if end == begin:  # frombuffer rejects count=0
            view = torch.empty(like.shape, dtype=dtype)
        else:
            view = torch.frombuffer(
                mm,
                dtype=dtype,
                count=(end - begin) // dtype.itemsize,
                offset=start + begin,
            ).view(like.shape)
        if dtype != like.dtype:
            result.copied.append(name)
            return view.to(like.dtype)
        result.mapped_bytes += end - begin
        return view

    # Resolve every tensor before changing any, so an error leaves the model as it was.
    plan: List[Tuple[torch.Tensor, List[_Ref], torch.Tensor]] = []
    for tensor, refs in _registrations(model):
        names = [name for _, _, name in refs]
        view = next(
            (v for v in (view_of(n, tensor) for n in names) if v is not None), None
        )
        if view is not None:
            plan.append((tensor, refs, view))
            continue
        result.missing += names
        if tensor.is_meta:
            result.still_meta += names
    # Like load_state_dict(strict=True), only persistent names are required: non-persistent buffers
    # (e.g. rotary frequencies) are never in a checkpoint; those left on meta are in still_meta.
    persistent = set(
        model.state_dict if isinstance(model, ExportedProgram) else model.state_dict()
    )
    required = [name for name in result.missing if name in persistent]
    if strict and required:
        raise ValueError(f"{len(required)} tensors are not in {path}: {required[:8]}")

    with torch.no_grad():
        for tensor, refs, view in plan:
            if tensor.is_meta:
                new = (
                    torch.nn.Parameter(view, requires_grad=tensor.requires_grad)
                    if isinstance(tensor, torch.nn.Parameter)
                    else view
                )
                for table, key, _ in refs:  # every registration of a tied weight
                    table[key] = new
            else:
                tensor.data = view

    with _lock:
        _live_mappings.append((base, len(mm), weakref.ref(mm)))
    logger.info(
        f"Mapped {result.mapped_bytes / 2**30:.2f} GiB of weights from {path}"
        + (f"; cast {len(result.copied)} tensors" if result.copied else "")
        + (f"; {len(result.missing)} not in the checkpoint" if result.missing else "")
    )
    return result


def save_weights_for_mapping(
    model: Union[torch.nn.Module, ExportedProgram], path: Union[str, "os.PathLike[str]"]
) -> None:
    """Write ``model``'s weights, as they are now (dtypes included), to a safetensors file ``map_weights``
    can map without copies: one entry per tensor object, under its first name (a tensor registered under
    several names, such as a tied weight, is written once; distinct views of one storage are written
    separately).
    """
    entries: List[Tuple[str, torch.Tensor]] = []
    for tensor, refs in _registrations(model):
        name = refs[0][2]
        if tensor.is_meta:
            raise ValueError(
                f"{name} is on the meta device; materialize it before saving"
            )
        if tensor.dtype not in _DTYPE_NAMES:
            raise ValueError(f"{name}: dtype {tensor.dtype} has no safetensors name")
        entries.append((name, tensor.detach().contiguous().cpu()))
    # Largest element size first, so every tensor's offset stays aligned to its dtype.
    entries.sort(key=lambda e: -e[1].element_size())
    header: Dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, t in entries:
        nbytes = t.numel() * t.element_size()
        header[name] = {
            "dtype": _DTYPE_NAMES[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)  # data starts 8-byte aligned
    with open(os.fspath(path), "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _, t in entries:
            if t.numel():
                f.write(t.reshape(-1).view(torch.uint8).numpy().data)
