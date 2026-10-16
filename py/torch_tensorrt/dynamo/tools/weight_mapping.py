# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile from a checkpoint's weights in place: memory-mapped, zero-copy model weights.

``map_weights`` maps a safetensors file and points a module's parameters and buffers at views of it.
Nothing is read until something touches a weight. During compilation that is TensorRT, while it builds
the engine that uses the weight (it takes weights as host pointers, so a CPU view is read in place).

``release_mapped_storage`` drops a mapped weight's pages; they are re-read from the file if the weight is
used again. ``release_weights_after_build`` calls it once a weight's last engine is built.

``save_weights_for_mapping`` writes a module's weights as they are after loading, for checkpoints whose
loader casts (a view must already have the model's dtype).
"""

import ctypes
import json
import mmap
import os
import struct
import threading
import weakref
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Tuple, Union

import torch

_DTYPES: Dict[str, torch.dtype] = {
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
_DTYPE_NAMES = {dtype: name for name, dtype in _DTYPES.items()}
_MAX_HEADER_BYTES = 100 * 1024 * 1024  # the safetensors format's limit

# Mapped files as (start address, size, weak reference to the mmap). A storage inside one is a mapped weight.
# The weak reference keeps an entry exactly as long as some view keeps its mmap alive.
_mappings: List[Tuple[int, int, "weakref.ReferenceType[mmap.mmap]"]] = []
_lock = threading.Lock()


@dataclass
class MappedWeights:
    """What ``map_weights`` did."""

    mapped_bytes: int = 0
    missing: List[str] = field(default_factory=list)
    """Names not in the file; their tensors are left as they were (still on ``meta`` if built there)."""


def _registrations(
    module: torch.nn.Module,
) -> Iterator[Tuple[str, Dict[str, torch.Tensor], str]]:
    """(full name, the dict holding it, its key there) for every parameter and buffer; a tied weight under
    each of its names."""
    for prefix, owner in module.named_modules(remove_duplicate=False):
        for table in (owner._parameters, owner._buffers):
            for key, tensor in table.items():
                if tensor is not None:
                    yield (f"{prefix}.{key}" if prefix else key), table, key


def _read_header(path: str) -> Tuple[Dict[str, Dict[str, object]], int]:
    """The safetensors header (8-byte little-endian length, then JSON) and where the data starts."""
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


def _view(
    mm: mmap.mmap, data_start: int, entry: object, like: torch.Tensor, name: str
) -> torch.Tensor:
    """A tensor over the file's bytes for `entry`, checked to have `like`'s dtype and shape."""
    if not isinstance(entry, dict):
        raise ValueError(f"{name}: invalid safetensors header entry {entry!r}")
    dtype = _DTYPES.get(str(entry.get("dtype")))
    if dtype != like.dtype:
        raise ValueError(
            f"{name}: file dtype {entry.get('dtype')} != model dtype {like.dtype} "
            "(write the weights with save_weights_for_mapping)"
        )
    if list(entry.get("shape", ())) != list(like.shape):
        raise ValueError(
            f"{name}: file shape {entry.get('shape')} != model shape {list(like.shape)}"
        )
    offsets = entry.get("data_offsets")
    begin, end = (
        offsets if isinstance(offsets, list) and len(offsets) == 2 else (-1, -1)
    )
    if not (
        isinstance(begin, int)
        and isinstance(end, int)
        and 0 <= begin <= end <= len(mm) - data_start
        and end - begin == like.numel() * dtype.itemsize
    ):
        raise ValueError(
            f"{name}: invalid data_offsets {offsets} for {entry.get('dtype')} {entry.get('shape')}"
        )
    if end == begin:  # frombuffer rejects count=0
        return torch.empty(like.shape, dtype=dtype)
    return torch.frombuffer(
        mm, dtype=dtype, count=like.numel(), offset=data_start + begin
    ).view(like.shape)


def map_weights(
    module: torch.nn.Module,
    path: Union[str, "os.PathLike[str]"],
    *,
    strict: bool = False,
) -> MappedWeights:
    """Point ``module``'s parameters and buffers at zero-copy views of the safetensors file ``path``.

    Build the module on the ``meta`` device to never allocate its weights; its tensors are replaced, and
    a tied weight stays tied. A real tensor, on any device, keeps its object and gets ``.data = view``, so
    the module ends up on the CPU, ready for CPU compilation. The file is mapped copy-on-write and never
    written, and must stay in place while the weights are used.

    Each tensor in the file must have the model's dtype and shape (``save_weights_for_mapping`` writes
    such a file). Names missing from the file are listed in ``missing``. With ``strict=True``, any missing
    name in ``module.state_dict()`` raises, as ``load_state_dict(strict=True)`` would; non-persistent
    buffers, such as rotary frequencies, are not required. If anything raises, the module is unchanged.
    """
    path = os.fspath(path)
    header, data_start = _read_header(path)
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
    result = MappedWeights()

    # Make every view first, so an error leaves the module as it was. Keyed by the original tensor's id,
    # which the stored original keeps from being reused.
    views: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name, table, key in _registrations(module):
        tensor = table[key]
        if id(tensor) in views:
            continue  # a tied weight, seen under another name
        if name not in header:
            result.missing.append(name)
            continue
        view = _view(mm, data_start, header[name], tensor, name)
        views[id(tensor)] = (tensor, view)
        result.mapped_bytes += view.nbytes
    if strict:
        persistent = set(module.state_dict())
        required = [name for name in result.missing if name in persistent]
        if required:
            raise ValueError(
                f"{len(required)} tensors are not in {path}: {required[:8]}"
            )

    with torch.no_grad():
        replacements: Dict[int, torch.Tensor] = {}
        for _, table, key in list(_registrations(module)):
            tensor = table[key]
            if id(tensor) not in views:
                continue
            view = views[id(tensor)][1]
            if not tensor.is_meta:
                tensor.data = view
                continue
            if id(tensor) not in replacements:
                replacements[id(tensor)] = (
                    torch.nn.Parameter(view, requires_grad=tensor.requires_grad)
                    if isinstance(tensor, torch.nn.Parameter)
                    else view
                )
            table[key] = replacements[id(tensor)]

    with _lock:
        _mappings.append((_address(mm), len(mm), weakref.ref(mm)))
    return result


def _address(mm: mmap.mmap) -> int:
    anchor = ctypes.c_char.from_buffer(mm)
    address = ctypes.addressof(anchor)
    del anchor  # a live export would keep the mmap from closing
    return address


def release_mapped_storage(storage: torch.UntypedStorage) -> bool:
    """Drop the resident pages of a storage that views a file mapped by ``map_weights``; True if it does.
    The tensor stays valid and is re-read from the file if used again, so writes made to it are lost.
    """
    ptr, nbytes = storage.data_ptr(), storage.nbytes()
    if not nbytes:
        return False
    with _lock:
        _mappings[:] = [m for m in _mappings if m[2]() is not None]
        entries = list(_mappings)
    for base, size, ref in entries:
        mm = ref()
        if mm is None or not (base <= ptr and ptr + nbytes <= base + size):
            continue
        if hasattr(mmap, "MADV_DONTNEED"):  # POSIX; elsewhere pages stay until unmapped
            # Only pages entirely inside the storage: a page at either end may hold a neighbour's data,
            # and MADV_DONTNEED would discard writes made to it.
            page = mmap.PAGESIZE
            lo = -(-(ptr - base) // page) * page
            hi = (ptr - base + nbytes) // page * page
            if hi > lo:
                mm.madvise(mmap.MADV_DONTNEED, lo, hi - lo)
        return True
    return False


def save_weights_for_mapping(
    module: torch.nn.Module, path: Union[str, "os.PathLike[str]"]
) -> None:
    """Write ``module``'s parameters and buffers, as they are now (dtypes included), to a safetensors file
    ``map_weights`` can map. A tied weight is written once, under its first name."""
    entries: Dict[int, Tuple[str, torch.Tensor]] = {}
    for name, table, key in _registrations(module):
        tensor = table[key]
        if id(tensor) in entries:
            continue
        if tensor.is_meta:
            raise ValueError(
                f"{name} is on the meta device; materialize it before saving"
            )
        if tensor.dtype not in _DTYPE_NAMES:
            raise ValueError(f"{name}: dtype {tensor.dtype} has no safetensors name")
        entries[id(tensor)] = (name, tensor.detach().contiguous().cpu())
    # Largest element size first, so every tensor's offset stays aligned to its dtype.
    ordered = sorted(entries.values(), key=lambda e: -e[1].element_size())
    header: Dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, t in ordered:
        header[name] = {
            "dtype": _DTYPE_NAMES[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + t.nbytes],
        }
        offset += t.nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)  # data starts 8-byte aligned
    with open(os.fspath(path), "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _, t in ordered:
            if t.numel():
                f.write(t.reshape(-1).view(torch.uint8).numpy().data)
