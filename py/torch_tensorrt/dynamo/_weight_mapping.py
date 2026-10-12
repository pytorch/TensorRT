# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile from a checkpoint's weights in place: zero-copy, memory-mapped model weights.

``map_weights`` points a module's parameters and buffers (or an ``ExportedProgram``'s state) at views of a
safetensors file. The file is mapped copy-on-write and never written; a weight's pages become resident
only when something reads them. During compilation that is TensorRT, while it builds the engine of the
partition that uses the weight: TensorRT takes weights as host pointers, and the converter passes a CPU
tensor's own data pointer, so a mapped model compiles without a host copy of its weights.

A mapped weight can be released without breaking its tensor: ``release_mapped_storage`` drops the pages
with ``madvise(MADV_DONTNEED)``; they are re-read from the file if the tensor is used again.

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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple, Union

import torch
from torch.export import ExportedProgram

logger = logging.getLogger(__name__)

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


@dataclass
class MappedWeights:
    """A safetensors file mapped onto a module or ``ExportedProgram`` by ``map_weights``."""

    path: str
    mapped_bytes: int = 0
    copied: List[str] = field(default_factory=list)
    """Names whose checkpoint dtype differed from the model's: cast into new memory (``strict=False``)."""
    missing: List[str] = field(default_factory=list)
    """Names not in the checkpoint; the model's tensor is left as it was."""
    still_meta: List[str] = field(default_factory=list)
    """Missing names whose tensor is on the ``meta`` device: the caller must materialize them."""
    _mm: Optional[mmap.mmap] = field(default=None, repr=False)
    _base: int = field(default=0, repr=False)

    @property
    def size(self) -> int:
        return len(self._mm) if self._mm is not None else 0

    def contains(self, data_ptr: int) -> bool:
        return self._mm is not None and self._base <= data_ptr < self._base + len(
            self._mm
        )

    def release(self, storage: torch.UntypedStorage) -> bool:
        """Drop the resident pages of a storage that lies in this mapping. True if it did."""
        ptr, nbytes = storage.data_ptr(), storage.nbytes()
        if (
            self._mm is None
            or not nbytes
            or not self.contains(ptr)
            or not hasattr(mmap, "MADV_DONTNEED")
        ):
            return False
        page = mmap.PAGESIZE
        lo = (ptr - self._base) // page * page
        hi = min(len(self._mm), -(-(ptr - self._base + nbytes) // page) * page)
        self._mm.madvise(mmap.MADV_DONTNEED, lo, hi - lo)
        return True

    def close(self) -> None:
        """Stop tracking the mapping. The file stays mapped while tensors still view it and is unmapped
        once they are gone (each view holds a reference to the mapping)."""
        with _lock:
            if self in _live_mappings:
                _live_mappings.remove(self)
        self._mm = None


_live_mappings: List[MappedWeights] = []
_lock = threading.Lock()


def release_mapped_storage(storage: torch.UntypedStorage) -> bool:
    """Drop the pages of a storage that views a file mapped by ``map_weights``. True if it did; False for
    any other storage (the caller can release it another way)."""
    with _lock:
        mappings = list(_live_mappings)
    return any(m.release(storage) for m in mappings)


def _register_with_weight_release() -> None:
    # When compilation releases weights after their last engine (release_weights_after_build), let it drop
    # mapped weights' pages instead of resizing a storage it cannot resize.
    try:
        from torch_tensorrt.dynamo._weight_release import register_release_handler
    except ImportError:
        return
    register_release_handler(release_mapped_storage)


def _read_header(path: str) -> Tuple[Dict[str, Dict[str, Any]], int]:
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path} is not a safetensors file (too short)")
        (n,) = struct.unpack("<Q", prefix)
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def _key_mapper(key_map: KeyMap) -> Callable[[str], str]:
    if key_map is None:
        return lambda name: name
    if callable(key_map):
        return key_map
    return lambda name: key_map.get(name, name)


def _named_tensors(module: torch.nn.Module) -> Iterator[Tuple[str, torch.Tensor, bool]]:
    for name, p in module.named_parameters(remove_duplicate=False):
        yield name, p, True
    for name, b in module.named_buffers(remove_duplicate=False):
        yield name, b, False


def _owner(module: torch.nn.Module, name: str) -> Tuple[torch.nn.Module, str]:
    *path, attr = name.split(".")
    for part in path:
        module = getattr(module, part)
    return module, attr


def map_weights(
    model: Union[torch.nn.Module, ExportedProgram],
    path: Union[str, "os.PathLike[str]"],
    *,
    key_map: KeyMap = None,
    strict: bool = False,
) -> MappedWeights:
    """Point ``model``'s weights at zero-copy views of the safetensors file ``path``.

    Args:
        model: A module (its tensors may be on the ``meta`` device, which never allocates them) or an
            ``ExportedProgram`` (its ``state_dict`` and ``constants`` are remapped; an ``ExportedProgram``
            references its module's tensors, so map before ``torch.export`` or pass the program).
        path: The safetensors file. Each tensor's dtype must equal the model's for a zero-copy view;
            ``save_weights_for_mapping`` writes a model's weights as they are after loading.
        key_map: Checkpoint name for a model name: a callable or a mapping (names not in it unchanged).
        strict: Raise if a tensor's dtype differs from the checkpoint's; otherwise cast it into new memory
            and list it in ``copied``. A shape mismatch always raises.

    Returns:
        A ``MappedWeights`` handle: mapped and copied tensors, names missing from the checkpoint, and
        ``close()``. A mapped tensor is resident only while it is read; ``release_mapped_storage`` (used by
        ``release_weights_after_build`` when available) drops its pages again.
    """
    path = os.fspath(path)
    header, start = _read_header(path)
    to_key = _key_mapper(key_map)
    with open(path, "rb") as f:
        mm = mmap.mmap(
            f.fileno(), 0, access=mmap.ACCESS_COPY
        )  # private: the file is never written
    anchor = ctypes.c_char.from_buffer(mm)
    base = ctypes.addressof(anchor)
    del anchor
    result = MappedWeights(path=path, _mm=mm, _base=base)

    def view_of(name: str, like: torch.Tensor) -> Optional[torch.Tensor]:
        meta = header.get(to_key(name))
        if meta is None:
            return None
        dtype = _SAFETENSORS_DTYPES.get(meta["dtype"])
        if dtype is None:
            raise ValueError(f"{name}: unsupported safetensors dtype {meta['dtype']}")
        if list(meta["shape"]) != list(like.shape):
            raise ValueError(
                f"{name}: checkpoint shape {meta['shape']} != model shape {list(like.shape)}"
            )
        begin, end = meta["data_offsets"]
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
            if strict:
                raise ValueError(
                    f"{name}: checkpoint dtype {dtype} != model dtype {like.dtype}"
                )
            result.copied.append(name)
            return view.to(like.dtype)
        result.mapped_bytes += end - begin
        return view

    with torch.no_grad():
        if isinstance(model, ExportedProgram):
            for table in (model.state_dict, model.constants):
                for name, tensor in list(table.items()):
                    if not isinstance(tensor, torch.Tensor):
                        continue
                    view = view_of(name, tensor)
                    if view is None:
                        result.missing.append(name)
                    elif tensor.is_meta:
                        table[name] = (
                            torch.nn.Parameter(view, requires_grad=False)
                            if isinstance(tensor, torch.nn.Parameter)
                            else view
                        )
                    else:
                        tensor.data = view
        else:
            done: Dict[int, torch.Tensor] = (
                {}
            )  # id of the original tensor -> its view (tied weights)
            names_of: Dict[int, List[str]] = {}
            tensors = list(
                _named_tensors(model)
            )  # snapshot: the loop below replaces meta tensors
            for name, tensor, _ in tensors:
                names_of.setdefault(id(tensor), []).append(name)
            for name, tensor, is_param in tensors:
                key = id(tensor)
                if key in done:
                    continue
                view = next(
                    (
                        v
                        for v in (view_of(n, tensor) for n in names_of[key])
                        if v is not None
                    ),
                    None,
                )
                if view is None:
                    result.missing.append(name)
                    if tensor.is_meta:
                        result.still_meta.append(name)
                    done[key] = tensor
                    continue
                if tensor.is_meta:
                    # Replace the meta tensor everywhere it is registered (tied weights share one object).
                    new: torch.Tensor = (
                        torch.nn.Parameter(view, requires_grad=tensor.requires_grad)
                        if is_param
                        else view
                    )
                    for n in names_of[key]:
                        owner, attr = _owner(model, n)
                        (owner._parameters if is_param else owner._buffers)[attr] = new
                else:
                    tensor.data = view
                done[key] = view

    with _lock:
        _live_mappings.append(result)
    _register_with_weight_release()
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
    can map without copies: one entry per storage, under the first name that refers to it.
    """
    seen: Dict[int, str] = {}
    entries: List[Tuple[str, torch.Tensor]] = []
    if isinstance(model, ExportedProgram):
        named = [
            (n, t)
            for table in (model.state_dict, model.constants)
            for n, t in table.items()
            if isinstance(t, torch.Tensor)
        ]
    else:
        named = [(n, t) for n, t, _ in _named_tensors(model)]
    for name, tensor in named:
        if tensor.is_meta:
            raise ValueError(
                f"{name} is on the meta device; materialize it before saving"
            )
        key = id(tensor)
        if key in seen:
            continue
        seen[key] = name
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
                f.write(t.reshape(-1).view(torch.uint8).numpy())
