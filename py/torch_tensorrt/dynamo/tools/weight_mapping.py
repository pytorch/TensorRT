# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile from a checkpoint's weights in place: memory-mapped, zero-copy model weights.

``map_weights`` maps a safetensors file and points a module's parameters and buffers at views of it.
Nothing is read until something touches a weight. During compilation that is TensorRT, while it builds
the engine that uses the weight (it takes weights as host pointers, so a CPU view is read in place).

``release_mapped_storage`` drops a mapped weight's resident pages; they are read back if the weight is used
again. ``release_weights_after_build`` calls it once a weight's last engine is built.

``save_weights_for_mapping`` writes a module's weights as they are after loading, for checkpoints whose
loader casts (a view must already have the model's dtype). With ``source=`` it converts a checkpoint
without loading it: the module is built on ``meta``, and each weight is read, cast and written in turn.
"""

import ctypes
import io
import json
import logging
import math
import mmap
import os
import struct
import sys
import threading
import warnings
import weakref
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

import torch
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

logger = logging.getLogger(__name__)

_DTYPES: Dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "U16": torch.uint16,
    "U32": torch.uint32,
    "U64": torch.uint64,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}
# Ways to drop a copy-on-write mapping's resident pages, best first. MADV_PAGEOUT (Linux 5.4+; Python names it
# only when built with headers that have it) keeps writes made to them; MADV_DONTNEED discards them.
_PAGE_DROP_ADVICE = tuple(
    advice
    for advice in (
        getattr(mmap, "MADV_PAGEOUT", 21 if sys.platform == "linux" else None),
        getattr(mmap, "MADV_DONTNEED", None),
    )
    if advice is not None
)
_DTYPE_NAMES = {dtype: name for name, dtype in _DTYPES.items()}
_MAX_HEADER_BYTES = 100 * 1024 * 1024  # the safetensors format's limit
# For a tensor on ``meta``: (its name, the tensor) -> its data, whole or as chunks along dimension 0.
WeightSource = Callable[
    [str, torch.Tensor], Union[torch.Tensor, Iterable[torch.Tensor]]
]


class _MappedFile(NamedTuple):
    """A file mapped by ``map_weights``. A storage between ``base`` and ``base + size`` is a mapped weight."""

    base: int
    size: int
    # Keeps the entry exactly as long as some view keeps the mmap alive.
    mm: "weakref.ReferenceType[mmap.mmap]"
    path: str
    # (st_dev, st_ino, st_size, st_mtime_ns) when it was mapped.
    file_id: Tuple[int, int, int, int]


_mappings: List[_MappedFile] = []
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
    """The safetensors header (8-byte little-endian length, then JSON) and where the data starts, checked
    against the file so a damaged one fails before anything is mapped. Only the header is read.
    """
    if path.endswith(".json"):
        raise ValueError(
            f"{path} is the index of a sharded checkpoint, and map_weights maps one file. Write the "
            "weights to one file with save_weights_for_mapping(..., source=checkpoint_source(index)), "
            "which reads the shards one tensor at a time."
        )
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
    _check_layout(path, header, size - 8 - n)
    return header, 8 + n


def _check_layout(path: str, header: Dict[str, Any], data_size: int) -> None:
    """Every entry's bytes must hold its dtype and shape, and the entries must cover the data exactly, one
    after another, as the safetensors format requires: a truncated file, bytes after the last tensor and
    overlapping or gapped tensors all fail. Tensors the module doesn't map are checked too.
    """
    spans = []
    for name, entry in header.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"{path}: invalid safetensors header entry for {name}: {entry!r}"
            )
        offsets, shape = entry.get("data_offsets"), entry.get("shape")
        if not (
            isinstance(offsets, list)
            and len(offsets) == 2
            and all(isinstance(o, int) and o >= 0 for o in offsets)
            and offsets[0] <= offsets[1]
        ):
            raise ValueError(
                f"{path}: {name}'s data_offsets {offsets} aren't two increasing byte offsets"
            )
        if not (
            isinstance(shape, list)
            and all(isinstance(d, int) and d >= 0 for d in shape)
        ):
            raise ValueError(f"{path}: {name}'s shape {shape} isn't a list of sizes")
        dtype = _DTYPES.get(str(entry.get("dtype")))  # unsupported: checked if mapped
        if (
            dtype is not None
            and offsets[1] - offsets[0] != math.prod(shape) * dtype.itemsize
        ):
            raise ValueError(
                f"{path}: {name}'s data_offsets {offsets} don't hold {entry.get('dtype')} {shape}"
            )
        spans.append((offsets[0], offsets[1], name))
    end = 0
    for begin, span_end, name in sorted(spans):
        if begin != end:
            raise ValueError(
                f"{path}: {name}'s data_offsets start at byte {begin}, where the tensor before it ends at "
                f"{end}: tensors must follow each other with no gap or overlap"
            )
        end = span_end
    if end != data_size:
        raise ValueError(
            f"{path}: the tensors' data_offsets end at byte {end} of the data, but the file holds "
            f"{data_size}: "
            + (
                "it is truncated"
                if end > data_size
                else "something follows the last tensor"
            )
        )


def _view(
    mm: mmap.mmap, data_start: int, entry: object, like: torch.Tensor, name: str
) -> torch.Tensor:
    """A tensor over the file's bytes for `entry`, checked to have `like`'s dtype and shape."""
    if not isinstance(entry, dict):
        raise ValueError(f"{name}: invalid safetensors header entry {entry!r}")
    dtype = _DTYPES.get(str(entry.get("dtype")))
    if dtype is None:
        raise ValueError(f"{name}: dtype {entry.get('dtype')} is not supported")
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
    strict: bool = True,
    prefix: str = "",
) -> MappedWeights:
    """Point ``module``'s parameters and buffers at zero-copy views of the safetensors file ``path``.

    Build the module on the ``meta`` device to never allocate its weights; its tensors are replaced, and
    a tied weight stays tied. A real tensor, on any device, keeps its object and gets ``.data = view``, so
    the module ends up on the CPU, ready for CPU compilation. The file is mapped copy-on-write and never
    written. It must stay in place and unchanged while the weights are used: rewriting it changes or
    invalidates them.

    Each tensor in the file must have the model's dtype and shape (``save_weights_for_mapping`` writes
    such a file). The file is checked from its header before anything is mapped: a truncated file, bytes
    after the last tensor, or tensors whose bytes overlap, leave gaps or don't hold their dtype and shape
    raise, also for tensors the module doesn't have. The bytes themselves aren't read, so damage inside a
    tensor isn't detected. Names missing from the file are listed in ``missing``. With ``strict=True``, the default,
    a parameter or persistent buffer missing from the file raises; non-persistent buffers, such as rotary
    frequencies, are not required, and tensors in the file that the module doesn't have are ignored. A
    missing tensor on ``meta`` stays there, with no values, and a warning names it. If anything raises,
    the module is unchanged.

    ``prefix`` is prepended to each name to find it in the file, as for ``load_state_dict``, so a submodule
    maps straight from a checkpoint of the whole model.
    """
    path = os.fspath(path)
    header, data_start = _read_header(path)
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
        stat = os.fstat(f.fileno())
    file_id = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    result = MappedWeights()

    # Make every view first, so an error leaves the module as it was. Keyed by the original tensor's id,
    # which the stored original keeps from being reused.
    views: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    absent: List[Tuple[str, torch.Tensor]] = []
    for name, table, key in _registrations(module):
        tensor = table[key]
        if id(tensor) in views:
            continue  # a tied weight, seen under another name
        if prefix + name not in header:
            absent.append((name, tensor))
            continue
        view = _view(mm, data_start, header[prefix + name], tensor, prefix + name)
        views[id(tensor)] = (tensor, view)
        result.mapped_bytes += view.nbytes
    # A tied weight the file stores under a later name is not missing.
    missing = [(name, tensor) for name, tensor in absent if id(tensor) not in views]
    result.missing = [name for name, _ in missing]
    persistent = set(module.state_dict())
    required = [name for name in result.missing if name in persistent]
    if required and strict:
        hint = (
            f" The module's names look like {required[0]!r} and the file's like "
            f"{next(iter(header))!r}; pass prefix= if the file's have a prefix the module's don't."
            if header and not views
            else ""
        )
        raise ValueError(
            f"{len(required)} tensors are not in {path}: {required[:8]}.{hint}"
        )
    logger.info(f"Mapped {result.mapped_bytes / 2**20:.1f} MiB of weights from {path}")
    on_meta = [name for name, tensor in missing if tensor.is_meta]
    if on_meta:
        logger.warning(
            f"{len(on_meta)} tensors are not in {path} and stay on the meta device, with no values: "
            f"{on_meta[:8]}. Write the file with save_weights_for_mapping, which includes non-persistent "
            "buffers, or set them before using the module."
        )
    kept = [name for name in required if name not in on_meta]
    if kept:
        logger.warning(
            f"{len(kept)} parameters and persistent buffers are not in {path}, so they keep "
            f"their values: {kept[:8]}"
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
        _mappings.append(
            _MappedFile(_address(mm), len(mm), weakref.ref(mm), path, file_id)
        )
    return result


def _address(mm: mmap.mmap) -> int:
    anchor = ctypes.c_char.from_buffer(mm)
    address = ctypes.addressof(anchor)
    del anchor  # a live export would keep the mmap from closing
    return address


def release_mapped_storage(storage: torch.UntypedStorage) -> bool:
    """Drop the resident pages of a storage that views a file mapped by ``map_weights``; True if it does.
    The tensor stays valid: its pages are read back when it is used again, from the file, or from swap for
    pages it has written. Kernels before Linux 5.4 can only discard written pages, which then revert to the
    file's data. Where pages can't be dropped (Windows), this returns False.
    """
    ptr, nbytes = storage.data_ptr(), storage.nbytes()
    if not nbytes:
        return False
    for mapped in _live_mappings():
        mm = mapped.mm()
        if mm is None or not (
            mapped.base <= ptr and ptr + nbytes <= mapped.base + mapped.size
        ):
            continue
        # Only pages entirely inside the storage: a page at either end may hold a neighbour's data.
        page = mmap.PAGESIZE
        lo = -(-(ptr - mapped.base) // page) * page
        hi = (ptr - mapped.base + nbytes) // page * page
        return hi <= lo or _drop_pages(mm, lo, hi - lo)
    return False


def _live_mappings() -> List[_MappedFile]:
    with _lock:
        _mappings[:] = [m for m in _mappings if m.mm() is not None]
        return list(_mappings)


def raise_if_mapped_files_changed(tensors: Iterable[torch.Tensor]) -> None:
    """Raise if a file that one of ``tensors`` is mapped from was removed or changed since ``map_weights``
    mapped it. A mapped weight is read when first used: from a file rewritten since, it reads the new
    contents, and past the end of a file truncated since, the read kills the process with SIGBUS.
    """
    ptrs = {
        t.untyped_storage().data_ptr()
        for t in tensors
        if not t.is_meta
        and t.layout == torch.strided
        and not is_traceable_wrapper_subclass(t)
    }
    for mapped in _live_mappings():
        if not any(mapped.base <= p < mapped.base + mapped.size for p in ptrs):
            continue
        try:
            stat = os.stat(mapped.path)
        except FileNotFoundError:
            raise ValueError(
                f"The weights file {mapped.path} was removed after map_weights mapped it. Keep it in "
                "place while the model is in use."
            ) from None
        if (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
        ) != mapped.file_id:
            raise ValueError(
                f"The weights file {mapped.path} changed after map_weights mapped it: it is "
                f"{stat.st_size} bytes, {mapped.file_id[2]} when mapped, and was modified or replaced "
                "since. Its weights are read when compilation first uses them, so map them again."
            )


def _drop_pages(mm: mmap.mmap, start: int, length: int) -> bool:
    for advice in _PAGE_DROP_ADVICE:
        try:
            mm.madvise(advice, start, length)
            return True
        except OSError:  # MADV_PAGEOUT on a kernel before 5.4
            continue
    warnings.warn(
        "Pages of memory-mapped weights can't be dropped on this platform, so released weights stay "
        "resident until their file is unmapped.",
        RuntimeWarning,
    )
    return False


def save_weights_for_mapping(
    module: torch.nn.Module,
    path: Union[str, "os.PathLike[str]"],
    *,
    source: Optional[WeightSource] = None,
) -> None:
    """Write ``module``'s parameters and buffers, with their dtypes, to a safetensors file ``map_weights`` can
    map. A tied weight is written once, under its first name.

    Tensors are copied to the host and written one at a time. A tensor on the ``meta`` device is read from
    ``source(name, tensor)`` instead, which returns its data in any dtype, as one tensor or as chunks that
    concatenate along dimension 0 into it. Each chunk is cast to the tensor's dtype and written as it comes.
    Converting a checkpoint this way holds one chunk in memory at a time: build the model on ``meta``, change
    it as the compiled model will be (dtypes, shapes), and let ``source`` read the checkpoint. The file is
    written under a temporary name and renamed when complete.
    """
    entries: Dict[int, Tuple[str, torch.Tensor]] = {}
    for name, table, key in _registrations(module):
        tensor = table[key]
        if id(tensor) in entries:
            continue
        if tensor.is_meta and source is None:
            raise ValueError(
                f"{name} is on the meta device; pass source= to read it, or materialize it"
            )
        if tensor.dtype not in _DTYPE_NAMES:
            raise ValueError(f"{name}: dtype {tensor.dtype} has no safetensors name")
        entries[id(tensor)] = (name, tensor)
    # Largest element size first, so every tensor's offset stays aligned to its dtype.
    ordered = sorted(entries.values(), key=lambda e: -e[1].element_size())
    header: Dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, t in ordered:
        nbytes = t.numel() * t.element_size()
        header[name] = {
            "dtype": _DTYPE_NAMES[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)  # data starts 8-byte aligned
    path = os.fspath(path)
    partial = path + ".partial"
    try:
        with open(partial, "wb") as f:
            f.write(struct.pack("<Q", len(blob)))
            f.write(blob)
            for name, t in ordered:
                _write_tensor(f, name, t, source)
        os.replace(partial, path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)


def _write_tensor(
    f: "io.BufferedWriter",
    name: str,
    like: torch.Tensor,
    source: Optional[WeightSource],
) -> None:
    """Write `like`'s data, from `source` if it is on ``meta``, one chunk at a time, in `like`'s dtype."""
    chunks: Union[torch.Tensor, Iterable[torch.Tensor]] = like
    if like.is_meta:
        assert source is not None
        chunks = source(name, like)
    if isinstance(chunks, torch.Tensor):
        chunks = [chunks]
    expected, written = like.numel() * like.element_size(), 0
    for chunk in chunks:
        if chunk.is_meta or (
            like.dim() and tuple(chunk.shape[1:]) != tuple(like.shape[1:])
        ):
            raise ValueError(
                f"{name}: source gave a {'meta' if chunk.is_meta else list(chunk.shape)} chunk "
                f"for a {list(like.shape)} tensor"
            )
        data = chunk.detach().to(device="cpu", dtype=like.dtype).contiguous()
        if data.numel():
            f.write(data.reshape(-1).view(torch.uint8).numpy().data)
        written += data.numel() * data.element_size()
        del data
    if written != expected:
        raise ValueError(f"{name}: source gave {written} bytes, expected {expected}")


def checkpoint_source(
    path: Union[str, "os.PathLike[str]"],
    key: Optional[Callable[[str], str]] = None,
    *,
    chunk_bytes: int = 64 << 20,
) -> WeightSource:
    """A ``source=`` for ``save_weights_for_mapping`` that reads each tensor from the safetensors checkpoint
    ``path``, under ``key(name)`` (the name itself by default), in chunks of at most ``chunk_bytes`` along
    dimension 0, so no whole checkpoint tensor is ever in memory. The file's dtype may differ; the writer
    casts each chunk. ``path`` may also be a sharded checkpoint's ``model.safetensors.index.json``.
    """
    try:
        from safetensors import safe_open
    except ImportError as e:
        raise ImportError(
            "checkpoint_source reads checkpoints with the safetensors package: pip install safetensors"
        ) from e

    path = os.fspath(path)
    shard_of: Dict[str, str] = {}  # for an index, the shard file of each name
    if path.endswith(".json"):
        with open(path) as f:
            shard_of = json.load(f)["weight_map"]
    directory = os.path.dirname(path)
    files: Dict[str, Tuple[Any, Set[str]]] = {}  # each file open once, with its names

    def missing_tensor_error(stored_name: str) -> ValueError:
        return ValueError(
            f"{stored_name} is not in {path}. A buffer a checkpoint doesn't store, such as non-persistent "
            "rotary frequencies, must be a real tensor, not on meta: build the model with accelerate's "
            "init_empty_weights(), which keeps buffers real, or set the buffer before writing."
        )

    def get_checkpoint_file(stored_name: str) -> Any:
        """The open safetensors file that stores ``stored_name``."""
        if shard_of and stored_name not in shard_of:
            raise missing_tensor_error(stored_name)
        file = os.path.join(directory, shard_of[stored_name]) if shard_of else path
        if file not in files:
            handle = safe_open(file, framework="pt", device="cpu")
            files[file] = (handle, set(handle.keys()))
        handle, names = files[file]
        if stored_name not in names:
            raise missing_tensor_error(stored_name)
        return handle

    def read_tensor_chunks(name: str, like: torch.Tensor) -> Iterable[torch.Tensor]:
        stored_name = key(name) if key is not None else name
        checkpoint_file = get_checkpoint_file(stored_name)
        stored = checkpoint_file.get_slice(stored_name)
        shape = list(stored.get_shape())
        if shape != list(like.shape):
            raise ValueError(
                f"{name}: {stored_name} in {path} has shape {shape}, the model {list(like.shape)}"
            )
        if not shape:
            return [checkpoint_file.get_tensor(stored_name)]
        if stored.get_dtype() not in _DTYPES:
            raise ValueError(f"{name}: dtype {stored.get_dtype()} is not supported")
        row_bytes = _DTYPES[stored.get_dtype()].itemsize
        for size in shape[1:]:
            row_bytes *= size
        rows = max(1, chunk_bytes // max(1, row_bytes))
        return (stored[i : i + rows] for i in range(0, shape[0], rows))

    return read_tensor_chunks
