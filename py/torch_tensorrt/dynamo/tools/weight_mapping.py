# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compile from a checkpoint's weights in place: memory-mapped, zero-copy model weights.

``map_weights`` maps a safetensors file and points a module's parameters and buffers at views of it.
Nothing is read until something touches a weight. During compilation that is TensorRT, while it builds
the engine that uses the weight (it takes weights as host pointers, so a CPU view is read in place).

``release_weights_after_build`` drops a mapped weight's resident pages once its last engine is built; they
are read back if the weight is used again.

``save_weights_for_mapping`` writes a module's weights as they are after loading, for checkpoints whose
loader casts (a view must already have the model's dtype). With ``source=`` it converts a checkpoint
without loading it: each weight is read from the checkpoint, cast and written in turn, so the module can
be built on ``meta``.
"""

import ctypes
import io
import json
import logging
import math
import mmap
import os
import platform
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
    Hashable,
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

# The dtype names of the safetensors format.
_SAFETENSORS_DTYPES: Dict[str, torch.dtype] = {
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
_SAFETENSORS_DTYPE_NAMES = {dtype: name for name, dtype in _SAFETENSORS_DTYPES.items()}
_MAX_HEADER_BYTES = 100 * 1024 * 1024  # the safetensors format's limit

# Python names it from 3.13; the fallback is its value on Linux x86-64 and aarch64.
_MAP_NORESERVE = getattr(
    mmap,
    "MAP_NORESERVE",
    (
        0x4000
        if sys.platform == "linux" and platform.machine() in ("x86_64", "aarch64")
        else 0
    ),
)
# Ways to drop a copy-on-write mapping's resident pages. MADV_PAGEOUT (Linux 5.4+; Python names it only when
# built with headers that have it) keeps writes made to them; MADV_DONTNEED discards them.
_MADV_PAGEOUT = getattr(mmap, "MADV_PAGEOUT", 21 if sys.platform == "linux" else None)
_MADV_DONTNEED = getattr(mmap, "MADV_DONTNEED", None)

# For a tensor on ``meta``: (its name, the tensor) -> its data, whole or as chunks along dimension 0.
WeightSource = Callable[
    [str, torch.Tensor], Union[torch.Tensor, Iterable[torch.Tensor]]
]


@dataclass
class MappedWeights:
    """What ``map_weights`` did."""

    mapped_bytes: int = 0
    missing: List[str] = field(default_factory=list)
    """Names not in the file; their tensors are left as they were (still on ``meta`` if built there)."""


class _TensorSlot(NamedTuple):
    """Where a module holds one of its parameters or buffers: ``table[key]``, where ``table`` is the owning
    module's ``_parameters`` or ``_buffers``, under the full name ``name``."""

    name: str
    table: Dict[str, torch.Tensor]
    key: str

    @property
    def tensor(self) -> torch.Tensor:
        return self.table[self.key]


def _find_tensor_slots(module: torch.nn.Module) -> Iterator[_TensorSlot]:
    """The slot of every parameter and buffer of ``module``; a tied weight has one under each of its names."""
    for prefix, owner in module.named_modules(remove_duplicate=False):
        for table in (owner._parameters, owner._buffers):
            for key, tensor in table.items():
                if tensor is not None:
                    yield _TensorSlot(f"{prefix}.{key}" if prefix else key, table, key)


def _group_tensor_slots_by_memory(
    module: torch.nn.Module, caller: str
) -> List[List[_TensorSlot]]:
    """``module``'s tensor slots, grouped by the memory their tensors are: a tied weight's names together, and
    also distinct tensors that are the same view of one storage, such as ``w`` and ``Parameter(w.detach())``.
    Tensors that share a storage otherwise raise, since ``caller`` would separate them.
    """
    groups: Dict[Hashable, List[_TensorSlot]] = {}
    first_view_of_storage: Dict[int, Tuple[str, Hashable]] = {}
    for slot in _find_tensor_slots(module):
        tensor = slot.tensor
        memory: Hashable = id(tensor)
        if tensor.layout == torch.strided and not is_traceable_wrapper_subclass(tensor):
            storage = tensor.untyped_storage()._cdata
            memory = (
                storage,
                tensor.storage_offset(),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.dtype,
            )
            other, other_memory = first_view_of_storage.setdefault(
                storage, (slot.name, memory)
            )
            if other_memory != memory:
                raise ValueError(
                    f"{slot.name} and {other} share memory without being the same tensor, and {caller} "
                    "would separate them. Make one a copy of the other (clone()), or use one tensor for both."
                )
        groups.setdefault(memory, []).append(slot)
    return list(groups.values())


def _read_safetensors_header(path: str) -> Tuple[Dict[str, Dict[str, Any]], int]:
    """The entries of the safetensors header (8-byte little-endian length, then JSON) and where the data
    starts, checked against the file so a damaged one fails before anything is mapped. Only the header is
    read."""
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
        entries = json.loads(f.read(n))
    if not isinstance(entries, dict):
        raise ValueError(f"{path}: the safetensors header is not a JSON object")
    entries.pop("__metadata__", None)
    _raise_if_layout_is_invalid(path, entries, size - 8 - n)
    return entries, 8 + n


def _raise_if_layout_is_invalid(
    path: str, entries: Dict[str, Any], data_bytes: int
) -> None:
    """Every entry's bytes must hold its dtype and shape, and the entries must cover the data exactly, one
    after another, as the safetensors format requires: a truncated file, bytes after the last tensor and
    overlapping or gapped tensors all fail. Tensors the module doesn't map are checked too.
    """
    spans = []
    for name, entry in entries.items():
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
        # An unsupported dtype is reported when its tensor is mapped.
        dtype = _SAFETENSORS_DTYPES.get(str(entry.get("dtype")))
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
    if end != data_bytes:
        raise ValueError(
            f"{path}: the tensors' data_offsets end at byte {end} of the data, but the file holds "
            f"{data_bytes}: "
            + (
                "it is truncated"
                if end > data_bytes
                else "something follows the last tensor"
            )
        )


def map_weights(
    module: torch.nn.Module,
    path: Union[str, "os.PathLike[str]"],
    *,
    strict: bool = True,
    prefix: str = "",
) -> MappedWeights:
    """Point ``module``'s parameters and buffers at zero-copy views of the safetensors file ``path``.

    Build the module on the ``meta`` device to never allocate its weights; its tensors are replaced. A tied
    weight stays tied, and tensors that are one view of a storage keep sharing it; tensors that share a
    storage otherwise raise. A real tensor, on any device, keeps its object and gets ``.data = view``, so
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
    entries, data_start = _read_safetensors_header(path)
    groups = _group_tensor_slots_by_memory(module, "map_weights")
    mm, stat = _map_copy_on_write(path)

    # Make every view first, so an error leaves the module as it was. One view per memory, under whichever
    # of its names the file stores.
    views: List[Tuple[List[_TensorSlot], torch.Tensor]] = []
    missing: List[_TensorSlot] = []
    for slots in groups:
        stored = [slot for slot in slots if prefix + slot.name in entries]
        if not stored:
            missing.extend(slots)
            continue
        name = prefix + stored[0].name
        view = _view_stored_tensor(
            mm, data_start, entries[name], stored[0].tensor, name
        )
        views.append((slots, view))
    _report_missing_tensors(module, path, entries, missing, strict, bool(views))

    with torch.no_grad():
        for slots, view in views:
            _point_tensors_at_view(slots, view)
    _register_mapped_file(mm, path, stat)
    result = MappedWeights(
        mapped_bytes=sum(view.nbytes for _, view in views),
        missing=[slot.name for slot in missing],
    )
    logger.info(f"Mapped {result.mapped_bytes / 2**20:.1f} MiB of weights from {path}")
    return result


def _map_copy_on_write(path: str) -> Tuple[mmap.mmap, os.stat_result]:
    """Map the whole file private and writable: writes stay in this process and never reach the file."""
    with open(path, "rb") as f:
        if sys.platform == "win32":
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
        else:
            # ACCESS_COPY without reserving memory for every page that could be written, which Linux refuses
            # for a file larger than RAM and swap (with vm.overcommit_memory=2 it still does).
            mm = mmap.mmap(
                f.fileno(),
                0,
                flags=mmap.MAP_PRIVATE | _MAP_NORESERVE,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        return mm, os.fstat(f.fileno())


def _view_stored_tensor(
    mm: mmap.mmap,
    data_start: int,
    entry: Dict[str, Any],
    like: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """A zero-copy tensor over the bytes the file stores for ``entry``, which must have ``like``'s dtype and
    shape. The entry's offsets were checked with the header."""
    dtype = _SAFETENSORS_DTYPES.get(str(entry.get("dtype")))
    if dtype is None:
        raise ValueError(f"{name}: dtype {entry.get('dtype')} is not supported")
    if dtype != like.dtype:
        raise ValueError(
            f"{name}: file dtype {entry.get('dtype')} != model dtype {like.dtype} "
            "(write the weights with save_weights_for_mapping)"
        )
    if list(entry["shape"]) != list(like.shape):
        raise ValueError(
            f"{name}: file shape {entry['shape']} != model shape {list(like.shape)}"
        )
    if not like.numel():  # frombuffer rejects count=0
        return torch.empty(like.shape, dtype=dtype)
    return torch.frombuffer(
        mm,
        dtype=dtype,
        count=like.numel(),
        offset=data_start + entry["data_offsets"][0],
    ).view(like.shape)


def _report_missing_tensors(
    module: torch.nn.Module,
    path: str,
    entries: Dict[str, Any],
    missing: List[_TensorSlot],
    strict: bool,
    mapped_any: bool,
) -> None:
    """Raise, with ``strict``, if a parameter or persistent buffer is missing from the file. Warn about
    missing tensors left on ``meta``, and, without ``strict``, about missing ones that keep their values.
    """
    persistent = set(module.state_dict())
    required = [slot.name for slot in missing if slot.name in persistent]
    if required and strict:
        hint = (
            f" The module's names look like {required[0]!r} and the file's like "
            f"{next(iter(entries))!r}; pass prefix= if the file's have a prefix the module's don't."
            if entries and not mapped_any
            else ""
        )
        raise ValueError(
            f"{len(required)} tensors are not in {path}: {required[:8]}.{hint}"
        )
    on_meta = [slot.name for slot in missing if slot.tensor.is_meta]
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


def _point_tensors_at_view(slots: List[_TensorSlot], view: torch.Tensor) -> None:
    """Point the tensors in ``slots``, which are one memory, at ``view``. A real tensor keeps its object and
    gets ``.data = view``; one on ``meta`` is replaced, and distinct ones stay distinct, over the one view.
    """
    replacements: Dict[int, torch.Tensor] = {}
    for slot in slots:
        tensor = slot.tensor
        if not tensor.is_meta:
            tensor.data = view
            continue
        if id(tensor) not in replacements:
            alias = view.view_as(view) if replacements else view
            replacements[id(tensor)] = (
                torch.nn.Parameter(alias, requires_grad=tensor.requires_grad)
                if isinstance(tensor, torch.nn.Parameter)
                else alias
            )
        slot.table[slot.key] = replacements[id(tensor)]


class _MappedFile(NamedTuple):
    """A file mapped by ``map_weights``. A storage between ``base`` and ``base + size`` is a mapped weight."""

    base: int
    size: int
    # Keeps the entry exactly as long as some view keeps the mmap alive.
    mm: "weakref.ReferenceType[mmap.mmap]"
    path: str
    # (st_dev, st_ino, st_size, st_mtime_ns) when it was mapped.
    file_id: Tuple[int, int, int, int]
    # Whether MADV_PAGEOUT drops its unwritten pages: Linux skips them unless the process owns the file or
    # may write to it.
    can_pageout: bool


_mapped_files: List[_MappedFile] = []
_mapped_files_lock = threading.Lock()


def _get_file_id(stat: os.stat_result) -> Tuple[int, int, int, int]:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _register_mapped_file(mm: mmap.mmap, path: str, stat: os.stat_result) -> None:
    can_pageout = (
        os.name != "posix"
        or os.geteuid() in (0, stat.st_uid)
        or os.access(path, os.W_OK)
    )
    mapped = _MappedFile(
        _get_mapping_address(mm),
        len(mm),
        weakref.ref(mm),
        path,
        _get_file_id(stat),
        can_pageout,
    )
    with _mapped_files_lock:
        _mapped_files.append(mapped)


def _get_mapping_address(mm: mmap.mmap) -> int:
    anchor = ctypes.c_char.from_buffer(mm)
    address = ctypes.addressof(anchor)
    del anchor  # a live export would keep the mmap from closing
    return address


def _get_live_mapped_files() -> List[_MappedFile]:
    """The mapped files that some view still uses; the rest are forgotten."""
    with _mapped_files_lock:
        _mapped_files[:] = [m for m in _mapped_files if m.mm() is not None]
        return list(_mapped_files)


def has_mapped_weights() -> bool:
    """Whether a file mapped by ``map_weights`` is still in use."""
    return bool(_get_live_mapped_files())


def raise_if_mapped_files_changed(tensors: Iterable[torch.Tensor]) -> None:
    """Raise if a file that one of ``tensors`` is mapped from was removed or changed since ``map_weights``
    mapped it. A mapped weight is read when first used: from a file rewritten since, it reads the new
    contents, and past the end of a file truncated since, the read kills the process with SIGBUS.
    """
    addresses = {
        t.untyped_storage().data_ptr()
        for t in tensors
        if not t.is_meta
        and t.layout == torch.strided
        and not is_traceable_wrapper_subclass(t)
    }
    for mapped in _get_live_mapped_files():
        if not any(mapped.base <= a < mapped.base + mapped.size for a in addresses):
            continue
        try:
            stat = os.stat(mapped.path)
        except FileNotFoundError:
            raise ValueError(
                f"The weights file {mapped.path} was removed after map_weights mapped it. Keep it in "
                "place while the model is in use."
            ) from None
        if _get_file_id(stat) != mapped.file_id:
            raise ValueError(
                f"The weights file {mapped.path} changed after map_weights mapped it: it is "
                f"{stat.st_size} bytes, {mapped.file_id[2]} when mapped, and was modified or replaced "
                "since. Its weights are read when compilation first uses them, so map them again."
            )


def release_mapped_storage(storage: torch.UntypedStorage) -> Optional[int]:
    """Drop the resident pages of a storage that views a file mapped by ``map_weights``, and return how many
    bytes left this process's memory; None if the storage isn't such a view. The tensor stays valid: its
    pages are read back when it is used again, from the file, or from swap for pages it has written.

    Where the process owns the file or may write to it, MADV_PAGEOUT (Linux 5.4+) drops the pages and keeps
    writes made to them; it leaves pages another process maps too. Elsewhere, MADV_DONTNEED drops them and
    discards such writes: those pages revert to the file's data. Only pages entirely inside the storage are
    dropped, so a weight smaller than a page releases nothing. On Windows nothing is.
    """
    address, nbytes = storage.data_ptr(), storage.nbytes()
    if not nbytes:
        return None
    for mapped in _get_live_mapped_files():
        mm = mapped.mm()
        if mm is None or not (
            mapped.base <= address and address + nbytes <= mapped.base + mapped.size
        ):
            continue
        # Only pages entirely inside the storage: a page at either end may hold a neighbour's data.
        page = mmap.PAGESIZE
        start = -(-(address - mapped.base) // page) * page
        end = (address - mapped.base + nbytes) // page * page
        return _drop_pages(mapped, mm, start, end - start) if end > start else 0
    return None


def _drop_pages(mapped: _MappedFile, mm: mmap.mmap, start: int, length: int) -> int:
    """Drop the pages of ``mm[start:start + length]``; return how many bytes left this process's memory."""
    address = mapped.base + start
    before = _count_resident_bytes(address, length)
    for advice in (_MADV_PAGEOUT if mapped.can_pageout else None, _MADV_DONTNEED):
        if advice is None:
            continue
        try:
            mm.madvise(advice, start, length)
        except OSError:  # MADV_PAGEOUT on a kernel before 5.4
            continue
        # Measured, since MADV_PAGEOUT also succeeds for pages it skips.
        after = _count_resident_bytes(address, length)
        return length if before is None or after is None else max(0, before - after)
    warnings.warn(
        "Pages of memory-mapped weights can't be dropped on this platform, so released weights stay "
        "resident until their file is unmapped.",
        RuntimeWarning,
    )
    return 0


def _count_resident_bytes(address: int, length: int) -> Optional[int]:
    """How much of the page-aligned range at ``address`` this process has in memory, from its page map;
    None where that can't be read."""
    page = mmap.PAGESIZE
    try:
        with open("/proc/self/pagemap", "rb") as f:
            f.seek(address // page * 8)
            entries = bytearray(f.read(length // page * 8))
    except OSError:
        return None
    # Bit 63 of a page's entry: the page is present.
    return int((torch.frombuffer(entries, dtype=torch.int64) < 0).sum()) * page


def save_weights_for_mapping(
    module: torch.nn.Module,
    path: Union[str, "os.PathLike[str]"],
    *,
    source: Optional[WeightSource] = None,
) -> None:
    """Write ``module``'s parameters and buffers, with their dtypes, to a safetensors file ``map_weights`` can
    map. A tied weight, or tensors that are one view of a storage, are written once, under the first name.

    Tensors are copied to the host and written one at a time. With ``source``, each is read from
    ``source(name, tensor)`` instead, which returns its data in any dtype, as one tensor or as chunks that
    concatenate along dimension 0 into it, and raises KeyError for a name it doesn't store. A real tensor
    that ``source`` doesn't store is written from the module (with a warning if it is persistent); one on
    ``meta`` raises. Each chunk is cast to the tensor's dtype and written as it comes. Converting a
    checkpoint this way holds one chunk in memory at a time: build the model on ``meta``, change it as the
    compiled model will be (dtypes, shapes), and let ``source`` read the checkpoint. The file is written
    under a new, random name next to ``path`` and renamed when complete.
    """
    tensors: List[Tuple[str, torch.Tensor]] = []
    for slots in _group_tensor_slots_by_memory(module, "save_weights_for_mapping"):
        name, tensor = slots[0].name, slots[0].tensor
        if tensor.is_meta and source is None:
            raise ValueError(
                f"{name} is on the meta device; pass source= to read it, or materialize it"
            )
        if tensor.dtype not in _SAFETENSORS_DTYPE_NAMES:
            raise ValueError(f"{name}: dtype {tensor.dtype} has no safetensors name")
        tensors.append((name, tensor))
    # Largest element size first, so every tensor's offset stays aligned to its dtype.
    tensors.sort(key=lambda named: -named[1].element_size())

    path = os.fspath(path)
    partial = f"{path}.{os.urandom(8).hex()}.partial"
    # O_EXCL creates a new file: it never writes through a link or into a file someone else made.
    fd = os.open(
        partial,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o666,
    )
    not_in_source: List[str] = []
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(_encode_safetensors_header(tensors))
            for name, tensor in tensors:
                chunks: Union[torch.Tensor, Iterable[torch.Tensor]] = tensor
                if source is not None:
                    try:
                        chunks = source(name, tensor)
                    except KeyError:
                        if tensor.is_meta:
                            raise ValueError(
                                f"{name} is not in the source, and on the meta device it has no values to "
                                "write. A buffer a checkpoint doesn't store, such as non-persistent rotary "
                                "frequencies, must be real: build the model with accelerate's "
                                "init_empty_weights(), which keeps buffers real, or set the buffer before "
                                "writing."
                            ) from None
                        not_in_source.append(name)
                _write_tensor_chunks(f, name, tensor, chunks)
        os.replace(partial, path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    persistent = set(module.state_dict()) if not_in_source else set()
    kept = [name for name in not_in_source if name in persistent]
    if kept:
        logger.warning(
            f"{len(kept)} parameters and persistent buffers are not in the source, so their values in the "
            f"module were written: {kept[:8]}"
        )


def _encode_safetensors_header(tensors: List[Tuple[str, torch.Tensor]]) -> bytes:
    """The safetensors header for ``tensors``, stored in this order: its 8-byte little-endian length, then
    the JSON, padded so the data starts 8-byte aligned."""
    header: Dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, tensor in tensors:
        nbytes = tensor.numel() * tensor.element_size()
        header[name] = {
            "dtype": _SAFETENSORS_DTYPE_NAMES[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    return struct.pack("<Q", len(blob)) + blob


def _write_tensor_chunks(
    f: "io.BufferedWriter",
    name: str,
    like: torch.Tensor,
    chunks: Union[torch.Tensor, Iterable[torch.Tensor]],
) -> None:
    """Write ``chunks``, which concatenate along dimension 0 into the data of ``like``, one at a time, each
    cast to ``like``'s dtype."""
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
    casts each chunk. A name the checkpoint doesn't store raises KeyError. ``path`` may also be a sharded
    checkpoint's ``model.safetensors.index.json``.
    """
    try:
        from safetensors import safe_open
    except ImportError as e:
        raise ImportError(
            "checkpoint_source reads checkpoints with the safetensors package: pip install safetensors"
        ) from e

    path = os.fspath(path)
    shard_by_name: Dict[str, str] = {}  # for a sharded checkpoint's index
    if path.endswith(".json"):
        with open(path) as f:
            shard_by_name = json.load(f)["weight_map"]
    directory = os.path.dirname(path)
    # Each file is opened once, with the names it stores.
    open_files: Dict[str, Tuple[Any, Set[str]]] = {}

    def get_checkpoint_file(stored_name: str) -> Any:
        """The open safetensors file that stores ``stored_name``; KeyError if none does."""
        if shard_by_name and stored_name not in shard_by_name:
            raise KeyError(stored_name)
        file = (
            os.path.join(directory, shard_by_name[stored_name])
            if shard_by_name
            else path
        )
        if file not in open_files:
            handle = safe_open(file, framework="pt", device="cpu")
            open_files[file] = (handle, set(handle.keys()))
        handle, names = open_files[file]
        if stored_name not in names:
            raise KeyError(stored_name)
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
        if stored.get_dtype() not in _SAFETENSORS_DTYPES:
            raise ValueError(f"{name}: dtype {stored.get_dtype()} is not supported")
        row_bytes = _SAFETENSORS_DTYPES[stored.get_dtype()].itemsize
        for size in shape[1:]:
            row_bytes *= size
        rows = max(1, chunk_bytes // max(1, row_bytes))
        return (stored[i : i + rows] for i in range(0, shape[0], rows))

    return read_tensor_chunks
