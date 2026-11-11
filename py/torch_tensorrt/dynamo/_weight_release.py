# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Release model weights during compilation, as soon as no TensorRT engine still to be built reads them.

TensorRT holds a network's weights by reference only until its engine is built, and a built engine carries
its own copy. With several engines (resource partitioning, PyTorch fallbacks), a weight is dead once the
last engine that reads it is built, yet the model, the ``ExportedProgram`` and the partitioned graph
modules keep it referenced until compilation ends. ``compile_module`` uses these helpers, when
``release_weights_after_build`` isn't ``False``, to free that memory: the storage itself is released,
because dropping references cannot free what the caller's model still points to. By default (``None``)
only weights mapped by ``map_weights`` are released, which only drops pages the file can give back.
"""

import contextlib
import logging
import weakref
from collections import Counter
from typing import Dict, Iterable, Iterator, List, Optional

import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch_tensorrt.dynamo.tools.weight_mapping import release_mapped_storage

logger = logging.getLogger(__name__)

# The tensors a sparse tensor keeps its data in; any of them may share memory with a dense weight.
_SPARSE_PARTS = {
    torch.sparse_coo: ("_indices", "_values"),
    torch.sparse_csr: ("crow_indices", "col_indices", "values"),
    torch.sparse_bsr: ("crow_indices", "col_indices", "values"),
    torch.sparse_csc: ("ccol_indices", "row_indices", "values"),
    torch.sparse_bsc: ("ccol_indices", "row_indices", "values"),
}


def _get_attr(module: torch.nn.Module, target: str) -> object:
    obj: object = module
    for part in target.split("."):
        obj = getattr(obj, part)
    return obj


def _get_storage_key(tensor: torch.Tensor) -> Optional[int]:
    """The data pointer of the storage holding ``tensor``, or None if it has no storage of its own."""
    # No real storage: a meta tensor, a sparse one, a fake one (torch.compile's backend), or a wrapper
    # subclass, whose storage is a placeholder and whose inner tensors are not reachable here.
    if (
        tensor.is_meta
        or tensor.layout != torch.strided
        or is_traceable_wrapper_subclass(tensor)
        or isinstance(tensor, FakeTensor)
    ):
        return None
    storage = tensor.untyped_storage()
    return storage.data_ptr() if storage.nbytes() else None


def _add_storage(found: Dict[int, torch.UntypedStorage], tensor: torch.Tensor) -> None:
    key = _get_storage_key(tensor)
    if key is not None:
        found[key] = tensor.untyped_storage()


def weight_storages(gm: torch.fx.GraphModule) -> Dict[int, torch.UntypedStorage]:
    """The storages read by ``gm``'s own ``get_attr`` nodes, and by the subgraphs they fetch (``torch.cond``
    branches), but not by the submodules ``gm`` calls; keyed by data pointer (tied weights collapse).
    """
    found: Dict[int, torch.UntypedStorage] = {}
    for node in gm.graph.nodes:
        if node.op != "get_attr":
            continue
        try:
            value = _get_attr(gm, str(node.target))
        except AttributeError:
            continue
        if isinstance(value, torch.fx.GraphModule):
            found.update(weight_storages(value))
        elif isinstance(value, torch.Tensor):
            _add_storage(found, value)
            for part in _SPARSE_PARTS.get(value.layout, ()):
                _add_storage(found, getattr(value, part)())
    return found


def parameter_storages(module: torch.nn.Module) -> Dict[int, torch.UntypedStorage]:
    """The storages of ``module``'s parameters, keyed by data pointer. Buffers are left out: mutable
    buffers are lifted into engine inputs (no partition reads them through ``get_attr``) and are
    inlined back into the compiled module afterwards, so they must survive compilation.
    """
    found: Dict[int, torch.UntypedStorage] = {}
    for tensor in module.parameters():
        _add_storage(found, tensor)
    return found


def release_storage(storage: torch.UntypedStorage, mapped_only: bool = False) -> int:
    """Release a storage's memory; return the bytes released (0 if it could not be released).
    With ``mapped_only``, only a mapped weight is released."""
    nbytes = int(storage.nbytes())
    if nbytes == 0:
        return 0
    # A weight mapped by map_weights is a view of a file: resize_ cannot free it, but its pages can be
    # dropped (re-read from the file if used again).
    released: Optional[int] = release_mapped_storage(storage)
    if released is not None:
        return released
    if mapped_only:
        return 0
    try:
        storage.resize_(0)
    except RuntimeError as e:  # e.g. a non-resizable storage over an external buffer
        logger.debug(f"Could not release a {nbytes}-byte weight storage: {e}")
        return 0
    return nbytes


class WeightReleaser:
    """Counts, for each storage of a module's parameters and buffers, the graphs that read it, and releases a
    storage once the last TensorRT partition that reads it has been built. Graphs that run in PyTorch (PyTorch
    partitions, the partitioned module's own graph) are counted but never built, so their storages are never
    released. The module's tensors over a released storage are pointed at empty tensors, so using one raises a
    shape error instead of reading freed memory. With ``mapped_only``, only weights mapped by ``map_weights``
    are released."""

    def __init__(self, module: torch.nn.Module, mapped_only: bool = False):
        self.mapped_only = mapped_only
        # Taken before partitioning, which registers every tensor a partition reads as that partition's buffer,
        # including the tensor constants lifted from the forward (a global, a cached tensor), whose memory the
        # caller still uses. Held weakly: conversion frees the folded constants it deletes.
        self._tensors: Dict[int, List["weakref.ref[torch.Tensor]"]] = {}
        for tensor in [*module.parameters(), *module.buffers()]:
            key = _get_storage_key(tensor)
            if key is not None:
                self._tensors.setdefault(key, []).append(weakref.ref(tensor))
        self._parameter_keys = set(parameter_storages(module))
        self.uses: Counter[int] = Counter()
        self.released = 0

    def count_readers(self, readers: Iterable[torch.fx.GraphModule]) -> None:
        for reader in readers:
            self.uses.update(
                key for key in weight_storages(reader) if key in self._tensors
            )

    def _release(self, key: int) -> int:
        tensors = [
            tensor
            for ref in self._tensors.pop(key, [])
            # A tensor rebound since (e.g. moved between devices) no longer reads this storage.
            if (tensor := ref()) is not None and _get_storage_key(tensor) == key
        ]
        if not tensors:
            return 0  # already freed, such as a folded constant that conversion deleted
        storage = tensors[0].untyped_storage()
        released = release_storage(storage, self.mapped_only)
        if not storage.nbytes():
            for tensor in tensors:
                tensor.data = tensor.new_empty(0)
        return released

    def release_unread_storages(self) -> int:
        """Release the module's parameters that no graph reads."""
        released = sum(
            self._release(key) for key in self._parameter_keys - set(self.uses)
        )
        if released:
            logger.info(
                f"Released {released / 2**20:.0f} MiB of weights that no partition reads"
            )
        self.released += released
        return released

    def release_storages_after_build(self, read: Iterable[int], name: str) -> int:
        """Record that the TensorRT partition ``name``, which read the storages keyed by ``read``, has been
        built; release the ones no other graph reads."""
        released = 0
        for key in read:
            if key not in self.uses:
                continue  # not counted at partition time (e.g. rebound since): another reader may hold it
            self.uses[key] -= 1
            if self.uses[key] == 0:
                # A later allocation at this address is a different storage.
                del self.uses[key]
                released += self._release(key)
        if released:
            logger.info(
                f"Released {released / 2**20:.0f} MiB of weights after building {name}"
            )
        self.released += released
        return released


@contextlib.contextmanager
def report_released_weights_on_failure(
    releaser: Optional[WeightReleaser], name: str
) -> Iterator[None]:
    """Build an engine; if that fails after weights were released, say what the source model lost."""
    try:
        yield
    except Exception:
        # Mapped weights stay valid: their pages are read back from the file.
        if releaser is not None and releaser.released and not releaser.mapped_only:
            logger.error(
                f"Building {name} failed after release_weights_after_build released "
                f"{releaser.released / 2**20:.0f} MiB of the source model's weights; reload the model "
                "before running it in PyTorch."
            )
        raise
