# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Release model weights during compilation, as soon as no TensorRT engine still to be built reads them.

TensorRT holds a network's weights by reference only until its engine is built, and a built engine carries
its own copy. With several engines (resource partitioning, PyTorch fallbacks), a weight is dead once the
last engine that reads it is built, yet the model, the ``ExportedProgram`` and the partitioned graph
modules keep it referenced until compilation ends. ``compile_module`` uses these helpers, when
``release_weights_after_build`` is set, to free that memory: the storage itself is released, because
dropping references cannot free what the caller's model still points to.
"""

import logging
from collections import Counter
from typing import Dict, Iterable

import torch
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch_tensorrt.dynamo.tools.weight_mapping import release_mapped_storage

logger = logging.getLogger(__name__)


def _get_attr(module: torch.nn.Module, target: str) -> object:
    obj: object = module
    for part in target.split("."):
        obj = getattr(obj, part)
    return obj


def _add_storage(found: Dict[int, torch.UntypedStorage], tensor: torch.Tensor) -> None:
    if tensor.is_meta or is_traceable_wrapper_subclass(tensor):
        return  # a wrapper subclass's storage is a placeholder; its inner tensors are not reachable here
    storage = tensor.untyped_storage()
    if storage.nbytes():
        found[storage.data_ptr()] = storage


def weight_storages(gm: torch.fx.GraphModule) -> Dict[int, torch.UntypedStorage]:
    """The storages read by ``gm``'s own ``get_attr`` nodes (not its submodules'), keyed by data
    pointer (tied weights collapse)."""
    found: Dict[int, torch.UntypedStorage] = {}
    for node in gm.graph.nodes:
        if node.op != "get_attr":
            continue
        try:
            value = _get_attr(gm, str(node.target))
        except AttributeError:
            continue
        if isinstance(value, torch.Tensor):
            _add_storage(found, value)
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


def release_storage(storage: torch.UntypedStorage) -> int:
    """Release a storage's memory; return the bytes released (0 if it could not be released)."""
    nbytes = int(storage.nbytes())
    if nbytes == 0:
        return 0
    # A weight mapped by map_weights is a view of a file: resize_ cannot free it, but its pages can be
    # dropped (re-read from the file if used again).
    if release_mapped_storage(storage):
        return nbytes
    try:
        storage.resize_(0)
    except RuntimeError as e:  # e.g. a non-resizable storage over an external buffer
        logger.debug(f"Could not release a {nbytes}-byte weight storage: {e}")
        return 0
    return nbytes


class WeightReleaser:
    """Counts, for each weight storage, the graphs that read it, and releases a storage once the last
    TensorRT partition that reads it has been built. Graphs that run in PyTorch (PyTorch partitions,
    the partitioned module's own graph) are counted but never built, so their storages are never
    released."""

    def __init__(self, readers: Iterable[torch.fx.GraphModule]):
        self.uses: Counter[int] = Counter()
        for reader in readers:
            self.uses.update(weight_storages(reader).keys())

    def release_unread(self, candidates: Dict[int, torch.UntypedStorage]) -> int:
        """Release the candidate storages that no graph reads."""
        released = sum(
            release_storage(s) for ptr, s in candidates.items() if ptr not in self.uses
        )
        if released:
            logger.info(
                f"Released {released / 2**20:.0f} MiB of weights that no partition reads"
            )
        return released

    def built(self, storages: Dict[int, torch.UntypedStorage], name: str = "") -> int:
        """Record that a TensorRT partition reading ``storages`` has been built; release the ones done."""
        released = 0
        for ptr, storage in storages.items():
            if ptr not in self.uses:
                continue  # not counted at partition time (e.g. rebound since): another reader may hold it
            self.uses[ptr] -= 1
            if self.uses[ptr] == 0:
                # A later allocation at this address is a different storage.
                del self.uses[ptr]
                released += release_storage(storage)
        if released:
            logger.info(
                f"Released {released / 2**20:.0f} MiB of weights after building {name}"
            )
        return released
