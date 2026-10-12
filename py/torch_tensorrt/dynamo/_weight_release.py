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
from typing import Callable, Dict, Iterable, List

import torch

logger = logging.getLogger(__name__)

# A handler returns True if it released the storage (for example, by dropping the pages of a
# memory-mapped weight); release_storage falls back to resizing the storage to zero.
ReleaseHandler = Callable[[torch.UntypedStorage], bool]
_release_handlers: List[ReleaseHandler] = []


def register_release_handler(handler: ReleaseHandler) -> None:
    """Let a storage owner release its own storages (for example, file-backed weights)."""
    if handler not in _release_handlers:
        _release_handlers.append(handler)


def unregister_release_handler(handler: ReleaseHandler) -> None:
    if handler in _release_handlers:
        _release_handlers.remove(handler)


def _get_attr(module: torch.nn.Module, target: str) -> object:
    obj: object = module
    for part in target.split("."):
        obj = getattr(obj, part)
    return obj


def weight_storages(gm: torch.fx.GraphModule) -> Dict[int, torch.UntypedStorage]:
    """The storages read by ``gm``'s ``get_attr`` nodes, keyed by data pointer (tied weights collapse)."""
    found: Dict[int, torch.UntypedStorage] = {}
    for node in gm.graph.nodes:
        if node.op != "get_attr":
            continue
        try:
            value = _get_attr(gm, str(node.target))
        except AttributeError:
            continue
        if isinstance(value, torch.Tensor) and not value.is_meta:
            storage = value.untyped_storage()
            if storage.nbytes():
                found[storage.data_ptr()] = storage
    return found


def parameter_storages(module: torch.nn.Module) -> Dict[int, torch.UntypedStorage]:
    """The storages of ``module``'s parameters, keyed by data pointer. Buffers are left out: mutable
    buffers are lifted into engine inputs (no partition reads them through ``get_attr``) and are
    inlined back into the compiled module afterwards, so they must survive compilation.
    """
    found: Dict[int, torch.UntypedStorage] = {}
    for tensor in module.parameters():
        if not tensor.is_meta:
            storage = tensor.untyped_storage()
            if storage.nbytes():
                found[storage.data_ptr()] = storage
    return found


def release_storage(storage: torch.UntypedStorage) -> int:
    """Release a storage's memory; return the bytes released (0 if it could not be released)."""
    nbytes = int(storage.nbytes())
    if nbytes == 0:
        return 0
    for handler in _release_handlers:
        if handler(storage):
            return nbytes
    try:
        storage.resize_(0)
    except RuntimeError as e:  # e.g. a non-resizable storage over an external buffer
        logger.debug(f"Could not release a {nbytes}-byte weight storage: {e}")
        return 0
    return nbytes


class WeightReleaser:
    """Counts, for each weight storage, the partitions that read it, and releases a storage once the
    last TensorRT partition that reads it has been built. Storages read by PyTorch partitions are
    counted but never decremented, so they are never released."""

    def __init__(self, partitions: Iterable[torch.fx.GraphModule]):
        self.uses: Counter[int] = Counter()
        for partition in partitions:
            self.uses.update(weight_storages(partition).keys())

    def release_unread(
        self, candidates: Dict[int, torch.UntypedStorage], keep: Iterable[int] = ()
    ) -> int:
        """Release the candidate storages that no partition reads (and that are not in ``keep``)."""
        kept = set(self.uses) | set(keep)
        released = sum(
            release_storage(s) for ptr, s in candidates.items() if ptr not in kept
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
            self.uses[ptr] -= 1
            if self.uses[ptr] <= 0:
                released += release_storage(storage)
        if released:
            logger.info(
                f"Released {released / 2**20:.0f} MiB of weights after building {name}"
            )
        return released
