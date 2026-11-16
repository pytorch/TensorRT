# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Prepared, value-bearing inputs for validation and measurement."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import torch
from torch.utils import _pytree

_StorageKey = tuple[torch.device, int]


class VerificationError(ValueError):
    """A verification request cannot be evaluated under the declared protocol."""


@dataclass
class InputSample:
    """One prepared invocation; tensors stay on the caller-selected device.

    ``profile`` describes a tested point, rather than claiming coverage of a
    dynamic profile. Values must be tensors or ordinary Python pytree leaves.
    Input snapshots preserve tensor storage aliasing and strides. Cross-aliasing
    inputs with module parameters/buffers is unsupported by the current protocol.
    """

    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    profile: str | None = "default"
    sample_id: str = ""
    label: str = ""

    def __post_init__(self) -> None:
        """Normalize call containers and validate report-facing metadata."""
        if not isinstance(self.args, (tuple, list)):
            raise VerificationError("InputSample args must be a tuple or list")
        if not isinstance(self.kwargs, dict):
            raise VerificationError("InputSample kwargs must be a dictionary")
        self.args = tuple(self.args)
        self.kwargs = dict(self.kwargs)
        if not all(isinstance(key, str) for key in self.kwargs):
            raise VerificationError("InputSample keyword names must be strings")
        if not isinstance(self.sample_id, str) or not isinstance(self.label, str):
            raise VerificationError("InputSample IDs and labels must be strings")
        if self.profile is not None and not isinstance(self.profile, str):
            raise VerificationError("InputSample profile must be a string or None")


def _storage_key(tensor: torch.Tensor) -> _StorageKey:
    """Return a process-local identity for a supported tensor's storage."""
    if tensor.layout != torch.strided or tensor.is_quantized:
        raise VerificationError("Verification requires dense, unquantized tensors")
    if tensor.device.type not in {"cpu", "cuda"}:
        raise VerificationError("Verification supports CPU and CUDA tensors")
    return tensor.device, tensor.untyped_storage()._cdata


def _snapshot(sample: InputSample) -> InputSample:
    """Copy storage once, preserving repeated leaves, views, and dtype views."""
    tensors, _ = _pytree.tree_flatten((sample.args, sample.kwargs))
    memo: dict[int, torch.Tensor] = {}
    storages: dict[_StorageKey, Any] = {}
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor) or id(tensor) in memo:
            continue
        key = _storage_key(tensor)
        if key not in storages:
            storages[key] = tensor.untyped_storage().clone()
        clone = torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
            storages[key], tensor.storage_offset(), tensor.size(), tensor.stride()
        )
        if tensor.is_conj():
            clone = clone.conj()
        if tensor.is_neg():
            clone = clone._neg_view()
        if tensor.requires_grad:
            clone.requires_grad_(True)
        memo[id(tensor)] = clone
    try:
        return copy.deepcopy(sample, memo)
    except Exception as error:
        raise VerificationError(
            "InputSample contains values that cannot be copied"
        ) from error


def _fingerprint(sample: InputSample) -> str:
    """Digest the call structure, tensor metadata and exact logical values."""
    leaves, spec = _pytree.tree_flatten((sample.args, sample.kwargs))
    digest = hashlib.sha256(str(spec).encode())
    storage_ids: dict[_StorageKey, int] = {}
    tensor_ids: dict[int, int] = {}
    for leaf in leaves:
        if isinstance(leaf, torch.Tensor):
            storage_key = _storage_key(leaf)
            storage_ids.setdefault(storage_key, len(storage_ids))
            tensor_ids.setdefault(id(leaf), len(tensor_ids))
            descriptor = [
                str(leaf.dtype),
                str(leaf.device),
                list(leaf.shape),
                list(leaf.stride()),
                leaf.storage_offset(),
                storage_ids[storage_key],
                tensor_ids[id(leaf)],
                leaf.requires_grad,
            ]
            data = leaf.detach().resolve_conj().resolve_neg().cpu().contiguous()
            # Flatten before viewing bytes: scalar tensors cannot change element size.
            payload = data.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(json.dumps(descriptor, separators=(",", ":")).encode())
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        elif leaf is None or isinstance(leaf, (bool, int, float, str)):
            if isinstance(leaf, float) and not math.isfinite(leaf):
                raise VerificationError(
                    "Prepared input floating-point values must be finite"
                )
            digest.update(type(leaf).__name__.encode())
            digest.update(json.dumps(leaf, allow_nan=False).encode())
        else:
            raise VerificationError(
                f"Unsupported prepared input leaf: {type(leaf).__name__}"
            )
    return digest.hexdigest()
