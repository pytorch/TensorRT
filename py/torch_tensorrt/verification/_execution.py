# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Isolated eval-mode execution and mutation checks shared by both APIs."""

from __future__ import annotations

import copy
import random
import sys
from contextlib import contextmanager
from typing import Any, Iterator, cast

import torch
from torch.utils import _pytree

from ._inputs import InputSample, VerificationError, _snapshot, _storage_key

_StorageKey = tuple[torch.device, int]
_TensorMetadata = tuple[_StorageKey, tuple[int, ...], tuple[int, ...], int, int | None]


@contextmanager
def _preserve_rng(seed: int) -> Iterator[None]:
    """Preserve caller RNG state while validating the requested seed."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed >= 2**64:
        raise VerificationError("seed must be an integer in [0, 2**64)")
    python_state = random.getstate()
    numpy = cast(Any, sys.modules.get("numpy"))
    numpy_state = numpy.random.get_state() if numpy is not None else None
    devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        if numpy_state is not None:
            numpy.random.set_state(numpy_state)


def _seed(seed: int) -> None:
    """Reset Torch, Python, and loaded NumPy RNGs for one invocation."""
    torch.manual_seed(seed)
    random.seed(seed)
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        numpy.random.seed(seed % (2**32))


def _isolated_module(module: torch.nn.Module) -> torch.nn.Module:
    """Deep-copy a module into eval mode without retaining caller tensor state."""
    if not isinstance(module, torch.nn.Module):
        raise VerificationError("Verification requires torch.nn.Module instances")
    try:
        isolated = copy.deepcopy(module)
        if {id(sub) for sub in isolated.modules()} & {
            id(sub) for sub in module.modules()
        }:
            raise VerificationError(
                "Deep-copied module still shares submodule objects with the original module"
            )
        original_storages = {
            _storage_key(tensor) for tensor in (*module.parameters(), *module.buffers())
        }
        if any(
            _storage_key(tensor) in original_storages
            for tensor in (*isolated.parameters(), *isolated.buffers())
        ):
            raise VerificationError(
                "Deep-copied module still shares tensor state with the original module"
            )
        try:
            isolated.eval()
        except NotImplementedError:
            # ExportedProgram.module() freezes mode in graph constants, and
            # deliberately disallows train()/eval(). Retain its capture semantics.
            if not isinstance(isolated, torch.fx.GraphModule):
                raise
        return isolated
    except VerificationError:
        raise
    except Exception as error:
        raise VerificationError(
            "Deep-copying the module failed during isolated verification"
        ) from error


def _tensors(sample: InputSample) -> list[torch.Tensor]:
    """Collect tensor leaves from an input sample's args and kwargs."""
    leaves, _ = _pytree.tree_flatten((sample.args, sample.kwargs))
    return [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]


def _reject_module_input_aliases(module: torch.nn.Module, sample: InputSample) -> None:
    """Reject input storage shared with a module parameter or buffer."""
    if not isinstance(module, torch.nn.Module):
        raise VerificationError("Verification requires torch.nn.Module instances")
    input_storages = {_storage_key(tensor) for tensor in _tensors(sample)}
    for tensor in (*module.parameters(), *module.buffers()):
        if _storage_key(tensor) in input_storages:
            raise VerificationError(
                "Input tensors must not share storage with module parameters or buffers"
            )


def _device(sample: InputSample, *modules: torch.nn.Module) -> torch.device:
    """Select the sole CUDA device in an invocation, or CPU when none exists."""
    devices = {tensor.device for tensor in _tensors(sample)}
    for module in modules:
        devices.update(tensor.device for tensor in module.parameters())
        devices.update(tensor.device for tensor in module.buffers())
    cuda_devices = {device for device in devices if device.type == "cuda"}
    if len(cuda_devices) > 1:
        raise VerificationError(
            "Verification supports at most one CUDA device per invocation"
        )
    if any(device.type not in {"cpu", "cuda"} for device in devices):
        raise VerificationError("Verification supports CPU and CUDA tensors")
    return next(iter(cuda_devices)) if cuda_devices else torch.device("cpu")


class _MutationGuard:
    """Snapshot registered tensor state and reject changes after an invocation."""

    def __init__(self, sample: InputSample, module: torch.nn.Module) -> None:
        """Capture tensor identities, metadata, and storage bytes."""
        self.module = module
        self.named_state = self._state()
        self.tensors = _tensors(sample) + list(self.named_state.values())
        self.metadata = [self._metadata(tensor) for tensor in self.tensors]
        self.storages: dict[_StorageKey, tuple[torch.Tensor, torch.Tensor]] = {}
        for tensor in self.tensors:
            key = _storage_key(tensor)
            if key not in self.storages:
                storage = tensor.untyped_storage()
                raw = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
                    storage, 0, (storage.nbytes(),), (1,)
                )
                self.storages[key] = (raw, raw.clone())

    def _state(self) -> dict[str, torch.Tensor]:
        """Return registered parameters and buffers under disjoint names."""
        return {
            **{
                f"parameter:{name}": value
                for name, value in self.module.named_parameters()
            },
            **{f"buffer:{name}": value for name, value in self.module.named_buffers()},
        }

    @staticmethod
    def _metadata(tensor: torch.Tensor) -> _TensorMetadata:
        """Capture tensor metadata and its mutation version when available."""
        try:
            version = tensor._version
        except RuntimeError:  # Inference tensors have no version counter.
            version = None
        return (
            _storage_key(tensor),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
            version,
        )

    def check(self) -> None:
        """Raise when registered state, tensor metadata, or bytes changed."""
        state = self._state()
        if state.keys() != self.named_state.keys() or any(
            state[key] is not value for key, value in self.named_state.items()
        ):
            raise VerificationError(
                "Module replaced registered state; verification requires stateless evaluation"
            )
        if any(
            self._metadata(tensor) != before
            for tensor, before in zip(self.tensors, self.metadata)
        ):
            raise VerificationError(
                "Input or module state mutated; verification requires stateless evaluation"
            )
        if any(not torch.equal(raw, before) for raw, before in self.storages.values()):
            raise VerificationError(
                "Input or module state mutated; verification requires stateless evaluation"
            )


def _prepare(
    module: torch.nn.Module, sample: InputSample, seed: int
) -> tuple[InputSample, _MutationGuard]:
    """Reset RNGs, snapshot inputs, and add mutation guards."""
    _seed(seed)
    prepared = _snapshot(sample)
    return prepared, _MutationGuard(prepared, module)


def _invoke(module: torch.nn.Module, sample: InputSample) -> Any:
    """Run one prepared module invocation with autograd disabled."""
    with torch.inference_mode():
        return module(*sample.args, **sample.kwargs)
