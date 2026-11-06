# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Capture region inputs from real model batches and save them without code.

``KernelGenerator(validation_inputs=...)`` supplies model-level batches. Before
any region is replaced, compilation runs the parent graph on them and keeps
the tensors that reach each generated region; every candidate is then checked
on these samples as well as on synthetic inputs. A run saves them with
``torch.save`` as a plain list of tensor lists, which loads with
``torch.load(weights_only=True)`` and so cannot run code from a shared run.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, Sequence, cast

import torch

from torch_tensorrt.region import RegionError

_LOGGER = logging.getLogger(__name__)

FILENAME = "samples.pt"
#: Batches used per region; each costs a model run and a check per evaluation.
MAX_SAMPLES = 8
#: Region input bytes kept from all validation batches of one compilation:
#: they stay in memory while its regions are generated, and each run saves its
#: region's share. Fewer batches are used past this; the first is always kept.
MAX_SAMPLE_BYTES = 1 << 30
# Bytes hashed at once, so a digest never copies a whole large tensor.
_HASH_CHUNK = 1 << 26

Sample = tuple[torch.Tensor, ...]


class _BoundaryRecorder(torch.fx.Interpreter):  # type: ignore[misc,unused-ignore]
    """Run a parent graph, keeping the tensors that enter the selected regions."""

    def __init__(self, module: torch.fx.GraphModule, regions: Iterable[Any]) -> None:
        super().__init__(module)
        self.regions = set(regions)
        self.inputs: dict[torch.fx.Node, Sample] = {}
        # Parameters, buffers and constants have the same value in every
        # batch, so all samples share one copy, which torch.save stores once.
        self.constants: dict[torch.fx.Node, torch.Tensor] = {}
        # Bytes of all copies, and of the latest batch's copies other than
        # the shared constants.
        self.nbytes = 0
        self.batch_nbytes = 0

    def run(self, *args: Any, **kwargs: Any) -> Any:
        self.batch_nbytes = 0
        return super().run(*args, **kwargs)

    def run_node(self, node: torch.fx.Node) -> Any:
        if node in self.regions:
            # Region live-ins are graph nodes holding tensors.
            operands = cast(Sequence[torch.fx.Node], node.args[1])
            self.inputs[node] = tuple(self.keep(operand) for operand in operands)
        return super().run_node(node)

    def keep(self, operand: torch.fx.Node) -> torch.Tensor:
        if operand in self.constants:
            return self.constants[operand]
        value: torch.Tensor = self.env[operand]
        # The search runs in other processes, so keep contiguous copies off
        # the GPU; a copy also never aliases the caller's batch.
        copy = value.detach().to(
            "cpu", copy=True, memory_format=torch.contiguous_format
        )
        self.nbytes += copy.nbytes
        if operand.op == "get_attr":
            self.constants[operand] = copy
        else:
            self.batch_nbytes += copy.nbytes
        return copy


def _contract(dtype: torch.dtype, shape: Sequence[int]) -> str:
    return f"{str(dtype).removeprefix('torch.')} {list(shape)}"


def _describe(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return _contract(value.dtype, value.shape)
    return type(value).__name__


def _arguments(batch: Any, gm: torch.fx.GraphModule, number: int) -> list[Any]:
    """Flatten a batch as the exported model flattens its inputs, and check it.

    A batch is a tensor or a tuple of positional inputs, followed by a dict of
    keyword inputs when the model was compiled with some; a dict alone holds
    keyword inputs only.
    """
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    # ep.module() records how export flattened (args, kwargs); a module that
    # export did not produce takes flat positional inputs.
    spec = getattr(gm, "_in_spec", None)
    takes = len(placeholders) if spec is None else spec.child(0).num_children
    names = [] if spec is None else list(spec.child(1).context)
    if isinstance(batch, torch.Tensor):
        batch = (batch,)
    if isinstance(batch, Mapping):
        args: tuple[Any, ...] = ()
        kwargs = dict(batch)
    elif isinstance(batch, (tuple, list)):
        args, kwargs = tuple(batch), {}
        if names and len(args) == takes + 1 and isinstance(args[-1], Mapping):
            args, kwargs = args[:-1], dict(args[-1])
    else:
        raise RegionError(
            f"KernelGenerator validation_inputs batch {number} must be a tensor, a "
            f"tuple of tensors, or a dict of keyword inputs; got {type(batch).__name__}"
        )
    if len(args) != takes:
        raise RegionError(
            f"KernelGenerator validation_inputs batch {number} has {len(args)} "
            f"positional inputs; the compiled model takes {takes}"
        )
    if set(kwargs) != set(names):
        raise RegionError(
            f"KernelGenerator validation_inputs batch {number} has keyword inputs "
            f"{sorted(kwargs)}; the compiled model takes {sorted(names)}, given as "
            "a dict after the positional inputs"
        )
    values = list(args)
    if spec is not None:
        try:
            # Nested inputs follow the exported structure; dict entries are
            # matched by key, not by order.
            values = spec.flatten_up_to((args, kwargs))
        except ValueError as error:
            raise RegionError(
                f"KernelGenerator validation_inputs batch {number} does not have "
                f"the structure of the compiled inputs: {error}"
            ) from None
    if len(values) != len(placeholders):
        raise RegionError(
            f"KernelGenerator validation_inputs batch {number} has {len(values)} "
            f"inputs; the compiled model takes {len(placeholders)}"
        )
    arguments = []
    for index, (value, node) in enumerate(zip(values, placeholders)):
        expected = node.meta.get("val")
        if isinstance(expected, torch.Tensor):
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != expected.shape
                or value.dtype != expected.dtype
            ):
                # Regions are compiled for static shapes; a short final batch
                # from a DataLoader needs drop_last=True.
                raise RegionError(
                    f"KernelGenerator validation_inputs batch {number} input {index} "
                    f"({node.name}) is {_describe(value)}; the model was compiled "
                    f"for {_describe(expected)}"
                )
            value = value.to(expected.device)
        arguments.append(value)
    return arguments


def _record(
    gm: torch.fx.GraphModule,
    batches: Iterable[Any],
    nodes: list[torch.fx.Node],
    budget: int,
) -> tuple[dict[torch.fx.Node, list[Sample]], int]:
    """Return samples from up to MAX_SAMPLES batches that fit ``budget`` bytes.

    The first batch is kept even if it does not fit. Also returns the bytes kept.
    """
    recorder = _BoundaryRecorder(gm, nodes)
    captured: dict[torch.fx.Node, list[Sample]] = {node: [] for node in nodes}
    for number, batch in enumerate(itertools.islice(batches, MAX_SAMPLES), 1):
        arguments = _arguments(batch, gm, number)
        with torch.no_grad():
            recorder.run(*arguments, enable_io_processing=False)
        for node in nodes:
            captured[node].append(recorder.inputs[node])
        # Batches have the compiled static shapes, so the next one would add
        # as many bytes as this one.
        if recorder.nbytes + recorder.batch_nbytes > budget:
            break
    if not captured[nodes[0]]:
        raise RegionError("KernelGenerator validation_inputs produced no batches")
    return captured, recorder.nbytes


def capture_samples(
    gm: torch.fx.GraphModule, sources: dict[torch.fx.Node, Iterable[Any]]
) -> dict[torch.fx.Node, list[Sample]]:
    """Run ``gm`` on the first batches of each region's validation inputs.

    ``sources`` maps a region's HOP node to its generator's ``validation_inputs``;
    regions that share an iterable share its model runs. All regions together
    keep at most MAX_SAMPLE_BYTES, apart from each iterable's first batch.
    """
    groups: dict[int, tuple[Iterable[Any], list[torch.fx.Node]]] = {}
    for node, batches in sources.items():
        groups.setdefault(id(batches), (batches, []))[1].append(node)
    captured: dict[torch.fx.Node, list[Sample]] = {}
    kept = 0
    for batches, nodes in groups.values():
        samples, nbytes = _record(gm, batches, nodes, MAX_SAMPLE_BYTES - kept)
        captured.update(samples)
        kept += nbytes
    if kept > MAX_SAMPLE_BYTES:
        _LOGGER.warning(
            "KernelGenerator validation_inputs keep %.3g MiB of region inputs, "
            "above the %.3g MiB limit, because a first batch is always kept",
            kept / 2**20,
            MAX_SAMPLE_BYTES / 2**20,
        )
    # These forward passes leave the model's activations cached in this
    # process; return them to the GPU for the evaluation workers. This does
    # nothing when CUDA was never initialized.
    torch.cuda.empty_cache()
    return captured


def check_samples(samples: Sequence[Sequence[Any]], specs: Sequence[Any]) -> None:
    """Raise ValueError unless every sample matches the region's input contract."""
    for number, sample in enumerate(samples, 1):
        if not isinstance(sample, (tuple, list)) or len(sample) != len(specs):
            raise ValueError(
                f"Validation sample {number} must hold {len(specs)} input tensors"
            )
        for index, (value, spec) in enumerate(zip(sample, specs)):
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != tuple(spec.shape)
                or value.dtype != spec.dtype
            ):
                raise ValueError(
                    f"Validation sample {number} input {index} is {_describe(value)}; "
                    f"the captured region contract is {_contract(spec.dtype, spec.shape)}"
                )


def samples_digest(samples: Sequence[Sample]) -> str:
    """Hash sample contents, so that equal digests mean equal data."""
    hasher = hashlib.sha256()
    for sample in samples:
        hasher.update(f"sample {len(sample)}".encode())
        for value in sample:
            hasher.update(f"{_describe(value)};".encode())
            data = value.detach().cpu().reshape(-1).view(torch.uint8)
            for start in range(0, data.numel(), _HASH_CHUNK):
                hasher.update(data[start : start + _HASH_CHUNK].numpy().tobytes())
    return hasher.hexdigest()


def save_samples(samples: Sequence[Sample], path: Path) -> None:
    torch.save([list(sample) for sample in samples], path)


def load_samples(path: str | Path) -> list[Sample]:
    """Load saved samples to the CPU without unpickling arbitrary objects."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, list) or any(
        not isinstance(sample, (list, tuple))
        or not all(isinstance(value, torch.Tensor) for value in sample)
        for sample in data
    ):
        raise ValueError(f"{path} does not hold a list of validation samples")
    return [tuple(sample) for sample in data]
