# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Capture region inputs from real model batches and save them without code.

``KernelGenerator(validation_inputs=...)`` supplies model-level batches; without
them, the example tensors given to compilation are one batch. Before any region
is replaced, compilation runs the parent graph on them and keeps the tensors
that reach each generated region; every candidate is then checked on these
samples as well as on synthetic inputs drawn within their range. A run saves
them with ``torch.save`` as a plain list of tensor lists, which loads with
``torch.load(weights_only=True)`` and so cannot run code from a shared run.
"""

from __future__ import annotations

import hashlib
import itertools
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Sequence, cast

import torch

from torch_tensorrt.region import RegionError

_LOGGER = logging.getLogger(__name__)

FILENAME = "samples.pt"
#: Batches used per region; each costs a model run and a check per evaluation.
MAX_SAMPLES = 8
#: Region input bytes kept from all validation batches of one compilation:
#: they stay in memory while its regions are generated, and each run saves its
#: region's share. Fewer validation_inputs batches are used past this, though
#: the first is always kept; a region whose inputs from the example inputs do
#: not fit keeps none of them.
MAX_SAMPLE_BYTES = 1 << 30
#: The factors by which the evaluator's magnitude cases ("scaled x0.001" and
#: "scaled x100") multiply every input that is not integer-valued; the sign
#: of each value is kept.
MAGNITUDE_SCALES = (1e-3, 1e2)
# Bytes hashed at once, so a digest never copies a whole large tensor.
_HASH_CHUNK = 1 << 26

Sample = tuple[torch.Tensor, ...]


def magnitude_limit(dtype: torch.dtype) -> float:
    """The largest magnitude a scaled or offset validation input takes.

    The product of two such values stays finite: these cases test magnitude
    handling, not trivial overflow.
    """
    return math.sqrt(torch.finfo(dtype).max) / 2


def integer_valued(value: torch.Tensor) -> bool:
    """Whether every finite element is an integer, as in positions and masks."""
    finite = value[value.isfinite()]
    return bool((finite == finite.round()).all())


Interval = tuple[float, float]


class _Values(NamedTuple):
    """What the coverage rule needs to know about one input tensor."""

    # The smallest and largest finite values; None when there are none.
    low: float | None
    high: float | None
    # Every finite value is an integer.
    integer: bool
    # Some value is NaN or infinite.
    nonfinite: bool
    # A digest of the tensor when it holds at most two distinct finite
    # values, as a mask does; None otherwise.
    exact: str | None


def _values(value: torch.Tensor) -> _Values:
    mask = value.isfinite()
    nonfinite = not bool(mask.all())
    finite = value[mask] if nonfinite else value
    if not finite.numel():
        return _Values(None, None, True, nonfinite, _tensor_digest(value))
    low, high = finite.min(), finite.max()
    # Each finite value is the smallest or the largest.
    selector = bool(((finite == low) | (finite == high)).all())
    return _Values(
        float(low),
        float(high),
        bool((finite == finite.round()).all()),
        nonfinite,
        _tensor_digest(value) if selector else None,
    )


class Coverage(NamedTuple):
    """The values of one region input that correctness checks have covered.

    It holds the checked samples' own ranges, which the samples and the random
    cases derived from them check with the in-range bounds. The magnitude
    cases do not widen it: they scale every input at once, and the values
    they take beyond the samples' range get only the lenient robustness bounds.
    """

    # Sorted, disjoint closed intervals: the union of the finite ranges of the
    # checked samples whose values were not all integers ...
    continuous: tuple[Interval, ...]
    # ... and of those whose values were all integers, which random cases
    # vary over integers only, so these intervals cover integer values only.
    integer: tuple[Interval, ...]
    # Digests of the checked tensors that hold at most two distinct finite
    # values, such as 0/1 masks; such an input is covered only by one of them.
    exact: frozenset[str]
    # Whether some checked value was NaN or infinite.
    nonfinite: bool


def _union(intervals: Iterable[Interval]) -> tuple[Interval, ...]:
    """Merge overlapping closed intervals, leaving the gaps between them."""
    merged: list[Interval] = []
    for low, high in sorted(intervals):
        if merged and low <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return tuple(merged)


def coverage(samples: Sequence[Sample]) -> list[Coverage] | None:
    """Each input's coverage after a correctness check on ``samples``.

    Every sample contributes its own finite range of each input: a range is
    covered, and the gap between the ranges of two samples is not. None
    stands for a check without samples.
    """
    if not samples:
        return None
    covered = []
    for values in zip(*samples):
        continuous: list[Interval] = []
        integer: list[Interval] = []
        exact: set[str] = set()
        nonfinite = False
        for value in map(_values, values):
            nonfinite |= value.nonfinite
            if value.exact is not None:
                exact.add(value.exact)
            if value.low is not None and value.high is not None:
                ranges = integer if value.integer else continuous
                ranges.append((value.low, value.high))
        covered.append(
            Coverage(_union(continuous), _union(integer), frozenset(exact), nonfinite)
        )
    return covered


def merge_coverage(
    first: Sequence[Coverage] | None, second: Sequence[Coverage] | None
) -> list[Coverage] | None:
    """The coverage of both checks; None stands for a check without samples."""
    if first is None:
        return None if second is None else list(second)
    if second is None:
        return list(first)
    return [
        Coverage(
            _union(one.continuous + other.continuous),
            _union(one.integer + other.integer),
            one.exact | other.exact,
            one.nonfinite or other.nonfinite,
        )
        for one, other in zip(first, second)
    ]


def covers(covered: Sequence[Coverage] | None, samples: Sequence[Sample]) -> bool:
    """Whether checks with ``covered`` coverage also covered every input of ``samples``.

    An input is covered when its finite values lie within one interval of
    the coverage: an interval of integer-valued samples covers integer values
    only. An input with at most two distinct finite values, such as a 0/1 or
    an additive mask, is covered only by a checked tensor with exactly its
    values: a range says nothing about which elements a mask selects, and
    fully masked rows, windows and padding are where kernels go wrong. NaN
    and infinite values need a checked sample of the input that had some.
    """
    if covered is None or not samples:
        return False
    for sample in samples:
        if len(sample) != len(covered):
            return False
        for tested, value in zip(covered, map(_values, sample)):
            if value.nonfinite and not tested.nonfinite:
                return False
            if value.exact is not None or value.low is None or value.high is None:
                if value.exact not in tested.exact:
                    return False
                continue
            intervals = tested.continuous + (tested.integer if value.integer else ())
            if not any(
                low <= value.low and value.high <= high for low, high in intervals
            ):
                return False
    return True


class _BoundaryRecorder(torch.fx.Interpreter):  # type: ignore[misc,unused-ignore]
    """Run a parent graph, keeping the tensors that enter the selected regions.

    With a ``limit``, a region whose inputs would bring the bytes kept past it
    keeps none of them, so they are never copied; ``oversized`` maps it to the
    bytes they would have added and the bytes kept before it.
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        regions: Iterable[Any],
        limit: int | None = None,
    ) -> None:
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
        self.limit = limit
        self.oversized: dict[torch.fx.Node, tuple[int, int]] = {}

    def run(self, *args: Any, **kwargs: Any) -> Any:
        self.batch_nbytes = 0
        return super().run(*args, **kwargs)

    def run_node(self, node: torch.fx.Node) -> Any:
        if node in self.regions:
            # Region live-ins are graph nodes holding tensors.
            operands = cast(Sequence[torch.fx.Node], node.args[1])
            added = 0 if self.limit is None else self.added(operands)
            if self.limit is not None and self.nbytes + added > self.limit:
                self.oversized[node] = (added, self.nbytes)
            else:
                self.inputs[node] = tuple(self.keep(operand) for operand in operands)
        return super().run_node(node)

    def added(self, operands: Sequence[torch.fx.Node]) -> int:
        """The bytes that keeping ``operands`` would add; see keep."""
        # A constant is copied once, other operands at every use.
        copied = [operand for operand in operands if operand.op != "get_attr"]
        copied += [
            operand
            for operand in dict.fromkeys(operands)
            if operand.op == "get_attr" and operand not in self.constants
        ]
        return sum(
            self.env[operand].numel() * self.env[operand].element_size()
            for operand in copied
        )

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


def _nbytes(samples: Mapping[Any, Sequence[Sample]]) -> int:
    """Bytes of the distinct tensors in ``samples``; shared constants count once."""
    tensors = {
        id(value): value
        for group in samples.values()
        for sample in group
        for value in sample
    }
    return sum(value.nbytes for value in tensors.values())


def capture_example_samples(
    gm: torch.fx.GraphModule,
    regions: Mapping[torch.fx.Node, str],
    batch: Any,
    kept: Mapping[torch.fx.Node, Sequence[Sample]],
) -> dict[torch.fx.Node, list[Sample]]:
    """Run ``gm`` once on the compilation's example inputs, as one sample.

    ``regions`` maps the HOP nodes of the regions to capture to their ids.
    Unlike a first validation_inputs batch, which a user passes on purpose and
    which is always kept, these samples stay within MAX_SAMPLE_BYTES, together
    with the samples ``kept`` from validation_inputs: a region whose inputs do
    not fit what is left keeps no sample, with a warning, and is validated as
    a region without samples is. The caller empties the CUDA cache.
    """
    already = _nbytes(kept)
    recorder = _BoundaryRecorder(gm, regions, max(0, MAX_SAMPLE_BYTES - already))
    arguments = _arguments(batch, gm, 1)
    with torch.no_grad():
        recorder.run(*arguments, enable_io_processing=False)
    for node, (added, before) in recorder.oversized.items():
        _LOGGER.warning(
            "Region %r keeps no validation sample from the compilation's example "
            "inputs: its inputs would add %.3g MiB to the %.3g MiB kept for other "
            "regions, past the %.3g MiB limit on validation samples per "
            "compilation. A kernel searched for it is validated on synthetic "
            "inputs only, and an identical region's kernel is reused for it "
            "unchecked; pass KernelGenerator validation_inputs to validate it on "
            "real inputs",
            regions[node],
            added / 2**20,
            (already + before) / 2**20,
            MAX_SAMPLE_BYTES / 2**20,
        )
    return {node: [sample] for node, sample in recorder.inputs.items()}


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


def _hash_tensor(hasher: Any, value: torch.Tensor) -> None:
    hasher.update(f"{_describe(value)};".encode())
    data = value.detach().cpu().reshape(-1).view(torch.uint8)
    for start in range(0, data.numel(), _HASH_CHUNK):
        hasher.update(data[start : start + _HASH_CHUNK].numpy().tobytes())


def _tensor_digest(value: torch.Tensor) -> str:
    hasher = hashlib.sha256()
    _hash_tensor(hasher, value)
    return hasher.hexdigest()


def samples_digest(samples: Sequence[Sample]) -> str:
    """Hash sample contents, so that equal digests mean equal data."""
    hasher = hashlib.sha256()
    for sample in samples:
        hasher.update(f"sample {len(sample)}".encode())
        for value in sample:
            _hash_tensor(hasher, value)
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
