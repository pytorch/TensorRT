# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Kernel-language adapters, independently resolvable in evaluation workers.

An extension is an explicitly trusted ``module:attribute`` object (or a class
with a no-argument constructor). Its module must be importable in the worker.
Extensions return the same KernelSpec/partial registration forms as built-ins;
candidate data crossing the process boundary must remain JSON serializable.
"""

from __future__ import annotations

import copy
import heapq
import importlib
import importlib.metadata
import itertools
import keyword
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Protocol, cast

if TYPE_CHECKING:
    from torch_tensorrt.kernels import KernelSpec


class GenerationBackend(Protocol):
    """A kernel language: its proposal schema, checks, and plugin registration.

    ``make_registration`` receives the captured tensor boundaries as objects
    exposing ``shape`` (a tuple of ints), ``dtype`` (a ``torch.dtype``), and
    ``device`` (a ``torch.device``), in the controller and the worker alike.

    A backend may also define ``launch(candidate, inputs, outputs, directory)``,
    which launches a validated candidate once on the current torch CUDA stream
    without synchronizing. It reads ``inputs`` and writes ``outputs``: contiguous
    CUDA tensors with the captured shapes and dtypes, possibly interior views of
    larger buffers. The evaluator uses it to detect out-of-bounds writes and,
    under the default ``SearchPolicy(screening="launch")``, as the search's
    executor: it checks every case through ``launch`` and ranks candidates by
    timing it in CUDA graphs. A launch must therefore run the kernel its plugin
    runs, with the same source and launch settings, and must be safe to capture
    in a CUDA graph after warm-up launches. The evaluator registers the plugin
    before a screen, which applies the plugin path's own checks, so
    ``make_registration`` should reject a candidate that the direct launch can
    run but the plugin cannot, as the Triton backend does for more shared
    memory than a plugin launch allows. A backend whose launch cannot meet
    these needs ``screening="engine"``.
    """

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def proposal_schema(self) -> dict[str, Any]: ...

    @property
    def instruction_resources(self) -> tuple[str, ...]: ...

    def target_metadata(self, device: Any) -> dict[str, Any]: ...

    def validate_candidate(
        self, proposal: dict[str, Any], num_inputs: int, num_outputs: int
    ) -> dict[str, Any]: ...

    def make_registration(
        self,
        candidate: dict[str, Any],
        input_specs: list[Any],
        output_specs: list[Any],
        directory: Path,
    ) -> KernelSpec | partial[None]: ...


def _proposal_schema(**properties: Any) -> dict[str, Any]:
    # Structured-output providers require every property and no others.
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def _dimensions(maximum: int) -> dict[str, Any]:
    """One to three launch dimensions; validators check per-axis limits."""
    return {
        "type": "array",
        "items": {"type": "integer", "minimum": 1, "maximum": maximum},
        "minItems": 1,
        "maxItems": 3,
    }


def _target_metadata(
    device: Any, versions: dict[str, Any], packages: tuple[str, ...]
) -> dict[str, Any]:
    """Describe the GPU and toolchain, adding a backend's own versions."""
    import tensorrt
    import torch

    import torch_tensorrt
    from torch_tensorrt.region import RegionError

    properties = torch.cuda.get_device_properties(device)
    versions = {
        "torch": torch.__version__,
        "torch_tensorrt": torch_tensorrt.__version__,
        "tensorrt": tensorrt.__version__,
        "cuda": torch.version.cuda,
        **versions,
    }
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            raise RegionError(f"Kernel generation backend requires {package}") from None
    return {
        "name": properties.name,
        "capability": [properties.major, properties.minor],
        "multiprocessors": properties.multi_processor_count,
        "versions": versions,
    }


def get_backend(name: str) -> GenerationBackend:
    """Resolve a built-in or an explicit extension, never a parent-only registry."""
    from ._candidate import CUTILE, TRITON
    from ._cuda import CUDA_CPP

    builtins: tuple[GenerationBackend, ...] = (TRITON, CUTILE, CUDA_CPP)
    for builtin in builtins:
        if name == builtin.name:
            return builtin
    from torch_tensorrt.region._generation import split_import_path

    try:
        module_name, attribute = split_import_path(name)
    except ValueError:
        raise ValueError(
            "Generated kernel backend must be 'triton', 'cutile', 'cuda_cpp', "
            "or an importable 'module:attribute' adapter"
        ) from None
    backend = getattr(importlib.import_module(module_name), attribute)
    if isinstance(backend, type):
        backend = backend()
    if (
        not isinstance(getattr(backend, "name", None), str)
        or not backend.name
        or not isinstance(getattr(backend, "version", None), str)
        or not backend.version
        or not isinstance(getattr(backend, "proposal_schema", None), dict)
        or not isinstance(getattr(backend, "instruction_resources", None), tuple)
        or not all(isinstance(path, str) for path in backend.instruction_resources)
        or not all(
            callable(getattr(backend, method, None))
            for method in ("target_metadata", "validate_candidate", "make_registration")
        )
    ):
        raise ValueError(f"Backend {name!r} does not implement GenerationBackend")
    return cast(GenerationBackend, backend)


def _matches_tuning_schema(value: Any, schema: dict[str, Any] | bool) -> bool:
    """Reject incompatible JSON shapes; full schema validation belongs to the backend.

    Unknown constraints are deliberately ignored. In particular, ``oneOf`` can
    distinguish overlapping shapes using constraints such as ``pattern`` or
    ``$ref``, so this preliminary check cannot require an exclusive match.
    """
    if isinstance(schema, bool):
        return schema
    for operator in ("anyOf", "oneOf"):
        if operator in schema and not any(
            _matches_tuning_schema(value, option) for option in schema[operator]
        ):
            return False
    kinds = schema.get("type", [])
    kinds = [kinds] if isinstance(kinds, str) else kinds
    matches = {
        "null": value is None,
        "boolean": type(value) is bool,
        # Validators require exact ints, so neither 4.0 nor True is an integer.
        "integer": type(value) is int,
        "number": type(value) in (int, float),
        "string": isinstance(value, str),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }
    if kinds and not any(matches.get(kind, False) for kind in kinds):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if type(value) in (int, float):
        if "minimum" in schema and value < schema["minimum"]:
            return False
        if "maximum" in schema and value > schema["maximum"]:
            return False
    if isinstance(value, list):
        if (
            not schema.get("minItems", 0)
            <= len(value)
            <= schema.get("maxItems", len(value))
        ):
            return False
        if isinstance(schema.get("items"), dict) and not all(
            _matches_tuning_schema(item, schema["items"]) for item in value
        ):
            return False
    return True


def validate_tuning_space(
    backend_name: str, tuning: dict[str, tuple[Any, ...]]
) -> dict[str, list[Any]]:
    """Snapshot backend-declared tuning choices without accepting source overrides.

    ``constants.NAME`` changes an existing named compile-time constant. Arbitrary
    nested paths are not supported. Source/ABI and cross-field launch checks are
    performed by ``validate_candidate`` on each materialized variant.
    """
    from torch_tensorrt.region._generation import _snapshot_tuning

    choices = _snapshot_tuning(tuning)
    properties = get_backend(backend_name).proposal_schema.get("properties", {})
    result = {}
    for name in sorted(choices):
        if name in ("source", "kernel_name"):
            raise ValueError(f"Tuning cannot override {name!r}")
        if name.startswith("constants."):
            constant = name.removeprefix("constants.")
            if (
                not constant.isidentifier()
                or keyword.iskeyword(constant)
                or "__" in constant
            ):
                raise ValueError(f"Invalid tuning constant name: {constant!r}")
            if "constants" in choices:
                raise ValueError("Tune either constants or constants.NAME, not both")
            schema = properties.get("constants", {})
            for key in ("items", "properties", "value"):
                schema = schema.get(key) if isinstance(schema, dict) else None
        else:
            schema = properties.get(name)
        if not isinstance(schema, (dict, bool)):
            raise ValueError(f"Backend {backend_name!r} has no tunable field {name!r}")
        if not all(_matches_tuning_schema(value, schema) for value in choices[name]):
            raise ValueError(
                f"Tuning choices for {name!r} do not match the backend schema"
            )
        result[name] = list(choices[name])
    return result


def _knob(candidate: dict[str, Any], name: str) -> tuple[dict[str, Any], str]:
    """Return the JSON object and key that hold a tuning knob's value."""
    if not name.startswith("constants."):
        return candidate, name
    constant = name.removeprefix("constants.")
    constants = candidate.get("constants")
    matches = (
        [
            item
            for item in constants
            if isinstance(item, dict) and item.get("name") == constant
        ]
        if isinstance(constants, list)
        else []
    )
    if len(matches) != 1:
        raise ValueError(
            f"Candidate must declare exactly one constant {constant!r} to tune"
        )
    return matches[0], "value"


def _nearest_first(choices: list[Any], current: Any) -> list[Any]:
    """Order choices outward from the authored value; earlier choices win ties."""
    origin = next(
        (
            index
            for index, choice in enumerate(choices)
            if type(choice) is type(current) and choice == current
        ),
        None,
    )
    if origin is None:
        return choices
    order = sorted(range(len(choices)), key=lambda index: (abs(index - origin), index))
    return [choices[index] for index in order]


def _graded(sizes: list[int]) -> Iterator[tuple[int, ...]]:
    """Yield every rank vector by increasing sum, then lexicographically."""
    start = (0,) * len(sizes)
    frontier, seen = [(0, start)], {start}
    while frontier:
        total, ranks = heapq.heappop(frontier)
        yield ranks
        for knob, size in enumerate(sizes):
            if ranks[knob] + 1 < size:
                successor = (*ranks[:knob], ranks[knob] + 1, *ranks[knob + 1 :])
                if successor not in seen:
                    seen.add(successor)
                    heapq.heappush(frontier, (total + 1, successor))


def candidate_variants(
    candidate: dict[str, Any],
    backend_name: str,
    tuning: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    """Enumerate at most ``limit`` tuned copies, nearest to the proposal first.

    Each knob ranks its choices outward from the proposal's own value when that
    value is a choice, and otherwise keeps their declared order. Variants follow
    increasing total rank, ties broken by the ranks in knob-name order, so the
    authored configuration comes first when allowed and every knob changes
    within the first ``len(tuning) + 1`` variants. Every variant applies every
    knob; no Python expressions or launch callbacks are accepted as values.
    """
    if type(limit) is not int or not 0 <= limit <= 4096:
        raise ValueError("Variant limit must be an integer in [0, 4096]")
    if not isinstance(candidate, dict):
        raise ValueError("A candidate must be a JSON object")
    choices = validate_tuning_space(backend_name, tuning)
    ranked = {}
    for name, values in choices.items():
        holder, key = _knob(candidate, name)
        ranked[name] = _nearest_first(values, holder.get(key, object()))
    variants = []
    for ranks in itertools.islice(_graded([len(v) for v in ranked.values()]), limit):
        variant = copy.deepcopy(candidate)
        for (name, values), rank in zip(ranked.items(), ranks):
            holder, key = _knob(variant, name)
            holder[key] = copy.deepcopy(values[rank])
        variants.append(variant)
    return variants
