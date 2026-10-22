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
import importlib
import importlib.metadata
import itertools
import keyword
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from torch_tensorrt.kernels import KernelSpec


class GenerationBackend(Protocol):
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


def _schema(backend: str) -> dict[str, Any]:
    dimensions = {
        "type": "array",
        "items": {"type": "integer", "minimum": 1},
        "minItems": 1,
        "maxItems": 3,
    }
    properties: dict[str, Any] = {
        "source": {"type": "string"},
        "kernel_name": {"type": "string"},
        "grid": dimensions,
    }
    if backend == "cuda_cpp":
        properties.update(
            block=copy.deepcopy(dimensions),
            shared_mem_bytes={"type": "integer", "minimum": 0, "maximum": 49152},
        )
    else:
        properties["constants"] = {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": ["boolean", "number"]},
                },
                "required": ["name", "value"],
            },
        }
        if backend == "triton":
            properties.update(
                num_warps={"type": ["integer", "null"]},
                num_stages={"type": ["integer", "null"]},
            )
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


@dataclass(frozen=True)
class _BuiltinBackend:
    name: str
    version: str = "1"

    @property
    def proposal_schema(self) -> dict[str, Any]:
        return _schema(self.name)

    @property
    def instruction_resources(self) -> tuple[str, ...]:
        return (f"references/{self.name}.md",)

    def target_metadata(self, device: Any) -> dict[str, Any]:
        import tensorrt
        import torch

        import torch_tensorrt
        from torch_tensorrt.region_annotations import RegionError

        properties = torch.cuda.get_device_properties(device)
        versions: dict[str, Any] = {
            "torch": torch.__version__,
            "torch_tensorrt": torch_tensorrt.__version__,
            "tensorrt": tensorrt.__version__,
            "cuda": torch.version.cuda,
        }
        packages: tuple[str, ...]
        if self.name == "cutile":
            from torch_tensorrt.kernels._cutile import (
                _cutile_driver_version,
                validate_cutile_toolchain,
            )

            versions["driver"] = list(_cutile_driver_version())
            versions["cutile_compiler"] = list(
                validate_cutile_toolchain("agent_generation")
            )
            packages = ("cuda-tile",)
        elif self.name == "triton":
            from torch_tensorrt.kernels._triton import _driver_max_ptx_version

            versions["driver_ptx"] = _driver_max_ptx_version()
            packages = ("triton",)
        else:
            from cuda.bindings import driver, nvrtc

            from torch_tensorrt.kernels._nvrtc import _cuda_core_imports

            _cuda_core_imports()
            status, driver_version = driver.cuDriverGetVersion()
            rtc_status, major, minor = nvrtc.nvrtcVersion()
            if int(status) or int(rtc_status):
                raise RegionError("CUDA C++ generation cannot query the CUDA toolchain")
            versions["driver"] = driver_version
            versions["nvrtc"] = [major, minor]
            packages = ("cuda-core", "cuda-bindings")
        for package in packages:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                raise RegionError(
                    f"Kernel generation backend requires {package}"
                ) from None
        return {
            "name": properties.name,
            "capability": [properties.major, properties.minor],
            "multiprocessors": properties.multi_processor_count,
            "versions": versions,
            "backend_version": self.version,
        }

    def validate_candidate(
        self, proposal: dict[str, Any], num_inputs: int, num_outputs: int
    ) -> dict[str, Any]:
        normalized: dict[str, Any]
        if self.name == "cuda_cpp":
            from ._cuda import validate_cuda_candidate

            normalized = validate_cuda_candidate(proposal, num_inputs, num_outputs)
        else:
            from ._candidate import _validate_dsl_candidate

            normalized = _validate_dsl_candidate(
                proposal, self.name, num_inputs, num_outputs
            )
        return normalized

    def make_registration(
        self,
        candidate: dict[str, Any],
        input_specs: list[Any],
        output_specs: list[Any],
        directory: Path,
    ) -> KernelSpec | partial[None]:
        if self.name == "cuda_cpp":
            from ._cuda import make_cuda_registration

            return make_cuda_registration(
                candidate, input_specs, output_specs, directory
            )
        from ._candidate import _make_dsl_registration

        return _make_dsl_registration(
            candidate, self.name, input_specs, output_specs, directory
        )


def get_backend(name: str) -> GenerationBackend:
    """Resolve a built-in or an explicit extension, never a parent-only registry."""
    if name in ("triton", "cutile", "cuda_cpp"):
        return _BuiltinBackend(name)
    if not isinstance(name, str) or name.count(":") != 1:
        raise ValueError(
            "Generated kernel backend must be 'triton', 'cutile', 'cuda_cpp', "
            "or an importable 'module:attribute' adapter"
        )
    module_name, attribute = name.split(":")
    if not all(part.isidentifier() for part in module_name.split(".")) or not (
        attribute.isidentifier()
    ):
        raise ValueError("Backend extension must use an importable 'module:attribute'")
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
        "integer": type(value) is int or (type(value) is float and value.is_integer()),
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
    from torch_tensorrt.region_annotations._generation import _snapshot_tuning

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


def candidate_variants(
    candidate: dict[str, Any],
    backend_name: str,
    tuning: dict[str, Any],
    limit: int,
) -> list[dict[str, Any]]:
    """Enumerate a deterministic bounded product, always applying every knob.

    With a tuning space, every returned proposal has all configured overrides.
    No Python expressions or launch callbacks are accepted as tuning values.
    """
    if type(limit) is not int or not 0 <= limit <= 4096:
        raise ValueError("Variant limit must be an integer in [0, 4096]")
    if not isinstance(candidate, dict):
        raise ValueError("A candidate must be a JSON object")
    choices = validate_tuning_space(backend_name, tuning)
    names = list(choices)
    variants = []
    for values in itertools.islice(itertools.product(*choices.values()), limit):
        variant = copy.deepcopy(candidate)
        for name, value in zip(names, values):
            if name.startswith("constants."):
                constant = name.removeprefix("constants.")
                constants = variant.get("constants")
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
                matches[0]["value"] = copy.deepcopy(value)
            else:
                variant[name] = copy.deepcopy(value)
        variants.append(variant)
    return variants
