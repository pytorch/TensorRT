# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Knobs: the options a planner may change, and the configs made of them.

A knob is one option with a short list of allowed values, for example
``runtime.cudagraphs`` (``False`` or ``True``). A config records the knobs
set to something other than their default. Some values need other settings,
for example a fused kernel that only runs under CUDA graphs; a knob declares
such requirements, :func:`apply_changes` refuses a config that breaks one,
and :func:`with_requirements` adds what a planner's change needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from torch_tensorrt.autotune._code_edits import is_edit_digest, is_edit_key
from torch_tensorrt.autotune._settings import Config

# The five layers a knob belongs to: four compiler layers (graph partitioning,
# TensorRT engine, kernel generation, runtime execution) and the options a
# workload defines itself.
LAYERS = ("graph", "engine", "kernels", "runtime", "workload")
# The bottleneck hints the profiler can report (see _profile.bottleneck_hints).
# A knob names the hints it can relieve, which guides the rule-based planner.
BOTTLENECK_HINTS = (
    "launch_overhead",
    "host_syncs",
    "fallback",
    "dominant_kernel",
    "precision",
    "roofline_gap",
    "copy_overhead",
    "compute_gap",
)


@dataclass(frozen=True)
class Knob:
    """An option with the finite list of values a planner may choose from.

    ``name`` is ``"<layer>.<option>"`` with a layer from ``graph``, ``engine``,
    ``kernels``, ``runtime`` and ``workload``. ``values`` lists the allowed
    values (JSON scalars) in the order the rule-based planner tries them;
    ``default`` is the value a config that omits the knob uses.
    ``description`` tells a planner what the option does.
    ``bottleneck_hints`` names the profiler's bottleneck hints the option can
    relieve.

    ``requires`` names the settings of other knobs that this knob's values
    need (fused kernels that only run under CUDA graphs, say), in one of two
    forms:

    * ``{"runtime.cudagraphs": True}`` (other knob to value): every
      non-default value needs them;
    * ``{"fused": {"runtime.cudagraphs": True}, "graph": {...}}`` (one of
      this knob's non-default values to such a mapping): each listed value
      needs its own settings and an unlisted value needs none. A sequence of
      ``(value, requirements)`` pairs is accepted too, for values that are
      equal as dictionary keys (``1`` and ``True``).

    A config that breaks a requirement is refused before anything is built.
    The rule-based planner proposes an option together with what it
    requires, and with what those settings require in turn.
    :func:`knobs_by_name` refuses a requirement on a knob or value no knob
    offers, and a value whose requirements contradict each other.

    ``prepare`` lists non-default values whose build must be done once before
    the search, such as a value that generates region kernels: the session
    builds the baseline config with each such value (and what it requires) in
    a worker of its own, records the region kernels the build generated or
    loaded, and pins later runs that use the value to exactly those kernels.
    """

    name: str
    values: tuple[Any, ...]
    default: Any
    description: str
    bottleneck_hints: tuple[str, ...] = ()
    requires: Any = ()
    prepare: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        layer, sep, option = self.name.partition(".")
        if not sep or layer not in LAYERS or not option:
            raise ValueError(
                f"Knob name must be '<layer>.<option>' with a layer in {LAYERS}; "
                f"got {self.name!r}"
            )
        values = tuple(self.values)
        if not values or any(not _is_scalar(value) for value in values):
            raise ValueError(f"Knob {self.name} values must be JSON scalars")
        if len({_key(value) for value in values}) != len(values):
            raise ValueError(f"Knob {self.name} lists a value twice")
        if not is_member(self.default, values):
            raise ValueError(f"Knob {self.name} default must be one of its values")
        unknown = set(self.bottleneck_hints) - set(BOTTLENECK_HINTS)
        if unknown:
            raise ValueError(
                f"Knob {self.name} names unknown bottleneck hints {sorted(unknown)}"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "bottleneck_hints", tuple(self.bottleneck_hints))
        object.__setattr__(self, "requires", self._normalized_requires(values))
        object.__setattr__(self, "prepare", self._normalized_prepare(values))

    def _normalized_prepare(self, values: tuple[Any, ...]) -> tuple[Any, ...]:
        """``prepare`` as a tuple in the order of ``values``, each one checked."""
        raw = self.prepare
        if not isinstance(raw, (tuple, list)):
            raise ValueError(
                f"Knob {self.name} prepare must be a tuple of its values; got {raw!r}"
            )
        seen: set[tuple[str, Any]] = set()
        for value in raw:
            if not _is_scalar(value) or not is_member(value, values):
                raise ValueError(
                    f"Knob {self.name} prepare lists {value!r}, which is not one of "
                    f"its values {list(values)}"
                )
            if _key(value) == _key(self.default):
                raise ValueError(
                    f"Knob {self.name} prepare lists its default {value!r}: a "
                    "config at the default omits the knob, so no run would be "
                    "pinned to the prepared build"
                )
            if _key(value) in seen:
                raise ValueError(f"Knob {self.name} prepare lists {value!r} twice")
            seen.add(_key(value))
        order = [_key(value) for value in values]
        return tuple(sorted(raw, key=lambda value: order.index(_key(value))))

    def _normalized_requires(self, values: tuple[Any, ...]) -> tuple[Any, ...]:
        """``requires`` as sorted ``(knob, value)`` pairs, or as ``(value, pairs)``
        entries in the order of ``values`` when it differs by value."""
        raw = self.requires
        items = list(raw.items()) if isinstance(raw, Mapping) else list(raw)
        if any(not isinstance(item, (tuple, list)) or len(item) != 2 for item in items):
            raise ValueError(
                f"Knob {self.name} requires must map other knobs to values, or "
                "this knob's values to such mappings"
            )
        nested = [isinstance(item[1], (Mapping, tuple, list)) for item in items]
        if not any(nested):
            return self._requirement_pairs(dict(items))
        if not all(nested):
            raise ValueError(
                f"Knob {self.name} requires mixes settings for every value with "
                "settings for single values; use one form"
            )
        by_value: list[tuple[Any, tuple[tuple[str, Any], ...]]] = []
        seen: set[tuple[str, Any]] = set()
        for value, needs in items:
            if not is_member(value, values):
                raise ValueError(
                    f"Knob {self.name} requires lists {value!r}, which is not one "
                    f"of its values {list(values)}"
                )
            if _key(value) == _key(self.default):
                raise ValueError(
                    f"Knob {self.name} requires lists its default {value!r}: a "
                    "config at the default omits the knob, so nothing would "
                    "enforce it"
                )
            if _key(value) in seen:
                raise ValueError(f"Knob {self.name} requires lists {value!r} twice")
            seen.add(_key(value))
            settings = self._requirement_pairs(dict(needs))
            if settings:
                by_value.append((value, settings))
        order = [_key(value) for value in values]
        return tuple(sorted(by_value, key=lambda entry: order.index(_key(entry[0]))))

    def _requirement_pairs(
        self, requires: dict[Any, Any]
    ) -> tuple[tuple[str, Any], ...]:
        """Check a ``{knob: value}`` requirement mapping; return it as sorted pairs."""
        if self.name in requires:
            raise ValueError(f"Knob {self.name} cannot require itself")
        for other, value in requires.items():
            if not isinstance(other, str):
                raise ValueError(
                    f"Knob {self.name} requires must name other knobs; got {other!r}"
                )
            if not _is_scalar(value):
                raise ValueError(f"Knob {self.name} requires JSON scalars")
        return tuple(sorted(requires.items()))

    @property
    def layer(self) -> str:
        """The layer the knob belongs to: the part of its name before the dot."""
        return self.name.partition(".")[0]

    @property
    def requires_by_value(self) -> bool:
        """Whether ``requires`` lists settings per value rather than for every value."""
        return bool(self.requires) and isinstance(self.requires[0][1], tuple)

    def requirements(self, value: Any) -> dict[str, Any]:
        """The settings of other knobs that ``value`` of this knob needs."""
        if _key(value) == _key(self.default):
            return {}
        if not self.requires_by_value:
            return dict(self.requires)
        for listed, needs in self.requires:
            if _key(listed) == _key(value):
                return dict(needs)
        return {}

    def to_json(self) -> dict[str, Any]:
        """The knob as JSON; per-value ``requires`` become ``{"value", "requires"}`` entries,
        so values that are not strings survive the round trip.
        """
        return {
            "name": self.name,
            "layer": self.layer,
            "values": list(self.values),
            "default": self.default,
            "description": self.description,
            "bottleneck_hints": list(self.bottleneck_hints),
            "requires": (
                [
                    {"value": value, "requires": dict(needs)}
                    for value, needs in self.requires
                ]
                if self.requires_by_value
                else dict(self.requires)
            ),
            "prepare": list(self.prepare),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "Knob":
        """Rebuild a knob from :meth:`to_json` (extra keys, such as ``current``, are ignored)."""
        requires = data["requires"]
        if isinstance(requires, (list, tuple)):
            requires = [(entry["value"], entry["requires"]) for entry in requires]
        return cls(
            data["name"],
            tuple(data["values"]),
            data["default"],
            data["description"],
            tuple(data["bottleneck_hints"]),
            requires,
            tuple(data["prepare"]),
        )


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _key(value: Any) -> tuple[str, Any]:
    """``value`` with its type, for comparisons.

    In Python ``True == 1 == 1.0``, but a knob's values must stay apart:
    ``True`` is not the optimization level 1.
    """
    return type(value).__name__, value


def is_member(value: Any, values: Iterable[Any]) -> bool:
    """Whether ``value`` is one of ``values``, with types compared too."""
    return any(_key(value) == _key(allowed) for allowed in values)


# The knobs a workload honors unless it declares its own (ModuleWorkload honors
# all of them).
STANDARD_KNOBS: tuple[Knob, ...] = (
    Knob(
        "runtime.cudagraphs",
        (False, True),
        False,
        "Replay the compiled module from a CUDA graph, removing per-call launch "
        "and host overhead; needs stream-capturable engines and static shapes.",
        ("launch_overhead", "host_syncs"),
    ),
    Knob(
        "runtime.pre_allocated_outputs",
        (False, True),
        False,
        "Reuse output buffers between calls instead of allocating them each call.",
        ("launch_overhead",),
    ),
    Knob(
        "engine.precision",
        ("fp32", "fp16", "bf16"),
        "fp32",
        "Compute precision of TensorRT layers. Lower precision halves memory "
        "traffic and enables tensor cores, within the accuracy limits.",
        ("precision", "dominant_kernel"),
    ),
    Knob(
        "engine.optimization_level",
        (3, 5, 4),
        3,
        "TensorRT builder optimization level; higher levels search more tactics "
        "and take longer to build.",
        ("dominant_kernel",),
    ),
    Knob(
        "engine.tiling_optimization_level",
        ("none", "moderate", "fast", "full"),
        "none",
        "TensorRT tiling optimization: searches cross-kernel tiled schedules for "
        "on-chip cache reuse, at extra build time.",
        ("dominant_kernel",),
    ),
    Knob(
        "graph.min_block_size",
        (5, 1, 3, 10),
        5,
        "Minimum number of operators in a TensorRT engine. Lower values place "
        "more of the graph in TensorRT; higher values avoid many small engines.",
        ("fallback",),
    ),
)


def knobs_by_name(knobs: Sequence[Knob]) -> dict[str, Knob]:
    """Index ``knobs`` by name, rejecting duplicates and requirements no config can meet.

    Every requirement must name one of the knobs and one of its values, and
    no value's requirements, followed transitively, may ask for two values of
    one knob.
    """
    indexed: dict[str, Knob] = {}
    for knob in knobs:
        if not isinstance(knob, Knob):
            raise TypeError(f"Expected a Knob; got {type(knob).__name__}")
        if knob.name in indexed:
            raise ValueError(f"Knob {knob.name} is defined twice")
        indexed[knob.name] = knob
    for knob in indexed.values():
        for value in knob.values:
            for name, needed in knob.requirements(value).items():
                other = indexed.get(name)
                if other is None or not is_member(needed, other.values):
                    option = (
                        f"{knob.name}={value!r}"
                        if knob.requires_by_value
                        else knob.name
                    )
                    raise ValueError(
                        f"Knob {option} requires {name}={needed!r}, which no "
                        "knob offers"
                    )
    for knob in indexed.values():
        for value in knob.values:
            if knob.requirements(value):
                try:
                    with_requirements({}, {knob.name: value}, indexed)
                except ValueError as error:
                    raise ValueError(
                        f"Knob {knob.name}={value!r} can never be used: {error}"
                    ) from None
    return indexed


def unmet_requirements(
    config: Mapping[str, Any],
    name: str,
    knobs: Mapping[str, Knob],
    value: Any,
) -> dict[str, Any]:
    """What ``config`` lacks of the settings knob ``name`` needs at ``value``."""
    needs = knobs[name].requirements(value)
    return {
        other: needed
        for other, needed in needs.items()
        if not is_member(config.get(other, knobs[other].default), (needed,))
    }


def with_requirements(
    config: Mapping[str, Any], changes: Mapping[str, Any], knobs: Mapping[str, Knob]
) -> dict[str, Any]:
    """``changes`` plus every setting they need that ``config`` lacks.

    A requirement of a required value is followed as well, and so on.
    Raises ``ValueError`` when two requirements, or a requirement and
    ``changes``, ask for different values of one knob. Unknown knobs are
    left to :func:`apply_changes` to report.
    """
    bundled = dict(changes)
    # Every value the experiment fixes: its changes and what they require.
    fixed = {name: value for name, value in changes.items() if not is_edit_key(name)}
    pending = list(fixed.items())
    while pending:
        name, value = pending.pop(0)
        knob = knobs.get(name)
        if knob is None:
            continue
        for other, needed in knob.requirements(value).items():
            if other in fixed:
                if not is_member(fixed[other], (needed,)):
                    raise ValueError(
                        f"{name}={value!r} requires {other}={needed!r}, which "
                        f"conflicts with {other}={fixed[other]!r}"
                    )
                continue
            fixed[other] = needed
            pending.append((other, needed))
            known = knobs.get(other)
            if known is None or not is_member(
                config.get(other, known.default), (needed,)
            ):
                bundled[other] = needed
    return bundled


def chosen_changes(
    changes: Mapping[str, Any], knobs: Mapping[str, Knob]
) -> dict[str, Any]:
    """The changes a planner chose, without the settings other changes require.

    ``{"kernels.k": "on", "runtime.cudagraphs": True}`` where ``kernels.k=on``
    requires CUDA graphs gives ``{"kernels.k": "on"}``: CUDA graphs alone
    remain untried. When every change is required by another, all count.
    """
    bundled = set()
    for name, value in changes.items():
        knob = knobs.get(name)
        if knob is None or is_edit_key(name):
            continue
        for other, needed in knob.requirements(value).items():
            if other in changes and is_member(changes[other], (needed,)):
                bundled.add(other)
    chosen = {name: value for name, value in changes.items() if name not in bundled}
    return chosen or dict(changes)


def apply_changes(
    config: Mapping[str, Any], changes: Mapping[str, Any], knobs: Mapping[str, Knob]
) -> Config:
    """Return ``config`` with ``changes`` applied, checked and normalized.

    Raises ``ValueError`` naming the first unknown knob, disallowed value or
    unmet requirement of a value the result sets. Options set to their
    default are dropped, so equal configurations compare equal.
    """
    merged = dict(config)
    for name, value in changes.items():
        if is_edit_key(name):
            if value and not is_edit_digest(value):
                raise ValueError(
                    f"{name} must be the SHA-256 digest of an edited file; "
                    f"got {value!r}"
                )
            merged[name] = value
            continue
        knob = knobs.get(name)
        if knob is None:
            raise ValueError(f"Unknown knob {name!r}; choose one of {sorted(knobs)}")
        if not is_member(value, knob.values):
            raise ValueError(
                f"Knob {name} does not allow {value!r}; allowed: {list(knob.values)}"
            )
        merged[name] = value
    for name, value in list(merged.items()):
        if is_edit_key(name):
            # An edit digest selects an edited file; empty means the original.
            if not value:
                del merged[name]
            elif not is_edit_digest(value):
                raise ValueError(
                    f"Config sets {name}={value!r}, not the SHA-256 digest of an "
                    "edited file"
                )
            continue
        knob = knobs.get(name)
        if knob is None:
            raise ValueError(f"Config sets unknown knob {name!r}")
        if not is_member(value, knob.values):
            raise ValueError(
                f"Config sets {name}={value!r}; allowed: {list(knob.values)}"
            )
        if _key(value) == _key(knob.default):
            del merged[name]
    for name in merged:
        if is_edit_key(name):
            continue
        unmet = unmet_requirements(merged, name, knobs, merged[name])
        if unmet:
            needs = ", ".join(f"{other}={value!r}" for other, value in unmet.items())
            raise ValueError(f"{name}={merged[name]!r} requires {needs}")
    return dict(sorted(merged.items()))


def resolve(config: Mapping[str, Any], knobs: Mapping[str, Knob]) -> dict[str, Any]:
    """Every knob's value under ``config`` (defaults included) and its edit digests.

    These are the values a worker builds a candidate from.
    """
    normalized = apply_changes(config, {}, knobs)
    values = {name: normalized.get(name, knob.default) for name, knob in knobs.items()}
    values.update(
        {name: value for name, value in normalized.items() if is_edit_key(name)}
    )
    return values


def difference(
    before: Mapping[str, Any], after: Mapping[str, Any], knobs: Mapping[str, Knob]
) -> dict[str, Any]:
    """The options (and edited files) whose value ``after`` changes, with their new values."""
    old, new = resolve(before, knobs), resolve(after, knobs)
    changed = {name: new[name] for name in knobs if _key(old[name]) != _key(new[name])}
    for name in sorted({*old, *new} - set(knobs)):
        if old.get(name) != new.get(name):
            changed[name] = new.get(name, "")
    return changed


def compile_settings(values: Mapping[str, Any]) -> dict[str, Any]:
    """Map resolved standard engine and graph knobs to ``torch_tensorrt.compile`` options.

    Precision is left to the workload: modules may cast weights and inputs, or
    use Torch-TensorRT autocast (:func:`autocast_settings`).
    """
    settings: dict[str, Any] = {}
    if "graph.min_block_size" in values:
        settings["min_block_size"] = values["graph.min_block_size"]
    if "engine.optimization_level" in values:
        settings["optimization_level"] = values["engine.optimization_level"]
    if "engine.tiling_optimization_level" in values:
        settings["tiling_optimization_level"] = values[
            "engine.tiling_optimization_level"
        ]
    return settings


def autocast_settings(precision: str) -> dict[str, Any]:
    """Torch-TensorRT autocast options for ``engine.precision``."""
    import torch

    if precision == "fp32":
        return {}
    low = {"fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    return {"enable_autocast": True, "autocast_low_precision_type": low}
