"""The test-suite manifest — pure data describing every test job.

A *suite* is one ``pytest`` invocation against one subsystem. A CI/local *job*
is a (suite x variant) pair, optionally sharded across N runners. Suites are
grouped two ways:

  * ``tier``  — the legacy L0/L1/L2 grouping, kept so the migration can be
    coverage-equivalent to today's ``ci_helpers.sh`` (``ci matrix --tier l0``).
  * ``lanes`` — the target grouping (``fast`` / ``full`` / ``nightly``) the
    redesign moves to (``ci matrix --lane fast``). Depth within a subsystem is
    expressed by a marker on the test, not a filename prefix.

Deliberate differences from the bash tiers (improvements, not regressions):
  * ``hlo/`` ran in BOTH l0_core (standard) and l1_dynamo_core — wasteful. It is
    one ``dynamo-hlo`` suite here, run once.
  * l2_plugin re-ran the whole ``conversion/`` dir (already covered by
    ``dynamo-converters``) and pointed four ``--junitxml`` runs at ONE path, so
    three suites' results vanished from the aggregate. Each suite here owns a
    unique junit name (derived from ``name`` + shard), and the redundant
    ``conversion/`` re-run is dropped.
  * L2 suites used raw ``pytest`` (no rerun wrapper / no repro hint); every suite
    now runs through the same wrapper uniformly (gated by ``TRT_PYTEST_RERUNS``).

Validate the manifest:  ``python -m tests.ci doctor``
Inspect a command:      ``python -m tests.ci run <name> --dry-run``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Tier = Literal["l0", "l1", "l2"]
# python-only validates the PYTHON_ONLY=1 wheel (no C++ runtime) against the
# dynamo runtime suite. It's its own lane because it pairs a distinct BUILD mode
# with a focused test set, orthogonal to fast/full/nightly depth.
Lane = Literal["fast", "full", "nightly", "python-only"]
Variant = Literal["standard", "rtx"]
# Test channels. SBSA (linux-aarch64) is build-only — no GPU test runners — so it
# is a build channel handled at the workflow level, not a suite platform here.
Platform = Literal["linux-x86_64", "windows"]

ALL_VARIANTS: tuple[Variant, ...] = ("standard", "rtx")
ALL_PLATFORMS: tuple[Platform, ...] = ("linux-x86_64", "windows")


@dataclass(frozen=True)
class Suite:
    """One ``pytest`` invocation against one subsystem.

    Fields map 1:1 onto the command the runner builds. ``overrides`` lets a
    single suite differ per variant (e.g. RTX selects a different path set)
    without splitting it into two entries with two names.
    """

    name: str
    tier: Tier
    lanes: tuple[Lane, ...]
    cwd: str = "tests/py/dynamo"  # relative to repo root
    paths: tuple[str, ...] = ()  # pytest positionals (rel to cwd); globs ok
    markers: str | None = None  # -m EXPR
    keyword: str | None = None  # -k EXPR
    dist: str | None = None  # --dist=loadscope
    maxfail: int | None = None  # --maxfail=N
    ir: str | None = None  # --ir torch_compile
    jobs: str | None = None  # xdist default: None=serial, "8"/"auto"/"4"
    reruns: bool = True  # wrap in the flake-rerun helper
    verbose: bool = False  # -v
    variants: tuple[Variant, ...] = ALL_VARIANTS
    platforms: tuple[Platform, ...] = ALL_PLATFORMS  # channels this suite runs on
    setup: tuple[str, ...] = ()  # named pre-steps: hub|executorch|cuda-core|mpi
    follow: tuple[tuple[str, ...], ...] = ()  # extra argv to run AFTER pytest
    env: dict[str, str] = field(default_factory=dict)
    overrides: dict[str, dict[str, Any]] = field(default_factory=dict)  # per-variant

    def for_variant(self, variant: Variant) -> dict[str, Any]:
        """This suite's effective fields for ``variant`` (applies overrides)."""
        base = {
            f: getattr(self, f)
            for f in (
                "cwd",
                "paths",
                "markers",
                "keyword",
                "dist",
                "maxfail",
                "ir",
                "jobs",
                "reruns",
                "verbose",
                "setup",
                "follow",
                "env",
            )
        }
        base.update(self.overrides.get(variant, {}))
        return base


# ── L0 — smoke / fast lane ────────────────────────────────────────────────────
_L0: list[Suite] = [
    Suite(
        "dynamo-converters",
        tier="l0",
        lanes=("fast", "full"),
        paths=("conversion/",),
        dist="--dist=loadscope",
        maxfail=20,
        jobs="8",
        # RTX does not shard converters with loadscope.
        overrides={"rtx": {"dist": None}},
    ),
    Suite(
        "dynamo-runtime-smoke",
        tier="l0",
        lanes=("fast", "full"),
        paths=("runtime/test_000_*",),
        jobs="8",
    ),
    Suite(
        "dynamo-partitioning-smoke",
        tier="l0",
        lanes=("fast", "full"),
        paths=("partitioning/test_000_*",),
        jobs="8",
        # RTX runs the whole partitioning suite (no smoke subset split).
        overrides={"rtx": {"paths": ("partitioning/",)}},
    ),
    Suite(
        "dynamo-lowering",
        tier="l0",
        lanes=("fast", "full"),
        paths=("lowering/",),
        jobs="8",
    ),
    Suite(
        "py-core",
        tier="l0",
        lanes=("fast", "full"),
        cwd="tests/py/core",
        paths=(".",),
        jobs="8",
    ),
    Suite(
        "ts-api",
        tier="l0",
        lanes=("fast", "full"),
        cwd="tests/py/ts",
        paths=("api/",),
        setup=("hub",),
        variants=("standard",),
    ),
]

# ── L1 — critical-path / full lane ────────────────────────────────────────────
_L1: list[Suite] = [
    Suite(
        "dynamo-runtime",
        tier="l1",
        lanes=("full",),
        paths=("runtime/test_001_*",),
        jobs="8",
    ),
    Suite(
        "dynamo-partitioning",
        tier="l1",
        lanes=("full",),
        paths=("partitioning/test_001_*",),
        jobs="8",
        variants=("standard",),
    ),
    Suite(
        # Was run in BOTH l0_core (std) and l1_dynamo_core (both) — deduped to once.
        "dynamo-hlo",
        tier="l1",
        lanes=("full",),
        paths=("hlo/",),
        jobs="8",
    ),
    Suite(
        "dynamo-models-critical",
        tier="l1",
        lanes=("full",),
        paths=("models/",),
        markers="critical",
    ),
    Suite(
        "torch-compile-backend",
        tier="l1",
        lanes=("full",),
        paths=("backend/",),
    ),
    Suite(
        "torch-compile-models-critical",
        tier="l1",
        lanes=("full",),
        paths=("models/test_models.py", "models/test_dyn_models.py"),
        markers="critical",
        ir="torch_compile",
    ),
    Suite(
        "ts-models",
        tier="l1",
        lanes=("full",),
        cwd="tests/py/ts",
        paths=("models/",),
        setup=("hub",),
        variants=("standard",),
    ),
]

# ── L2 — exhaustive / full + nightly ──────────────────────────────────────────
_L2: list[Suite] = [
    Suite(
        "torch-compile-models",
        tier="l2",
        lanes=("full", "nightly"),
        paths=("models/test_models.py", "models/test_dyn_models.py"),
        markers="not critical",
        ir="torch_compile",
        jobs="auto",
    ),
    Suite(
        "dynamo-models",
        tier="l2",
        lanes=("full", "nightly"),
        paths=("models/",),
        markers="not critical",
        jobs="auto",
    ),
    Suite(
        "dynamo-llm",
        tier="l2",
        lanes=("nightly",),
        paths=("llm/",),
        jobs="auto",
    ),
    Suite(
        "dynamo-runtime-full",
        tier="l2",
        lanes=("full", "nightly"),
        paths=("runtime/",),
        keyword="not test_000_ and not test_001_",
        jobs="auto",
    ),
    Suite(
        "executorch",
        tier="l2",
        lanes=("nightly",),
        paths=("executorch/",),
        setup=("executorch",),
        jobs="auto",
        variants=("standard",),
        platforms=("linux-x86_64",),
    ),
    Suite(
        # Standard: the automatic-plugin trio. RTX: the whole automatic_plugin dir.
        # (The redundant conversion/ re-run from the old l2_plugin is dropped.)
        "plugins-automatic",
        tier="l2",
        lanes=("nightly",),
        jobs="auto",
        paths=(
            "automatic_plugin/test_automatic_plugin.py",
            "automatic_plugin/test_automatic_plugin_with_attrs.py",
            "automatic_plugin/test_flashinfer_rmsnorm.py",
        ),
        overrides={"rtx": {"paths": ("automatic_plugin/",)}},
    ),
    Suite(
        "kernels",
        tier="l2",
        lanes=("nightly",),
        cwd="tests/py/kernels",
        paths=(".",),
        setup=("cuda-core",),
        jobs="auto",
        variants=("standard",),
        platforms=("linux-x86_64",),
    ),
    Suite(
        "ts-integrations",
        tier="l2",
        lanes=("nightly",),
        cwd="tests/py/ts",
        paths=("integrations/",),
        setup=("hub",),
        jobs="auto",
        variants=("standard",),
    ),
    Suite(
        "distributed",
        tier="l2",
        lanes=("nightly",),
        paths=(
            "distributed/test_nccl_ops.py",
            "distributed/test_native_nccl.py",
            "distributed/test_export_save_load.py",
        ),
        jobs="auto",
        verbose=True,
        reruns=False,
        variants=("standard",),
        platforms=("linux-x86_64",),
        setup=("mpi",),
        env={"USE_HOST_DEPS": "1", "CI_BUILD": "1", "USE_TRTLLM_PLUGINS": "1"},
        follow=(
            (
                "-m",
                "torch_tensorrt.distributed.run",
                "--nproc_per_node=2",
                "distributed/test_native_nccl.py",
                "--multirank",
            ),
            (
                "-m",
                "torch_tensorrt.distributed.run",
                "--nproc_per_node=2",
                "distributed/test_export_save_load.py",
                "--multirank",
            ),
        ),
    ),
]

# ── python-only — validates the PYTHON_ONLY=1 wheel against the runtime suite ──
_PYTHON_ONLY: list[Suite] = [
    Suite(
        "python-only-runtime",
        tier="l1",
        lanes=("python-only",),
        paths=("runtime/",),
        jobs="8",
        # Runs for BOTH backends: the PYTHON_ONLY=1 wheel is validated against
        # standard TensorRT and TensorRT-RTX (variants default to both).
    ),
]

SUITES: tuple[Suite, ...] = tuple(_L0 + _L1 + _L2 + _PYTHON_ONLY)


def by_name(name: str) -> Suite:
    for s in SUITES:
        if s.name == name:
            return s
    raise KeyError(f"no suite named {name!r}; try `python -m tests.ci list`")
