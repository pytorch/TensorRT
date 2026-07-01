"""The one engine that turns a ``Suite`` into a ``pytest`` command and runs it.

This is the *only* place that knows pytest mechanics (xdist workers, junit paths,
the flake-rerun wrapper, ``--dist``, optional-dep setup, variant resolution). The
manifest (``suites.py``) stays pure data.
"""

from __future__ import annotations

import glob
import os
import shlex
import subprocess
import sys
from pathlib import Path

from .suites import SUITES, Suite, Variant, by_name

# Repo root: tests/ci/runner.py -> parents[2]. Honor TRT_REPO_ROOT like the bash.
REPO_ROOT = Path(
    os.environ.get("TRT_REPO_ROOT", str(Path(__file__).resolve().parents[2]))
)

# Known transient cudagraph/TRT-driver flake signatures. Expand ONLY with
# concrete evidence — a broad regex hides real bugs.
_RERUN_ARGS = [
    "--reruns", "1", "--reruns-delay", "5",
    "--only-rerun", "cudaErrorStreamCaptureInvalidated",
    "--only-rerun", "Stream capture invalidated",
]


def _launcher() -> list[str]:
    """The python/pytest launcher. CI leaves PYTHON unset (-> container python);
    the justfile sets PYTHON='uv run --no-sync python' to use the built venv."""
    return shlex.split(os.environ.get("PYTHON", "python"))


def _results_dir() -> Path:
    d = os.environ.get("RUNNER_TEST_RESULTS_DIR")
    if not d:
        tmp = os.environ.get("TMPDIR", "/tmp")
        d = str(Path(tmp) / "trt_test_results")
    Path(d).mkdir(parents=True, exist_ok=True)
    return Path(d)


def junit_path(suite: Suite) -> Path:
    """Unique per suite (no more four-runs-one-file collisions)."""
    return _results_dir() / f"{suite.name}.xml"


def _nproc(jobs: str | None) -> list[str]:
    """``-n`` token. TRT_JOBS overrides the suite default; jobs=None -> serial."""
    if jobs is None:
        return []
    return ["-n", os.environ.get("TRT_JOBS") or jobs]


def _reruns_enabled(reruns: bool) -> bool:
    return reruns and os.environ.get("TRT_PYTEST_RERUNS", "1") != "0"


def _expand_paths(cwd: Path, paths: tuple[str, ...]) -> list[str]:
    """Shell-style glob expansion (the bash relied on the shell for this).

    A pattern with ``*`` is expanded relative to ``cwd`` and sorted; a pattern
    that matches nothing is kept literal (matches bash nullglob-off behavior, so
    pytest reports the missing path rather than silently collecting 0 tests).
    """
    out: list[str] = []
    for p in paths:
        if "*" in p:
            matches = sorted(
                str(Path(m).relative_to(cwd)) for m in glob.glob(str(cwd / p))
            )
            out.extend(matches or [p])
        else:
            out.append(p)
    return out


def build_pytest_args(suite: Suite, variant: Variant) -> list[str]:
    """The pytest args (everything after ``-m pytest``) for this suite+variant."""
    v = suite.for_variant(variant)
    cwd = REPO_ROOT / v["cwd"]
    args: list[str] = []
    if _reruns_enabled(v["reruns"]):
        args += _RERUN_ARGS
    if v["markers"]:
        args += ["-m", v["markers"]]
    args += ["-ra"]
    args += _nproc(v["jobs"])
    args += ["--junitxml", str(junit_path(suite))]
    if v["dist"]:
        args += [v["dist"]]
    if v["maxfail"] is not None:
        args += [f"--maxfail={v['maxfail']}"]
    if v["ir"]:
        args += ["--ir", v["ir"]]
    if v["keyword"]:
        args += ["-k", v["keyword"]]
    if v["verbose"]:
        args += ["-v"]
    args += _expand_paths(cwd, tuple(v["paths"]))
    return args


def _setup_commands(step: str) -> list[tuple[list[str], Path]]:
    """(argv, cwd) pairs for a named setup step."""
    launcher = _launcher()
    if step == "hub":
        return [(launcher + ["hub.py"], REPO_ROOT / "tests/modules")]
    if step == "executorch":
        return [(launcher + ["-m", "pip", "install", "pyyaml", "executorch>=1.3.1"],
                 REPO_ROOT)]
    if step == "cuda-core":
        return [(launcher + ["-m", "pip", "install", "cuda-python", "cuda-core"],
                 REPO_ROOT)]
    if step == "mpi":
        return [(["dnf", "install", "-y", "mpich", "mpich-devel",
                  "openmpi", "openmpi-devel"], REPO_ROOT)]
    raise KeyError(f"unknown setup step {step!r} in a suite definition")


def describe(suite: Suite, variant: Variant) -> str:
    """The full command line, for --dry-run / show (quoting-safe display)."""
    v = suite.for_variant(variant)
    pre = []
    for step in v["setup"]:
        for argv, cwd in _setup_commands(step):
            pre.append(f"  (cd {cwd.relative_to(REPO_ROOT)} && {shlex.join(argv)})")
    cmd = shlex.join(_launcher() + ["-m", "pytest"] + build_pytest_args(suite, variant))
    lines = pre + [f"  (cd {v['cwd']} && {cmd})"]
    for f in v["follow"]:
        lines.append(f"  (cd {v['cwd']} && {shlex.join(_launcher() + list(f))})")
    return "\n".join(lines)


def run_suite(
    suite: Suite,
    variant: Variant,
    *,
    dry_run: bool = False,
    extra: list[str] | None = None,
) -> int:
    """Run setup steps, the pytest command, then any follow commands. Returns the
    process exit code (non-zero on first failure), mirroring the bash tiers."""
    v = suite.for_variant(variant)
    extra = extra or []
    env = {**os.environ, **{k: str(val) for k, val in v["env"].items()}}
    cwd = REPO_ROOT / v["cwd"]
    pytest_cmd = _launcher() + ["-m", "pytest"] + build_pytest_args(suite, variant) + extra

    if dry_run:
        print(describe(suite, variant))
        if extra:
            print(f"  # + extra pytest args: {shlex.join(extra)}")
        return 0

    for step in v["setup"]:
        for argv, scwd in _setup_commands(step):
            print(f"==> setup[{step}]: {shlex.join(argv)}", flush=True)
            rc = subprocess.run(argv, cwd=scwd, env=env).returncode
            if rc != 0:
                print(f"::warning::setup step {step!r} exited {rc}", flush=True)

    print(f"==> {suite.name} [{variant}]: {shlex.join(pytest_cmd)}", flush=True)
    rc = subprocess.run(pytest_cmd, cwd=cwd, env=env).returncode
    if rc != 0:
        repro = shlex.join(["uv", "run", "--no-sync", "pytest"]
                           + build_pytest_args(suite, variant) + extra)
        print(f"::warning::{suite.name} failed. Reproduce: cd {v['cwd']} && {repro}",
              flush=True)
        return rc

    for f in v["follow"]:
        fcmd = _launcher() + list(f)
        print(f"==> {suite.name} follow: {shlex.join(fcmd)}", flush=True)
        frc = subprocess.run(fcmd, cwd=cwd, env=env).returncode
        if frc != 0:
            return frc
    return 0


def select(
    *,
    lane: str | None = None,
    tier: str | None = None,
    variant: str | None = None,
    platform: str | None = None,
    names: list[str] | None = None,
) -> list[tuple[Suite, Variant]]:
    """All (suite, variant) jobs matching the filters. No filter on an axis = all."""
    jobs: list[tuple[Suite, Variant]] = []
    pool = [by_name(n) for n in names] if names else list(SUITES)
    for s in pool:
        if lane is not None and lane not in s.lanes:
            continue
        if tier is not None and s.tier != tier:
            continue
        if platform is not None and platform not in s.platforms:
            continue
        for var in s.variants:
            if variant is not None and var != variant:
                continue
            jobs.append((s, var))
    return jobs


def matrix(**filters: str | None) -> list[dict[str, str]]:
    """GitHub-Actions matrix ``include`` entries for the selected jobs."""
    return [
        {"suite": s.name, "variant": var, "tier": s.tier,
         "cwd": s.for_variant(var)["cwd"]}
        for s, var in select(**filters)
    ]
