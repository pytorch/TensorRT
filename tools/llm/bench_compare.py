# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
.. _bench_compare:

Head-to-head benchmarks of Torch-TensorRT configs, PyTorch eager and Inductor
=============================================================================

Measures configurations of one ``torch_tensorrt.autotune`` workload (TensorRT
configs, or the same model run eagerly or compiled by Inductor) with the
measurement that Auto-Tuner sessions use, so that every configuration follows
one protocol: the workload's ``Timing``, accuracy checked against one FP32
reference on the first call and again after timing, the measurement's own
quiet-GPU wait, and re-timing of unstable measurements.

``all`` adds what a fair comparison across processes needs:

* every run is a fresh Python process and builds cold: Inductor's and
  Triton's caches and ``TMPDIR`` (where Torch-TensorRT keeps TensorRT's
  timing cache) are the run's own, under the output directory, emptied
  before the run starts and deleted after it (``--keep-caches`` keeps them);
* rounds run the configurations in balanced orders, the rows of a Williams
  Latin square: over ``n`` rounds (``2n`` for an odd number ``n`` of
  configurations, and over any multiple of that) each one runs equally often
  in every position and right after every other one;
* before every run, and before computing the reference, the driver takes the
  GPU lock and waits until the GPU is idle; the run inherits the lock, and a
  run whose driver stops is stopped too (see below);
* every run is checked against the same reference outputs, whose SHA-256 it
  records together with the accuracy limits it missed; the summary gives
  each configuration's median, over rounds, of its runs' median latencies,
  and warns when the runs used different reference outputs or ran different
  code (Torch-TensorRT's source, the workload's directory or PyTorch).

Commands::

    python tools/llm/bench_compare.py reference SPEC [OUT_DIR]
    python tools/llm/bench_compare.py all SPEC OUT_DIR [CONFIG ...] [--rounds 4]
    python tools/llm/bench_compare.py summary OUT_DIR [--base CONFIG] [--write]
    python tools/llm/bench_compare.py run SPEC CONFIG OUT.json [--reference PATH]
    python tools/llm/bench_compare.py quiet

``reference`` computes the reference outputs in this process (``all``
computes them first when they are missing). ``run`` measures one
configuration in this process. ``quiet`` waits for an idle GPU, for scripts
that start other GPU jobs. ``all`` keeps its resolved spec in
``OUT_DIR/spec.json``, every run's result and log in ``OUT_DIR/r<round>/``
and the summary in ``OUT_DIR/summary.json``; ``summary`` prints the summary
again (``--write`` also rewrites ``summary.json``). ``all`` refuses an output
directory that already holds a head-to-head unless ``--resume`` is given,
which keeps the runs that left a result and runs the others: runs not made
yet (further rounds and configurations included), and runs that crashed or
timed out, whose result the driver wrote.

A spec is a JSON file::

    {
      "workload": "tools/llm/agentic_decode.py:create_case",
      "args": {"model": "Qwen/Qwen3-0.6B", "isl": 8192, "osl": 64},
      "reference": "reference.pt",
      "head_to_head": ["trt-baseline", "trt-session", "inductor-max-autotune"],
      "configs": {
        "trt-baseline": {"recipe": {}},
        "trt-session": {"recipe": {"workload.cache": "static_v3"}},
        "trt-session-cudagraphs": {
          "extends": "trt-session", "recipe": {"runtime.cudagraphs": true}
        },
        "inductor-max-autotune": {
          "bench": {"backend": "inductor", "mode": "max-autotune"}
        }
      }
    }

* ``workload``: a ``path/to/file.py:function`` factory, relative to the
  repository root or absolute, or ``package.module:function``; ``args`` are
  its keyword arguments.
* ``reference``: the reference outputs, relative to the output directory or
  absolute (``--reference`` overrides it). An Auto-Tuner session's
  ``reference.pt`` for the same workload and arguments holds the same
  outputs.
* ``configs``: a ``recipe`` is an Auto-Tuner config of the workload's knobs,
  resolved as a session resolves one (defaults filled in, requirements
  enforced: a recipe that breaks one fails at the ``recipe`` stage).
  ``bench`` entries reach ``Workload.build`` as ``bench.<key>`` values,
  which are not knobs; the workloads in this directory read
  ``bench.backend`` (``tensorrt``, ``eager`` or ``inductor``),
  ``bench.mode`` (a ``torch.compile`` mode), ``bench.attn`` (a Hugging Face
  ``attn_implementation``), ``bench.inductor`` (``torch._inductor.config``
  overrides) and keys of their own (see their docstrings). ``extends`` starts
  from another configuration's ``recipe`` and ``bench`` and applies this
  one's entries on top of them.
* ``head_to_head``: the configurations ``all`` runs when none are named.
* optional: ``timeout_s`` per run (default 3600); ``noise_control``, the
  measurement's ``quiet_gpu_s``, ``retries`` and ``max_spread`` (as in
  ``NoiseControl``; default 60, 2 and 0.03); ``accuracy``, limits that
  replace the workload's defaults per metric, like a session's
  ``--accuracy``; ``description`` and ``notes``.

``--session DIR`` takes ``trt-session`` from a finished Auto-Tuner session:
its winner's config (configurations that extend ``trt-session`` follow), the
edited files that config uses and the session's accuracy limits. The
session must be one of the spec's workload and arguments.

GPU lock and idle wait (``all``, around each run and the reference):

* ``--lock FILE`` (default ``$BENCH_GPU_LOCK``, else
  ``/tmp/torch_tensorrt_gpu.lock``; ``none`` turns it off) holds an
  exclusive ``flock(2)`` lock on FILE while the driver waits for an idle GPU
  and while the run lasts. It is the lock that ``flock FILE command`` takes,
  so start the machine's other GPU jobs that way. As under ``flock``, the
  run inherits the locked file, so the lock holds as long as the run's
  process does, even if the driver is gone.
* ``--quiet-command CMD`` (default ``$BENCH_GPU_QUIET``) runs CMD before each
  run and waits for it to exit. Otherwise the driver waits until
  ``nvidia-smi`` shows the GPU (the first one in ``CUDA_VISIBLE_DEVICES``,
  else GPU 0) at most 3% busy, with at least ``--quiet-min-free-mb`` MB free,
  in ``--quiet-s`` consecutive one-second samples (default 10; 0 turns the
  wait off). It gives up after ``--quiet-timeout-s`` seconds (default 3600):
  the run then goes ahead, and its line says so.

A run is stopped with its driver: on Ctrl-C, SIGTERM or SIGHUP the driver
kills the run's process group and exits (status 130, 143 or 129), and a
driver killed outright takes its run along through Linux's parent-death
signal. The run then leaves no result, so ``--resume`` measures it again.
When a run exits, whatever it left running in its process group is killed.
A run past ``timeout_s`` first gets SIGUSR1, on which it writes its Python
stacks to its log, and is then killed with its process group.

Runs are offline (``HF_HUB_OFFLINE=1`` unless it is set), so the model must
be in the Hugging Face cache. Computing the reference uses the driver's own
environment, so it downloads the model unless ``HF_HUB_OFFLINE`` is set
there.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, NamedTuple, Sequence

# tools/llm/bench_compare.py -> the repository root.
REPO = Path(__file__).resolve().parents[2]
# The configuration that --session fills in.
SESSION_CONFIG = "trt-session"
DEFAULT_TIMEOUT_S = 3600.0
DEFAULT_NOISE_CONTROL: dict[str, float] = {
    "quiet_gpu_s": 60.0,
    "retries": 2,
    "max_spread": 0.03,
}
# The GPU counts as idle at or below this utilization (percent).
IDLE_UTILIZATION = 3.0
LOCK_ENV = "BENCH_GPU_LOCK"
QUIET_ENV = "BENCH_GPU_QUIET"
# One fixed path for the whole machine (not the temporary directory, which
# depends on TMPDIR), so that every job that uses the default excludes the
# others.
DEFAULT_LOCK = Path("/tmp/torch_tensorrt_gpu.lock")
# Set by ``all`` for its runs: the driver's process ID (see _die_with_driver).
DRIVER_ENV = "BENCH_COMPARE_DRIVER"
# How long a run past its timeout gets to write its stacks before SIGKILL.
STACKS_GRACE_S = 3.0
# ``written_by`` of the results the driver writes for runs that left none.
DRIVER_RECORD = "driver"

_SPEC_KEYS = frozenset(
    {
        "workload",
        "args",
        "reference",
        "configs",
        "head_to_head",
        "timeout_s",
        "noise_control",
        "accuracy",
        "description",
        "notes",
        # Written by ``all`` into OUT_DIR/spec.json (and kept by older specs).
        "session",
        "edits_dir",
        "source",
        "note",
    }
)
_CONFIG_KEYS = frozenset({"recipe", "bench", "extends"})
# Keys that decide whether two resolved specs measure the same thing.
_IDENTITY_KEYS = (
    "workload",
    "args",
    "reference",
    "timeout_s",
    "noise_control",
    "accuracy",
)
_CONFIG_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_COMPARATORS: dict[str, Callable[[float, float], bool]] = {
    "<=": lambda actual, limit: actual <= limit,
    ">=": lambda actual, limit: actual >= limit,
    "<": lambda actual, limit: actual < limit,
    ">": lambda actual, limit: actual > limit,
}


class SpecError(ValueError):
    """A spec, session or output directory that cannot be used as given."""


class Terminated(BaseException):
    """SIGTERM or SIGHUP reached the driver (see :func:`stop_on_signals`).

    A ``BaseException``, like ``KeyboardInterrupt``, so that only the
    handlers that clean up (and kill the run in progress) see it.
    """

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


# --------------------------------------------------------------------- spec


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text())
    except OSError as error:
        raise SpecError(f"Cannot read {path}: {error.strerror}") from None
    except ValueError as error:
        raise SpecError(f"{path} is not valid JSON: {error}") from None


def _write_json(path: Path, value: Any) -> None:
    """Write ``value`` atomically, so a reader never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(value, stream, indent=2, default=str)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _workload(factory: Any, source: Path) -> str:
    """``factory`` with a ``.py`` file made absolute (relative to the repository)."""
    if not isinstance(factory, str) or ":" not in factory:
        raise SpecError(
            f"{source}: workload must be 'path/to/file.py:function' or "
            f"'package.module:function'; got {factory!r}"
        )
    module, _, function = factory.rpartition(":")
    if not module.endswith(".py"):
        return factory
    path = Path(module).expanduser()
    if not path.is_absolute():
        path = REPO / path
    if not path.is_file():
        raise SpecError(f"{source}: the workload file {path} does not exist")
    return f"{path.resolve()}:{function}"


def _location(name: str, value: Any, out_dir: str | Path | None, source: Path) -> Path:
    """An absolute path from a spec entry that may be relative to ``out_dir``."""
    if not isinstance(value, str) or not value:
        raise SpecError(f"{source}: {name} must be a path; got {value!r}")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if out_dir is None:
        hint = "pass --reference PATH, or " if name == "reference" else ""
        raise SpecError(
            f"{source}: {name} {value!r} is relative to an output directory, "
            f"and none was given ({hint}make it absolute in the spec)"
        )
    return (Path(out_dir) / path).resolve()


def _number(name: str, value: Any, source: Path, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        kind = "a positive" if positive else "a non-negative"
        raise SpecError(f"{source}: {name} must be {kind} number; got {value!r}")
    return float(value)


def _noise_control(value: Any, source: Path) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise SpecError(f"{source}: noise_control must be an object")
    unknown = set(value) - set(DEFAULT_NOISE_CONTROL)
    if unknown:
        raise SpecError(
            f"{source}: unknown noise_control settings {sorted(unknown)}; "
            f"choose from {sorted(DEFAULT_NOISE_CONTROL)}"
        )
    merged = {**DEFAULT_NOISE_CONTROL, **value}
    settings: dict[str, float] = {
        "quiet_gpu_s": _number(
            "noise_control.quiet_gpu_s", merged["quiet_gpu_s"], source
        ),
        "max_spread": _number(
            "noise_control.max_spread", merged["max_spread"], source, positive=True
        ),
    }
    retries = merged["retries"]
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise SpecError(
            f"{source}: noise_control.retries must be a non-negative integer"
        )
    settings["retries"] = retries
    return settings


def _accuracy(value: Any, source: Path) -> dict[str, list[Any]]:
    if not isinstance(value, Mapping):
        raise SpecError(f"{source}: accuracy must map metrics to [comparator, limit]")
    bounds: dict[str, list[Any]] = {}
    for metric, bound in value.items():
        if (
            not isinstance(bound, (list, tuple))
            or len(bound) != 2
            or bound[0] not in _COMPARATORS
            or isinstance(bound[1], bool)
            or not isinstance(bound[1], (int, float))
            or not math.isfinite(bound[1])
        ):
            raise SpecError(
                f"{source}: the accuracy limit of {metric!r} must be [comparator, "
                f"number] with a comparator in {sorted(_COMPARATORS)}; got {bound!r}"
            )
        bounds[str(metric)] = [bound[0], float(bound[1])]
    return bounds


def _mapping(value: Any, what: str, source: Path) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(k, str) for k in value):
        raise SpecError(f"{source}: {what} must be an object with string keys")
    return copy.deepcopy(dict(value))


def _configs(value: Any, source: Path) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or not value:
        raise SpecError(f"{source}: configs must be a non-empty object")
    configs: dict[str, dict[str, Any]] = {}
    for name, entry in value.items():
        if not isinstance(name, str) or not _CONFIG_NAME.fullmatch(name):
            raise SpecError(
                f"{source}: config name {name!r} must start with a letter or digit "
                "and use only letters, digits and '_.+-' (it names files)"
            )
        if not isinstance(entry, Mapping):
            raise SpecError(f"{source}: config {name!r} must be an object")
        unknown = set(entry) - _CONFIG_KEYS
        if unknown:
            raise SpecError(
                f"{source}: config {name!r} has unknown keys {sorted(unknown)}; "
                f"a config has {sorted(_CONFIG_KEYS)}"
            )
        config: dict[str, Any] = {
            "recipe": _mapping(entry.get("recipe", {}), f"{name}.recipe", source),
            "bench": _mapping(entry.get("bench", {}), f"{name}.bench", source),
        }
        if "extends" in entry:
            if not isinstance(entry["extends"], str):
                raise SpecError(f"{source}: {name}.extends must name a config")
            config["extends"] = entry["extends"]
        configs[name] = config
    return configs


def _flatten(
    configs: Mapping[str, Mapping[str, Any]], source: Path
) -> dict[str, dict[str, Any]]:
    """Apply ``extends``: each config's recipe and bench on top of its base's."""
    flat: dict[str, dict[str, Any]] = {}

    def resolve(name: str, chain: tuple[str, ...]) -> dict[str, Any]:
        if name in flat:
            return flat[name]
        if name in chain:
            cycle = " -> ".join((*chain[chain.index(name) :], name))
            raise SpecError(f"{source}: configs extend each other in a cycle: {cycle}")
        entry = configs[name]
        recipe: dict[str, Any] = {}
        bench: dict[str, Any] = {}
        base = entry.get("extends")
        if base is not None:
            if base not in configs:
                raise SpecError(
                    f"{source}: config {name!r} extends {base!r}, which the spec "
                    "does not define"
                )
            parent = resolve(base, (*chain, name))
            recipe.update(copy.deepcopy(parent["recipe"]))
            bench.update(copy.deepcopy(parent["bench"]))
        recipe.update(copy.deepcopy(entry["recipe"]))
        bench.update(copy.deepcopy(entry["bench"]))
        flat[name] = {"recipe": recipe, "bench": bench}
        return flat[name]

    for name in configs:
        resolve(name, ())
    return {name: flat[name] for name in configs}


def session_config(
    directory: str | Path, workload: str, args: Mapping[str, Any]
) -> dict[str, Any]:
    """What a finished Auto-Tuner session in ``directory`` contributes to a spec.

    Returns ``directory``, the ``winner_config``, the session's
    ``accuracy_bounds``, ``decision`` and ``stop_reason``. Refuses a session
    of another workload, or one with other arguments.
    """
    directory = Path(directory).expanduser().resolve()
    report_path, session_path = directory / "report.json", directory / "session.json"
    if not report_path.is_file() or not session_path.is_file():
        raise SpecError(
            f"{directory} is not a finished Auto-Tuner session (it needs "
            "report.json and session.json)"
        )
    report, session = _read_json(report_path), _read_json(session_path)
    winner = (report.get("winner") or {}).get("config")
    if not isinstance(winner, dict):
        raise SpecError(f"{report_path} names no winner config")
    recorded = session.get("workload_factory") or {}
    factory = str(recorded.get("factory", ""))
    ours = workload.rpartition(":")
    theirs = factory.rpartition(":")
    if (Path(theirs[0]).name, theirs[2]) != (Path(ours[0]).name, ours[2]) or dict(
        recorded.get("kwargs") or {}
    ) != dict(args):
        raise SpecError(
            f"{directory} is a session of {factory} with {recorded.get('kwargs')}, "
            f"not of this spec's {workload} with {dict(args)}"
        )
    return {
        "directory": str(directory),
        "winner_config": winner,
        "accuracy_bounds": report.get("accuracy_bounds") or {},
        "decision": report.get("decision"),
        "stop_reason": report.get("stop_reason"),
    }


def load_spec(
    path: str | Path,
    *,
    out_dir: str | Path | None = None,
    reference: str | Path | None = None,
    session: str | Path | None = None,
) -> dict[str, Any]:
    """Read the spec at ``path`` and resolve it into the form a run uses.

    The result has an absolute workload file, an absolute ``reference`` (a
    relative one is taken relative to ``out_dir``; ``reference`` overrides
    it, relative to the working directory), its defaults filled in and
    ``configs`` flattened (``extends`` applied). ``session`` replaces
    ``trt-session`` by that session's winner config (see
    :func:`session_config`). Raises :class:`SpecError` naming what is wrong.
    """
    source = Path(path)
    raw = _read_json(source)
    if not isinstance(raw, Mapping):
        raise SpecError(f"{source}: a spec is a JSON object")
    unknown = set(raw) - _SPEC_KEYS
    if unknown:
        raise SpecError(
            f"{source}: unknown keys {sorted(unknown)}; a spec has "
            f"{sorted(_SPEC_KEYS)}"
        )
    spec: dict[str, Any] = copy.deepcopy(dict(raw))
    spec["workload"] = _workload(raw.get("workload"), source)
    spec["args"] = _mapping(raw.get("args", {}), "args", source)
    if reference is not None:
        spec["reference"] = str(Path(reference).expanduser().resolve())
    else:
        spec["reference"] = str(
            _location("reference", raw.get("reference"), out_dir, source)
        )
    if raw.get("edits_dir") is not None:
        spec["edits_dir"] = str(
            _location("edits_dir", raw["edits_dir"], out_dir, source)
        )
    spec["timeout_s"] = _number(
        "timeout_s", raw.get("timeout_s", DEFAULT_TIMEOUT_S), source, positive=True
    )
    spec["noise_control"] = _noise_control(raw.get("noise_control", {}), source)
    spec["accuracy"] = _accuracy(raw.get("accuracy", {}), source)
    configs = _configs(raw.get("configs"), source)
    if session is not None:
        info = session_config(session, spec["workload"], spec["args"])
        configs[SESSION_CONFIG] = {
            "recipe": copy.deepcopy(info["winner_config"]),
            "bench": {},
        }
        spec["session"] = info
        spec["accuracy"] = {
            **_accuracy(info["accuracy_bounds"], Path(info["directory"])),
            **spec["accuracy"],
        }
        spec["edits_dir"] = info["directory"]
    spec["configs"] = _flatten(configs, source)
    head = raw.get("head_to_head")
    if head is not None:
        if not isinstance(head, list) or not all(isinstance(n, str) for n in head):
            raise SpecError(f"{source}: head_to_head must list config names")
        missing = [name for name in head if name not in spec["configs"]]
        if missing:
            raise SpecError(
                f"{source}: head_to_head names configs the spec lacks: {missing}"
            )
    return spec


def edit_digests(spec: Mapping[str, Any]) -> set[str]:
    """The edited files (``edit.<name>`` values, SHA-256) that the spec's recipes use."""
    return {
        value
        for config in spec["configs"].values()
        for key, value in config["recipe"].items()
        if key.startswith("edit.") and isinstance(value, str) and value
    }


def _differences(old: Mapping[str, Any], new: Mapping[str, Any]) -> list[str]:
    """Why two resolved specs do not measure the same thing (shared configs only)."""
    found = [key for key in _IDENTITY_KEYS if old.get(key) != new.get(key)]
    found += [
        f"config {name}"
        for name, config in new.get("configs", {}).items()
        if name in old.get("configs", {}) and old["configs"][name] != config
    ]
    return found


# --------------------------------------------------------------- the order


def round_order(configs: Sequence[str], index: int) -> list[str]:
    """The order in which round ``index`` (from 0) runs ``configs``.

    Rows of a Williams design (a balanced Latin square): the first row is
    ``0, 1, n-1, 2, n-2, ...`` and each later row adds one to every entry,
    modulo ``n``; for an odd ``n`` the next ``n`` rows are those rows
    reversed. Over ``n`` rounds (``2n`` for an odd ``n``) every configuration
    runs equally often in every position and right after every other one, so
    neither its place in a round nor the run before it favors one of them.
    """
    count = len(configs)
    if count == 0:
        return []
    first, low, high = [0], 1, count - 1
    while len(first) < count:
        first.append(low)
        low += 1
        if len(first) < count:
            first.append(high)
            high -= 1
    row = index % (count if count % 2 == 0 else 2 * count)
    order = [configs[(position + row) % count] for position in first]
    return order[::-1] if row >= count else order


# ------------------------------------------------------- lock and idle GPU


@contextlib.contextmanager
def gpu_lock(path: str | Path | None) -> Iterator[int | None]:
    """Hold an exclusive ``flock(2)`` lock on ``path`` (the lock ``flock(1)`` takes).

    Yields the locked file's descriptor (None without a ``path``). A
    ``flock`` lock belongs to the open file, not to a process: a child that
    inherits the descriptor (``subprocess.Popen(pass_fds=...)``) keeps the
    lock until it exits too, as the command does under ``flock FILE
    command``.
    """
    if path is None:
        yield None
        return
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(path, os.O_RDONLY | os.O_CREAT, 0o666)
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"waiting for the GPU lock {path}", flush=True)
            fcntl.flock(handle, fcntl.LOCK_EX)
        yield handle
    finally:
        os.close(handle)


def visible_gpu() -> str | None:
    """The ``nvidia-smi --id`` of the GPU the runs use (None: the first one)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return "0"
    return visible.split(",")[0].strip() or None


def read_gpu(device: str | None = None) -> tuple[float, float] | None:
    """Utilization (%) and free memory (MB) of ``device`` from nvidia-smi; None if unknown."""
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return None
    command = [
        binary,
        "--query-gpu=utilization.gpu,memory.free",
        "--format=csv,noheader,nounits",
    ]
    if device:
        command.insert(1, f"--id={device}")
    try:
        output = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=True
        ).stdout
        busy, free = (
            float(field) for field in output.strip().splitlines()[0].split(",")
        )
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None
    return busy, free


def wait_for_idle_gpu(
    samples: int,
    timeout_s: float,
    min_free_mb: float = 0.0,
    *,
    read: Callable[[], tuple[float, float] | None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[bool, str]:
    """Wait for ``samples`` consecutive one-second readings of an idle GPU.

    Idle is at most :data:`IDLE_UTILIZATION` percent busy with at least
    ``min_free_mb`` free. Returns whether the GPU was idle (or could not be
    read, or ``samples`` is 0) and a line saying what happened.
    """
    if samples <= 0:
        return True, "idle wait off"
    reader = read or (lambda: read_gpu(visible_gpu()))
    idle = waited = 0
    while True:
        reading = reader()
        if reading is None:
            return True, "idle wait skipped: nvidia-smi cannot read the GPU"
        busy, free = reading
        idle = idle + 1 if busy <= IDLE_UTILIZATION and free >= min_free_mb else 0
        if idle >= samples:
            return (
                True,
                f"GPU idle for {samples} s (waited {waited} s, {free:.0f} MB free)",
            )
        if waited >= timeout_s:
            return False, (
                f"GPU still busy after {waited} s ({busy:.0f}% busy, "
                f"{free:.0f} MB free); running anyway"
            )
        sleep(1.0)
        waited += 1


def run_quiet_command(command: str) -> tuple[bool, str]:
    """Run a user's idle-GPU command; whether it succeeded, and its last line."""
    try:
        process = subprocess.run(shlex.split(command), capture_output=True, text=True)
    except OSError as error:
        return False, f"{command!r} did not start: {error.strerror}"
    lines = (process.stdout + process.stderr).strip().splitlines()
    last = lines[-1] if lines else f"{command!r} printed nothing"
    if process.returncode:
        return False, f"{last} (exit status {process.returncode}); running anyway"
    return True, last


class Held(NamedTuple):
    """What :meth:`Gate.hold` yields."""

    # What the idle wait reported.
    quiet: str
    # The lock's descriptor, for the run to inherit (pass_fds); empty
    # without a lock.
    fds: tuple[int, ...]


class Gate:
    """The GPU lock and the idle wait that ``all`` puts around every run."""

    def __init__(
        self,
        lock: str | Path | None,
        quiet_command: str | None = None,
        quiet_s: int = 10,
        quiet_timeout_s: float = 3600.0,
        quiet_min_free_mb: float = 0.0,
    ) -> None:
        self.lock = Path(lock) if lock else None
        self.quiet_command = quiet_command or None
        self.quiet_s = quiet_s
        self.quiet_timeout_s = quiet_timeout_s
        self.quiet_min_free_mb = quiet_min_free_mb

    def wait(self) -> str:
        if self.quiet_command:
            return run_quiet_command(self.quiet_command)[1]
        return wait_for_idle_gpu(
            self.quiet_s, self.quiet_timeout_s, self.quiet_min_free_mb
        )[1]

    @contextlib.contextmanager
    def hold(self) -> Iterator[Held]:
        """Take the lock, then wait for an idle GPU (see :class:`Held`)."""
        with gpu_lock(self.lock) as handle:
            yield Held(self.wait(), () if handle is None else (handle,))


@contextlib.contextmanager
def stop_on_signals() -> Iterator[None]:
    """Turn SIGTERM and SIGHUP into :class:`Terminated` while the block runs.

    Their default action would end the driver at once and leave the run in
    progress running, unsupervised; as an exception they reach
    :func:`_child`, which kills the run's process group. A signal that is
    ignored (``nohup``) stays ignored. Only the main thread can install
    signal handlers; elsewhere this does nothing.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def stop(signum: int, frame: Any) -> None:
        raise Terminated(signum)

    previous: dict[int, Any] = {}
    for signum in (signal.SIGTERM, signal.SIGHUP):
        handler = signal.getsignal(signum)
        # None: a handler installed outside Python, which cannot be restored.
        if handler is signal.SIG_IGN or handler is None:
            continue
        previous[signum] = signal.signal(signum, stop)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _die_with_driver() -> None:
    """In a run that ``all`` started: die with the driver.

    Linux delivers SIGKILL to this process when the driver dies, however it
    dies (``prctl(PR_SET_PDEATHSIG)``), so that a run never measures, or
    writes a result, unsupervised. A driver that died before that took
    effect is caught by comparing the parent's process ID with the one the
    driver passed. Elsewhere only that check applies.
    """
    driver = os.environ.pop(DRIVER_ENV, None)
    if not driver:
        return
    with contextlib.suppress(Exception):
        import ctypes

        PR_SET_PDEATHSIG = 1
        ctypes.CDLL(None, use_errno=True).prctl(
            PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0
        )
    if str(os.getppid()) != driver:
        print("bench_compare.py: the driver has exited; not running", flush=True)
        os._exit(1)


def _as_run() -> None:
    """Set up the ``run`` and ``reference`` commands, which ``all`` starts."""
    _die_with_driver()
    import faulthandler

    # On SIGUSR1 (a run past its timeout) write every thread's stack to
    # stderr, the run's log. Not possible when stderr is not a file.
    with contextlib.suppress(AttributeError, OSError, ValueError):
        faulthandler.register(signal.SIGUSR1, all_threads=True)


# -------------------------------------------------------------- one run


def violations(
    metrics: Mapping[str, Any], bounds: Mapping[str, Sequence[Any]]
) -> list[str]:
    """Every limit in ``bounds`` that ``metrics`` miss (a missing metric misses)."""
    missed = []
    for name, (comparator, limit) in sorted(bounds.items()):
        actual = metrics.get(name)
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(actual)
        ):
            missed.append(f"{name} is unavailable (needs {comparator} {limit:g})")
        elif not _COMPARATORS[comparator](actual, limit):
            missed.append(f"{name}={actual:.6g} (needs {comparator} {limit:g})")
    return missed


def _values(workload: Any, config: Mapping[str, Any], edits_dir: str | None) -> dict:
    """Every knob's value for ``config``, its edited files as paths and its bench keys."""
    from torch_tensorrt.autotune._code_edits import EditedFileStore, is_edit_key
    from torch_tensorrt.autotune._knobs import knobs_by_name, resolve

    values = resolve(config["recipe"], knobs_by_name(workload.knobs()))
    declared = {file.key for file in workload.editable_files()}
    for key in [key for key in values if is_edit_key(key)]:
        if key not in declared:
            raise ValueError(
                f"The recipe sets {key}, which the workload does not declare"
            )
        if edits_dir is None:
            raise ValueError(
                f"The recipe sets {key} to a session's edited file: pass --session "
                "with the session that made it"
            )
        path = EditedFileStore(Path(edits_dir)).path(values[key])
        if not path.is_file():
            raise ValueError(f"{key}: the edited file {path} does not exist")
        values[key] = str(path)
    for key in declared:
        values.setdefault(key, None)
    values.update({f"bench.{key}": value for key, value in config["bench"].items()})
    return values


def measure(spec: Mapping[str, Any], config: str) -> dict[str, Any]:
    """Build, check and time one configuration of ``spec`` in this process."""
    import torch
    from torch_tensorrt.autotune import WorkloadFactory
    from torch_tensorrt.autotune._measure import measure_candidate
    from torch_tensorrt.autotune._worker import environment, load_reference
    from torch_tensorrt.region import generated_kernels, missing_kernels

    entry = spec["configs"][config]
    reference = Path(spec["reference"])
    result: dict[str, Any]
    values: dict[str, Any] | None = None
    bounds: dict[str, list[Any]] = {}
    registered, missing = len(generated_kernels()), len(missing_kernels())
    try:
        workload = WorkloadFactory(spec["workload"], spec["args"]).load()
        bounds = {
            name: list(bound) for name, bound in workload.accuracy_defaults().items()
        }
        bounds.update(spec.get("accuracy") or {})
        try:
            values = _values(workload, entry, spec.get("edits_dir"))
        except ValueError as error:
            result = {"status": "failed", "stage": "recipe", "error": str(error)}
        else:
            result = measure_candidate(
                workload,
                values,
                load_reference(reference),
                workload.timing(),
                profile=True,
                noise_control=spec["noise_control"],
            )
    except Exception as error:  # noqa: BLE001 - every failure becomes a result
        import traceback

        traceback.print_exc()
        message = "".join(traceback.format_exception_only(type(error), error))
        result = {"status": "failed", "stage": "setup", "error": message.strip()}
    result["config"] = config
    result["recipe"] = entry["recipe"]
    result["bench"] = entry["bench"]
    # The region kernels this build generated or loaded (each with its origin and
    # digests) and those it found no cached kernel for, as the Auto-Tuner's
    # worker records them.
    result["kernels"] = [
        kernel.to_json() for kernel in generated_kernels()[registered:]
    ]
    absent = [kernel.to_json() for kernel in missing_kernels()[missing:]]
    if absent:
        result["missing_kernels"] = absent
    if values is not None:
        result["values"] = values
    result["workload"] = {"factory": spec["workload"], "args": spec["args"]}
    result["reference"] = {
        "path": str(reference),
        "sha256": _sha256(reference) if reference.is_file() else None,
    }
    result["accuracy_bounds"] = bounds
    if result.get("status") == "ok":
        result["accuracy_violations"] = [
            f"{when}: {missed}"
            for when, key in (
                ("first call", "accuracy"),
                ("after timing", "accuracy_after_timing"),
            )
            for missed in violations(result.get(key) or {}, bounds)
        ]
    result["torch"] = torch.__version__
    with contextlib.suppress(Exception):
        result["environment"] = environment()
    with contextlib.suppress(Exception):
        from torch_tensorrt.autotune._worker import workload_files

        # The workload's own code: its file and the modules beside it.
        result["workload"]["files"] = workload_files(spec["workload"])
    with contextlib.suppress(Exception):
        from torch._dynamo.utils import counters

        result["dynamo"] = {
            "graph_breaks": sum(counters["graph_break"].values()),
            "unique_graphs": counters["stats"].get("unique_graphs", 0),
        }
    return result


def run(
    spec_path: str | Path,
    config: str,
    out: str | Path,
    *,
    reference: str | Path | None = None,
    session: str | Path | None = None,
) -> dict[str, Any]:
    """Measure ``config`` in this process and write its result to ``out``."""
    spec = load_spec(spec_path, reference=reference, session=session)
    if config not in spec["configs"]:
        raise SpecError(f"{spec_path} has no config {config!r}")
    if not Path(spec["reference"]).is_file():
        raise SpecError(
            f"The reference {spec['reference']} does not exist: compute it with "
            "`bench_compare.py reference`"
        )
    result = measure(spec, config)
    _write_json(Path(out), result)
    return result


def make_reference(
    spec_path: str | Path,
    out_dir: str | Path | None = None,
    *,
    reference: str | Path | None = None,
) -> dict[str, Any]:
    """Compute the workload's reference outputs in this process and save them."""
    from torch_tensorrt.autotune import WorkloadFactory
    from torch_tensorrt.autotune._worker import save_reference

    spec = load_spec(spec_path, out_dir=out_dir, reference=reference)
    path = Path(spec["reference"])
    start = time.perf_counter()
    workload = WorkloadFactory(spec["workload"], spec["args"]).load()
    partial = path.with_name(f".{path.name}.partial")
    description = save_reference(workload.reference(), partial)
    os.replace(partial, path)
    info = {
        "reference": str(path),
        "sha256": _sha256(path),
        "seconds": round(time.perf_counter() - start, 1),
        "description": description,
    }
    print(json.dumps(info, default=str))
    return info


# ------------------------------------------------------------- summaries


def _first_line(text: Any, size: int = 240) -> str:
    lines = str(text).strip().splitlines() or [""]
    line = lines[0]
    return line if len(line) <= size else line[: size - 3] + "..."


def failure_reason(result: Mapping[str, Any] | None) -> str:
    if result is None:
        return "no result (see the log)"
    return f"{result.get('stage', '?')}: {_first_line(result.get('error'))}"


def _timing_stability(result: Mapping[str, Any]) -> Mapping[str, Any]:
    """A run's timing-stability record; results written before the Auto-Tuner
    rename call it ``measurement``."""
    return result.get("timing_stability") or result.get("measurement") or {}


def run_line(result: Mapping[str, Any] | None) -> str:
    """One run's outcome: its latency quartiles and accuracy, or why it failed."""
    if result is None or result.get("status") != "ok":
        return f"FAILED at {failure_reason(result)}"
    latency = result["latency_ms"]
    text = (
        f"{latency['median']:.4f} ms/unit [{latency['p25']:.4f}, "
        f"{latency['p75']:.4f}]"
    )
    if not _timing_stability(result).get("stable", True):
        text += " UNSTABLE"
    missed = result.get("accuracy_violations")
    if missed:
        return f"{text} MISSES {'; '.join(missed)}"
    metrics = " ".join(
        f"{name}={value:.4g}"
        for name, value in sorted((result.get("accuracy") or {}).items())
        if isinstance(value, (int, float))
    )
    return f"{text} {metrics}".rstrip()


def _rounds(out_dir: Path) -> list[Path]:
    rounds = [
        path for path in out_dir.glob("r*") if path.is_dir() and path.name[1:].isdigit()
    ]
    return sorted(rounds, key=lambda path: int(path.name[1:]))


def _result(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def code_version(result: Mapping[str, Any] | None) -> str | None:
    """A digest of the code a run measured; None when its result does not say.

    It covers Torch-TensorRT (version, and the commit and uncommitted changes
    of a checkout), the workload's directory (``workload.files``) and
    PyTorch's version. Results the driver wrote, and older results, have
    none.
    """
    if not result or "environment" not in result:
        return None
    environment = result.get("environment") or {}
    code = {
        "torch_tensorrt": environment.get("torch_tensorrt"),
        "torch_tensorrt_source": environment.get("torch_tensorrt_source"),
        "torch": result.get("torch"),
        "workload_files": (result.get("workload") or {}).get("files"),
    }
    return hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()


def summarize(out_dir: str | Path, *, base: str | None = None) -> dict[str, Any]:
    """Aggregate every round of a head-to-head in ``out_dir``.

    Per configuration: its runs and how many succeeded, the median over runs
    of each run's median latency with the smallest and largest run, the
    accuracy limits any run missed (``None`` when the runs recorded no
    limits), its unstable runs and failures, and its ratio to ``base`` (by
    default ``trt-session`` if it ran, else the spec's first configuration
    that ran). ``references`` lists the reference outputs the runs used and
    ``code`` the runs of each version of the code they ran
    (:func:`code_version`); more than one of either makes the runs
    incomparable.
    """
    out_dir = Path(out_dir)
    spec = _result(out_dir / "spec.json")
    runs: dict[str, list[tuple[str, dict[str, Any] | None]]] = {}
    for round_dir in _rounds(out_dir):
        names = {path.stem for path in round_dir.glob("*.json")} | {
            path.stem for path in round_dir.glob("*.log")
        }
        for name in sorted(names):
            runs.setdefault(name, []).append(
                (round_dir.name, _result(round_dir / f"{name}.json"))
            )
    configs: dict[str, dict[str, Any]] = {}
    references: set[str] = set()
    code: dict[str, list[str]] = {}
    for name, entries in runs.items():
        ok = [
            (r, result)
            for r, result in entries
            if result and result.get("status") == "ok"
        ]
        medians = {r: result["latency_ms"]["median"] for r, result in ok}
        for r, result in entries:
            digest = ((result or {}).get("reference") or {}).get("sha256")
            if digest:
                references.add(digest)
            version = code_version(result)
            if version:
                code.setdefault(version, []).append(f"{r}/{name}")
        checked = bool(ok) and all("accuracy_violations" in result for _, result in ok)
        configs[name] = {
            "runs": len(entries),
            "ok": len(ok),
            "median_ms": statistics.median(medians.values()) if medians else None,
            "min_ms": min(medians.values()) if medians else None,
            "max_ms": max(medians.values()) if medians else None,
            "round_medians_ms": medians,
            "accuracy_violations": (
                sorted(
                    {
                        f"{r}: {v}"
                        for r, result in ok
                        for v in result["accuracy_violations"]
                    }
                )
                if checked
                else None
            ),
            "unstable_runs": [
                r
                for r, result in ok
                if not _timing_stability(result).get("stable", True)
            ],
            "failures": [
                f"{r}: {failure_reason(result)}"
                for r, result in entries
                if not result or result.get("status") != "ok"
            ],
            "build_s": (
                statistics.median(result.get("build_s", 0.0) for _, result in ok)
                if ok
                else None
            ),
        }
    order = list((spec or {}).get("configs") or {})
    if base is None:
        candidates = [SESSION_CONFIG, *order, *sorted(configs)]
        base = next(
            (n for n in candidates if (configs.get(n) or {}).get("median_ms")), None
        )
    reference_ms = (configs.get(base) or {}).get("median_ms") if base else None
    for entry in configs.values():
        entry["vs_base"] = (
            entry["median_ms"] / reference_ms
            if entry["median_ms"] is not None and reference_ms
            else None
        )
    ranked = sorted(
        configs.items(),
        key=lambda item: (item[1]["median_ms"] is None, item[1]["median_ms"] or 0.0),
    )
    return {
        "out_dir": str(out_dir),
        "rounds": len(_rounds(out_dir)),
        "base": base,
        "references": sorted(references),
        "code": {version: sorted(code[version]) for version in sorted(code)},
        "configs": dict(ranked),
    }


def format_summary(summary: Mapping[str, Any]) -> str:
    """The summary as a table, fastest first."""
    configs = summary["configs"]
    if not configs:
        return f"{summary['out_dir']}: no runs yet"
    width = max(len("config"), *(len(name) for name in configs))
    base = summary.get("base")
    lines = [
        f"{'config':{width}s}  {'median ms':>10s}  {'range of runs':>19s}  "
        f"{'ok':>5s}  {('vs ' + base) if base else '':>14s}  accuracy"
    ]
    for name, entry in configs.items():
        ok = f"{entry['ok']}/{entry['runs']}"
        if entry["median_ms"] is None:
            failures = entry["failures"][:1] or ["no result"]
            lines.append(
                f"{name:{width}s}  {'FAILED':>10s}  {'':>19s}  {ok:>5s}  {failures[0]}"
            )
            continue
        missed = entry["accuracy_violations"]
        accuracy = (
            "not checked"
            if missed is None
            else ("ok" if not missed else f"MISSES {missed[0]}")
        )
        notes = []
        if entry["unstable_runs"]:
            notes.append(f"unstable in {', '.join(entry['unstable_runs'])}")
        if entry["failures"]:
            notes.append(f"{len(entry['failures'])} failed")
        ratio = f"{entry['vs_base']:.3f}x" if entry["vs_base"] is not None else ""
        spread = f"{entry['min_ms']:.4f} - {entry['max_ms']:.4f}"
        lines.append(
            f"{name:{width}s}  {entry['median_ms']:10.4f}  {spread:>19s}  {ok:>5s}  "
            f"{ratio:>14s}  {accuracy}{'; ' + '; '.join(notes) if notes else ''}"
        )
    if len(summary.get("references") or []) > 1:
        lines.append(
            "WARNING: the runs used different reference outputs: "
            + ", ".join(digest[:12] for digest in summary["references"])
        )
    code = summary.get("code") or {}
    if len(code) > 1:
        lines.append(
            "WARNING: the runs ran different code (Torch-TensorRT, the workload's "
            "files or PyTorch; see each result's environment and workload): "
            + "; ".join(
                f"{version[:12]} in {', '.join(runs[:4])}"
                + (f" and {len(runs) - 4} more" if len(runs) > 4 else "")
                for version, runs in code.items()
            )
        )
    return "\n".join(lines)


def write_summary(
    out_dir: str | Path, *, base: str | None = None, write: bool = True
) -> str:
    """Summarize ``out_dir``, into ``summary.json`` if ``write``; return the table."""
    if not Path(out_dir).is_dir():
        raise SpecError(f"{out_dir} is not a directory")
    summary = summarize(out_dir, base=base)
    if base is not None and base not in summary["configs"]:
        raise SpecError(f"{out_dir} has no runs of {base!r}")
    if write:
        _write_json(Path(out_dir) / "summary.json", summary)
    return format_summary(summary)


# ------------------------------------------------------------- all runs


def _exit_note(code: int | None) -> str:
    if code is None:
        return "timeout"
    if code < 0:
        with contextlib.suppress(ValueError):
            return signal.Signals(-code).name
    return str(code)


def _kill_group(process: subprocess.Popen) -> None:
    """SIGKILL ``process``'s process group, then reap ``process``.

    Call it before ``process`` is reaped: until then its process ID, which is
    the group's ID, cannot be reused, so the signal reaches only the run's
    processes.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def _exited(process: subprocess.Popen) -> bool:
    """Whether ``process`` has exited, without reaping it where the OS allows."""
    if not hasattr(os, "waitid"):
        return process.poll() is not None
    try:
        state = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:  # already reaped
        return True
    return state is not None


def _wait(process: subprocess.Popen, timeout_s: float) -> bool:
    """Wait up to ``timeout_s`` for ``process`` to exit; whether it did."""
    deadline = time.monotonic() + timeout_s
    delay = 0.001
    while not _exited(process):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(delay, remaining))
        delay = min(2 * delay, 0.1)
    return True


def _child(
    command: Sequence[str],
    log: Path,
    env: Mapping[str, str],
    timeout_s: float,
    *,
    pass_fds: Sequence[int] = (),
) -> int | None:
    """Run ``command`` in its own process group; its exit status, None on timeout.

    The command inherits ``pass_fds`` (the GPU lock) and learns the driver's
    process ID (see :func:`_die_with_driver`). Past ``timeout_s`` it gets
    SIGUSR1 and :data:`STACKS_GRACE_S` seconds to write its stacks. Once it
    has exited or timed out, or this process is interrupted, its whole
    process group is killed: nothing a run started outlives it.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as stream:
        process = subprocess.Popen(
            list(command),
            env={**env, DRIVER_ENV: str(os.getpid())},
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=tuple(pass_fds),
        )
        try:
            finished = _wait(process, timeout_s)
            if not finished:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(process.pid, signal.SIGUSR1)
                _wait(process, STACKS_GRACE_S)
        finally:
            _kill_group(process)
        return process.returncode if finished else None


def _python(*args: str) -> list[str]:
    # -P: the script's directory must not shadow modules; -X faulthandler: a
    # crash leaves its Python stacks in the log.
    return [
        sys.executable,
        "-X",
        "faulthandler",
        "-P",
        str(Path(__file__).resolve()),
        *args,
    ]


def _log_tail(log: Path, lines: int = 20) -> list[str]:
    with contextlib.suppress(OSError):
        return log.read_text(errors="replace").strip().splitlines()[-lines:]
    return []


def _kept(path: Path) -> dict[str, Any] | None:
    """The result at ``path`` if ``--resume`` keeps it: one the run itself wrote."""
    result = _result(path)
    if result is None or result.get("written_by") == DRIVER_RECORD:
        return None
    return result


def run_all(
    spec_path: str | Path,
    out_dir: str | Path,
    configs: Sequence[str] = (),
    *,
    rounds: int = 4,
    gate: Gate | None = None,
    session: str | Path | None = None,
    reference: str | Path | None = None,
    resume: bool = False,
    keep_caches: bool = False,
    base: str | None = None,
) -> dict[str, Any]:
    """Run the head-to-head of ``spec_path`` into ``out_dir``; return its summary.

    SIGTERM and SIGHUP raise :class:`Terminated` meanwhile (see
    :func:`stop_on_signals`).
    """
    with stop_on_signals():
        return _run_all(
            spec_path,
            out_dir,
            configs,
            rounds=rounds,
            gate=gate,
            session=session,
            reference=reference,
            resume=resume,
            keep_caches=keep_caches,
            base=base,
        )


def _run_all(
    spec_path: str | Path,
    out_dir: str | Path,
    configs: Sequence[str],
    *,
    rounds: int,
    gate: Gate | None,
    session: str | Path | None,
    reference: str | Path | None,
    resume: bool,
    keep_caches: bool,
    base: str | None,
) -> dict[str, Any]:
    if rounds < 1:
        raise SpecError("rounds must be at least 1")
    gate = gate or Gate(None, quiet_s=0)
    out_dir = Path(out_dir).expanduser().resolve()
    spec = load_spec(spec_path, out_dir=out_dir, reference=reference, session=session)
    names = list(configs) or list(spec.get("head_to_head") or spec["configs"])
    unknown = [
        name
        for name in (*names, *([base] if base else []))
        if name not in spec["configs"]
    ]
    if unknown:
        raise SpecError(f"{spec_path} has no configs {unknown}")
    if len(set(names)) != len(names):
        raise SpecError(f"configs named twice: {names}")
    stored = out_dir / "spec.json"
    if stored.is_file():
        if not resume:
            raise SpecError(
                f"{out_dir} already holds a head-to-head ({stored}): pass --resume "
                "to continue it, or choose another directory"
            )
        changed = _differences(_read_json(stored), spec)
        if changed:
            raise SpecError(
                f"{out_dir} holds a head-to-head of another spec (it differs in "
                f"{', '.join(changed)}); choose another directory"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    digests = edit_digests(spec)
    if digests:
        # Keep the edited files with the results: OUT_DIR/edited_files/<sha>.py.
        edited = out_dir / "edited_files"
        for digest in sorted(digests):
            if not _DIGEST.fullmatch(digest) or not spec.get("edits_dir"):
                raise SpecError(
                    f"A recipe uses the edited file {digest!r}: pass --session with "
                    "the session that made it"
                )
            source = Path(spec["edits_dir"]) / "edited_files" / f"{digest}.py"
            if source.resolve() != (edited / source.name).resolve():
                if not source.is_file():
                    raise SpecError(f"The edited file {source} does not exist")
                edited.mkdir(exist_ok=True)
                shutil.copyfile(source, edited / source.name)
        spec["edits_dir"] = str(out_dir)
    spec["source"] = {
        "path": str(Path(spec_path).resolve()),
        "sha256": _sha256(Path(spec_path)),
    }
    _write_json(stored, spec)
    environment = dict(os.environ)
    if not Path(spec["reference"]).is_file():
        log = out_dir / "reference.log"
        with gate.hold() as held:
            start = time.monotonic()
            code = _child(
                _python("reference", str(stored)),
                log,
                environment,
                spec["timeout_s"],
                pass_fds=held.fds,
            )
        if code != 0 or not Path(spec["reference"]).is_file():
            raise SpecError(
                f"Computing the reference failed (exit {_exit_note(code)}); see {log}"
            )
        print(
            f"reference {spec['reference']} {time.monotonic() - start:6.0f}s  "
            f"[{held.quiet}]",
            flush=True,
        )
    environment.setdefault("HF_HUB_OFFLINE", "1")
    for index in range(rounds):
        round_dir = out_dir / f"r{index + 1}"
        order = round_order(names, index)
        print(f"r{index + 1} order: {' '.join(order)}", flush=True)
        for name in order:
            out, log = round_dir / f"{name}.json", round_dir / f"{name}.log"
            kept = _kept(out) if resume else None
            if kept is not None:
                print(f"r{index + 1} {name:42s} kept      {run_line(kept)}", flush=True)
                continue
            out.unlink(missing_ok=True)
            scratch = round_dir / "cache" / name
            env = {
                **environment,
                "TORCHINDUCTOR_CACHE_DIR": str(scratch / "inductor"),
                "TRITON_CACHE_DIR": str(scratch / "triton"),
                "TMPDIR": str(scratch / "tmp"),
            }
            try:
                with gate.hold() as held:
                    # Cold: nothing of an earlier attempt (a stopped driver's,
                    # or one with --keep-caches) may stay in the caches.
                    if scratch.exists():
                        shutil.rmtree(scratch)
                    (scratch / "tmp").mkdir(parents=True)
                    start = time.monotonic()
                    code = _child(
                        _python("run", str(stored), name, str(out)),
                        log,
                        env,
                        spec["timeout_s"],
                        pass_fds=held.fds,
                    )
                    elapsed = time.monotonic() - start
            finally:
                if not keep_caches:
                    shutil.rmtree(scratch, ignore_errors=True)
            result = _result(out)
            if result is None:
                # The process died or was killed before it wrote a result.
                result = {
                    "status": "failed",
                    "stage": "timeout" if code is None else "crash",
                    "error": (
                        f"killed after timeout_s={spec['timeout_s']:g}"
                        if code is None
                        else f"exited with status {_exit_note(code)} and no result"
                    ),
                    "config": name,
                    "log_tail": _log_tail(log),
                    # --resume runs it again.
                    "written_by": DRIVER_RECORD,
                }
                _write_json(out, result)
            print(
                f"r{index + 1} {name:42s} exit={_exit_note(code):7s} {elapsed:6.0f}s  "
                f"{run_line(result)}   [{held.quiet}]",
                flush=True,
            )
        with contextlib.suppress(OSError):
            (round_dir / "cache").rmdir()
    summary = summarize(out_dir, base=base)
    _write_json(out_dir / "summary.json", summary)
    print(format_summary(summary), flush=True)
    return summary


# ------------------------------------------------------------------- CLI


def _gate_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--quiet-command",
        default=os.environ.get(QUIET_ENV) or None,
        help=f"command that returns once the GPU is idle (default ${QUIET_ENV}; "
        "otherwise the built-in wait)",
    )
    parser.add_argument(
        "--quiet-s",
        type=int,
        default=10,
        help="consecutive idle one-second samples to wait for; 0 turns it off",
    )
    parser.add_argument("--quiet-timeout-s", type=float, default=3600.0)
    parser.add_argument(
        "--quiet-min-free-mb",
        type=float,
        default=0.0,
        help="free GPU memory an idle sample needs",
    )


def _lock_path(value: str | None) -> Path | None:
    if value is None:
        value = os.environ.get(LOCK_ENV) or str(DEFAULT_LOCK)
    return None if value.lower() in ("", "none") else Path(value).expanduser()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench_compare.py",
        description=__doc__.split("\n\n")[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    reference = commands.add_parser("reference", help="compute the reference outputs")
    reference.add_argument("spec")
    reference.add_argument("out_dir", nargs="?")
    reference.add_argument("--reference", help="where to save them instead")

    one = commands.add_parser("run", help="measure one config in this process")
    one.add_argument("spec")
    one.add_argument("config")
    one.add_argument("out")
    one.add_argument("--reference", help="the reference outputs (overrides the spec)")
    one.add_argument("--session", help="take trt-session from this Auto-Tuner session")

    every = commands.add_parser("all", help="the head-to-head, one process per run")
    every.add_argument("spec")
    every.add_argument("out_dir")
    every.add_argument(
        "configs",
        nargs="*",
        help="default: the spec's head_to_head (also accepted after the options)",
    )
    every.add_argument("--rounds", type=int, default=4)
    every.add_argument(
        "--session", help="take trt-session from this Auto-Tuner session"
    )
    every.add_argument("--reference", help="the reference outputs (overrides the spec)")
    every.add_argument(
        "--lock",
        help=f"lock file held around each run (default ${LOCK_ENV}, else {DEFAULT_LOCK}; "
        "'none' turns it off)",
    )
    every.add_argument(
        "--resume",
        action="store_true",
        help="continue OUT_DIR: keep the runs that left a result, run the others",
    )
    every.add_argument("--keep-caches", action="store_true")
    every.add_argument("--base", help="the config the summary compares with")
    _gate_options(every)

    summary = commands.add_parser("summary", help="summarize a head-to-head")
    summary.add_argument("out_dir")
    summary.add_argument("--base", help="the config to compare with")
    summary.add_argument(
        "--write", action="store_true", help="also rewrite OUT_DIR/summary.json"
    )

    quiet = commands.add_parser("quiet", help="wait until the GPU is idle")
    _gate_options(quiet)

    # parse_known_args: argparse leaves config names given after an option
    # of ``all`` (``all SPEC OUT --rounds 4 a b``) unparsed.
    args, extra = parser.parse_known_args(argv)
    if args.command == "all":
        args.configs = [*args.configs, *(a for a in extra if not a.startswith("-"))]
        extra = [a for a in extra if a.startswith("-")]
    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    try:
        if args.command in ("reference", "run"):
            _as_run()
        if args.command == "reference":
            make_reference(args.spec, args.out_dir, reference=args.reference)
        elif args.command == "run":
            run(
                args.spec,
                args.config,
                args.out,
                reference=args.reference,
                session=args.session,
            )
        elif args.command == "all":
            gate = Gate(
                _lock_path(args.lock),
                args.quiet_command,
                args.quiet_s,
                args.quiet_timeout_s,
                args.quiet_min_free_mb,
            )
            run_all(
                args.spec,
                args.out_dir,
                args.configs,
                rounds=args.rounds,
                gate=gate,
                session=args.session,
                reference=args.reference,
                resume=args.resume,
                keep_caches=args.keep_caches,
                base=args.base,
            )
        elif args.command == "summary":
            print(write_summary(args.out_dir, base=args.base, write=args.write))
        else:
            if args.quiet_command:
                idle, message = run_quiet_command(args.quiet_command)
            else:
                idle, message = wait_for_idle_gpu(
                    args.quiet_s, args.quiet_timeout_s, args.quiet_min_free_mb
                )
            print(message)
            return 0 if idle else 1
    except SpecError as error:
        print(f"bench_compare.py: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("bench_compare.py: interrupted", file=sys.stderr)
        return 130
    except Terminated as stop:
        name = signal.Signals(stop.signum).name
        print(f"bench_compare.py: stopped by {name}", file=sys.stderr)
        return 128 + stop.signum
    return 0


if __name__ == "__main__":
    sys.exit(main())
