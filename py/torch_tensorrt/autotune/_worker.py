# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""The worker process: ``python -m torch_tensorrt.autotune._worker REQUEST.json``.

The session runs every build and measurement as a job in a fresh worker
process, so candidates cannot leak GPU memory, process-global registrations
or crashes into each other or into the session. A worker rebuilds the
workload from its factory and runs one job (see :func:`run_job`):

* ``reference``: describe the workload and save its reference outputs (the
  first run of a session);
* ``describe``: describe the workload again (a resumed session compares the
  description and the environment with the recorded ones);
* ``prepare``: build one config without timing it, before the search, so
  that the region kernels its build generates are ready for later builds;
* ``measure``: build, check, time and profile one config (and, for the
  export at the end of a session, save it).

A ``prepare`` or ``measure`` result lists the region kernels its build
registered (``kernels``, see :func:`torch_tensorrt.region.generated_kernels`).
A request with ``"allow_generation": false`` runs its build with kernel
generation turned off (``TORCH_TENSORRT_REGION_GENERATION=0``), so that a
kernel missing from the cache fails the build instead of starting a search;
the result then lists the regions whose kernel was missing
(``missing_kernels``, see :func:`torch_tensorrt.region.missing_kernels`).

It writes its result to the request's ``result_path``. A worker that crashes
leaves its Python stacks in its log (``faulthandler``), and on ``SIGUSR1``,
which the session sends before it kills a worker that ran out of time, it
writes them there and carries on.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import torch
import torch.utils._pytree as pytree

if TYPE_CHECKING:
    from torch_tensorrt.autotune._settings import WorkloadFactory


def write_json(path: Path, value: Any) -> None:
    """Write ``value`` atomically, so a reader never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def save_reference(tree: Any, path: Path) -> None:
    """Save reference outputs (tensors, numbers and containers of them) on the CPU."""
    leaves, _ = pytree.tree_flatten(tree)
    for leaf in leaves:
        if not isinstance(leaf, (torch.Tensor, int, float, bool)):
            raise TypeError(
                "Reference outputs must be tensors, numbers and containers; "
                f"got {type(leaf).__name__}"
            )
    on_cpu = pytree.tree_map(
        lambda leaf: leaf.detach().cpu() if isinstance(leaf, torch.Tensor) else leaf,
        tree,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(on_cpu, path)


def load_reference(path: Path) -> Any:
    """Load the reference outputs saved by :func:`save_reference`."""
    # weights_only: the file holds tensors and containers, never code.
    return torch.load(path, weights_only=True)


# A git query that takes longer fails (the source then cannot be pinned).
_GIT_TIMEOUT_S = 60


class GitError(RuntimeError):
    """A git query failed; the message says why (git's own error line if it gave one).

    ``no_git`` is set when git is not installed, ``no_repository`` when git
    found no repository at or above the directory.
    """

    def __init__(
        self, message: str, *, no_git: bool = False, no_repository: bool = False
    ) -> None:
        super().__init__(message)
        self.no_git = no_git
        self.no_repository = no_repository


def _git(directory: Path, *args: str) -> bytes:
    """The output of ``git -C directory args``; raises :class:`GitError` if git fails.

    The queries only read: no fsmonitor hook, no optional index writes, and
    the C locale, so git's messages do not depend on the user's language.
    They pass ``safe.directory=*`` because git refuses a checkout that
    another user owns ("dubious ownership"), and the worker already imports
    and runs the code of the checkout it asks about.
    """
    command = [
        "git",
        "-C",
        str(directory),
        "-c",
        "core.fsmonitor=false",
        "-c",
        "safe.directory=*",
        *args,
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=_GIT_TIMEOUT_S,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C"},
        )
    except FileNotFoundError:
        raise GitError("git is not installed", no_git=True) from None
    except subprocess.TimeoutExpired:
        raise GitError(f"git {args[0]} timed out after {_GIT_TIMEOUT_S} s") from None
    except OSError as error:
        raise GitError(f"git could not run: {error}") from None
    if completed.returncode:
        lines = [
            line.strip()
            for line in completed.stderr.decode(errors="replace").splitlines()
            if line.strip()
        ]
        message = next(
            (line for line in lines if line.startswith(("fatal:", "error:"))),
            lines[0] if lines else f"exit status {completed.returncode}",
        )
        if len(message) > 300:
            message = message[:297] + "..."
        raise GitError(
            f"git {args[0]} failed: {message}",
            no_repository="not a git repository" in message,
        )
    return completed.stdout


def _below_a_git_directory(path: Path) -> bool:
    """Whether ``path`` or a directory above it holds a ``.git`` entry."""
    for parent in (path, *path.parents):
        try:
            if (parent / ".git").exists():
                return True
        except OSError:
            continue
    return False


def source_provenance(package: Path) -> dict[str, Any] | None:
    """The git commit of ``package``'s checkout and a digest of its uncommitted changes.

    ``commit`` is the checkout's ``HEAD``. ``uncommitted_sha256`` hashes
    ``git diff HEAD`` of the package directory and the untracked files in it
    (None when there are none). Returns None for a package that no git
    checkout tracks, such as an installed wheel (even one inside another
    project's repository), which its version identifies. When the package is
    in a checkout that git cannot read (git is missing, the repository is
    damaged, a query timed out), returns ``{"unavailable": <why>}`` with
    git's error line instead, so the session can say why it cannot pin the
    source.
    """
    try:
        commit = _git(package, "rev-parse", "--verify", "HEAD")
        if not _git(package, "ls-files", "-z", "--", "."):
            return None
        diff = _git(
            package,
            "diff",
            "--no-color",
            "--no-ext-diff",
            "--no-textconv",
            "--binary",
            "HEAD",
            "--",
            ".",
        )
        untracked = _git(
            package, "ls-files", "-z", "--others", "--exclude-standard", "--", "."
        )
    except GitError as error:
        if error.no_repository or (
            error.no_git and not _below_a_git_directory(package)
        ):
            return None
        return {"unavailable": str(error)}
    sha = hashlib.sha256(diff)
    names = sorted(
        name for name in untracked.decode(errors="replace").split("\0") if name
    )
    for name in names:
        sha.update(b"\0untracked\0" + name.encode() + b"\0")
        try:
            sha.update((package / name).read_bytes())
        except OSError:
            sha.update(b"<unreadable>")
    return {
        "commit": commit.decode(errors="replace").strip(),
        "uncommitted_sha256": sha.hexdigest() if diff or names else None,
    }


def workload_files(factory: str) -> dict[str, str]:
    """SHA-256 of the workload factory's file and of the other ``.py`` modules beside it.

    For a ``module:function`` factory the file is the imported module's (call
    this after :meth:`WorkloadFactory.load`); an empty mapping if it has none.
    """
    module_name = factory.rpartition(":")[0]
    if module_name.endswith(".py"):
        path: Path | None = Path(module_name)
    else:
        module = sys.modules.get(module_name)
        location = getattr(module, "__file__", None)
        path = Path(location) if location else None
    if path is None:
        return {}
    files = sorted(path.resolve().parent.glob("*.py"))
    return {
        str(file): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in files
        if file.is_file()
    }


def environment() -> dict[str, Any]:
    """The software and hardware that every measurement in a session depends on.

    ``torch_tensorrt_source`` identifies the Torch-TensorRT code itself (see
    :func:`source_provenance`), since a development build keeps one version
    string across commits and edits; ``{"unavailable": <why>}`` when its
    checkout cannot be read.
    """
    import platform
    import shutil

    import tensorrt
    import torch_tensorrt

    driver = None
    binary = shutil.which("nvidia-smi")
    if binary is not None:
        try:
            driver = subprocess.run(
                [
                    binary,
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                    "-i",
                    "0",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            driver = None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tensorrt": tensorrt.__version__,
        "torch_tensorrt": torch_tensorrt.__version__,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "driver": driver,
        "torch_tensorrt_source": source_provenance(
            Path(torch_tensorrt.__file__).resolve().parent
        ),
    }


def describe_workload(workload: Any) -> dict[str, Any]:
    """What the workload declares, which the session records and checks on resume.

    That is its name, knobs, baseline config, timing, accuracy defaults and
    editable files. The session records this description (with the SHA-256
    of the workload's source files) and refuses to resume when it changes.
    """
    from dataclasses import asdict

    from torch_tensorrt.autotune._knobs import apply_changes, knobs_by_name

    knobs = knobs_by_name(workload.knobs())
    return {
        "name": workload.name,
        "knobs": [knob.to_json() for knob in knobs.values()],
        "baseline": apply_changes(workload.baseline(), {}, knobs),
        "timing": asdict(workload.timing()),
        "accuracy_defaults": {
            name: list(bound) for name, bound in workload.accuracy_defaults().items()
        },
        "editable_files": [
            editable.to_json() for editable in workload.editable_files()
        ],
    }


# The environment variable that turns region kernel generation off for every
# KernelGenerator, as KernelGenerator(allow_generation=False) does for one.
GENERATION_SWITCH = "TORCH_TENSORRT_REGION_GENERATION"


@contextlib.contextmanager
def generation_allowed(allowed: bool) -> Iterator[None]:
    """Run the block with region kernel generation allowed or not.

    ``False`` sets ``TORCH_TENSORRT_REGION_GENERATION=0`` for the block, so
    that every :class:`~torch_tensorrt.region.KernelGenerator` only reuses
    cached kernels, and restores the variable afterwards. ``True`` leaves the
    environment as it is.
    """
    if allowed:
        yield
        return
    previous = os.environ.get(GENERATION_SWITCH)
    os.environ[GENERATION_SWITCH] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(GENERATION_SWITCH, None)
        else:
            os.environ[GENERATION_SWITCH] = previous


def run_job(request: dict[str, Any]) -> dict[str, Any]:
    """Run one job and return its result: ``reference``, ``describe``, ``prepare`` or ``measure``.

    Any kind other than ``reference``, ``describe`` and ``prepare`` measures
    the request's ``config``. The config's ``edit.<name>`` digests become the
    paths of the stored edited files, and an editable file the config does
    not edit gets ``None``. A ``prepare`` or ``measure`` result lists the
    region kernels registered during the job (``kernels``) and, if there
    were any, the regions whose kernel was missing (``missing_kernels``); its
    build runs without kernel generation when the request has
    ``"allow_generation": false``.
    """
    from torch_tensorrt.autotune._settings import WorkloadFactory

    factory = WorkloadFactory(**request["workload_factory"])
    result: dict[str, Any]
    if request["kind"] in ("describe", "reference"):
        workload = factory.load()
        result = {
            "status": "ok",
            "workload": {
                **describe_workload(workload),
                "workload_files": workload_files(factory.factory),
            },
            "environment": environment(),
        }
        if request["kind"] == "reference":
            save_reference(workload.reference(), Path(request["reference_path"]))
        return result
    from torch_tensorrt.region import generated_kernels, missing_kernels

    registered, missing = len(generated_kernels()), len(missing_kernels())
    with generation_allowed(bool(request.get("allow_generation", True))):
        result = _build_job(request, factory)
    result["kernels"] = [
        kernel.to_json() for kernel in generated_kernels()[registered:]
    ]
    absent = [kernel.to_json() for kernel in missing_kernels()[missing:]]
    if absent:
        result["missing_kernels"] = absent
    return result


def _build_job(request: dict[str, Any], factory: WorkloadFactory) -> dict[str, Any]:
    """Build the request's config: prepare it, or measure it."""
    from torch_tensorrt.autotune._code_edits import EditedFileStore, is_edit_key
    from torch_tensorrt.autotune._knobs import knobs_by_name, resolve
    from torch_tensorrt.autotune._measure import measure_candidate, prepare_candidate
    from torch_tensorrt.autotune._settings import Timing

    workload = factory.load()
    knobs = knobs_by_name(workload.knobs())
    values = resolve(request["config"], knobs)
    editable_files = {editable.key: editable for editable in workload.editable_files()}
    store = EditedFileStore(Path(request["session_dir"]))
    for key in [key for key in values if is_edit_key(key)]:
        if key not in editable_files:
            raise ValueError(
                f"The config edits {key}, which the workload does not declare"
            )
        values[key] = str(store.path(values[key]))
    for key in editable_files:
        values.setdefault(key, None)
    if request["kind"] == "prepare":
        return prepare_candidate(workload, values)
    export_dir = request.get("export_dir")
    return measure_candidate(
        workload,
        values,
        load_reference(Path(request["reference_path"])),
        Timing(**request["timing"]),
        profile=bool(request.get("profile", True)),
        export_dir=Path(export_dir) if export_dir else None,
        noise_control=request.get("noise_control"),
    )


def _install_faulthandler() -> None:
    """Make faulthandler the active handler for fatal signals and SIGUSR1."""
    import faulthandler
    import signal

    if faulthandler.is_enabled():
        faulthandler.disable()
    faulthandler.enable(all_threads=True)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, all_threads=True)


def main(argv: list[str]) -> int:
    """Run the job in ``argv[1]``; write a result even if the job cannot start."""
    if len(argv) != 2:
        print(
            "usage: python -m torch_tensorrt.autotune._worker REQUEST.json",
            file=sys.stderr,
        )
        return 2
    # A fatal signal (a segfault in a native library, an abort) writes every
    # thread's Python stack to the log before the worker dies; SIGUSR1 writes
    # them and carries on. The session starts workers with -X faulthandler,
    # which also covers the imports above; this covers a worker run by hand.
    _install_faulthandler()
    request = json.loads(Path(argv[1]).read_text())
    result_path = Path(request["result_path"])
    try:
        # Native libraries loaded by these imports (TensorRT, Triton and
        # whatever else the environment brings) may install their own fatal
        # signal handlers over faulthandler's: install it again, last, so a
        # crash still leaves the Python stacks in the log. faulthandler calls
        # the handler it replaced after writing them.
        import torch_tensorrt  # noqa: F401

        _install_faulthandler()
        result = run_job(request)
    except Exception as error:  # noqa: BLE001 - report, do not hide, setup failures
        import traceback

        traceback.print_exc()
        result = {
            "status": "failed",
            "stage": "setup",
            "error": "".join(
                traceback.format_exception_only(type(error), error)
            ).strip(),
        }
    write_json(result_path, result)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
