# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Atomic local evidence storage and provenance for compiler workflows.

Importing this module uses only the standard library and reporting contracts.
Torch is imported lazily by environment/weight discovery, never by JSON or file APIs.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib.metadata
import os
import platform
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, Literal, Mapping

from ..reporting import (
    ArtifactEntry,
    ArtifactManifest,
    Diagnostic,
    EnvironmentReport,
    Failure,
    FailureCategory,
    GPUInfo,
    ReportStatus,
    Stage,
    classify_exception,
    status_for_failure,
    to_json,
)


def fingerprint(value: Any) -> str:
    """Hash canonical finite JSON, rejecting runtime objects or non-string keys."""
    return hashlib.sha256(to_json(value).encode("utf-8")).hexdigest()


def hash_file(path: str | Path) -> str:
    """Hash a regular file without following a final-component symlink."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.fspath(path), flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Artifacts must be regular files")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        after = os.fstat(descriptor)
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Artifact changed while being hashed")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def hash_graph(graph: Any) -> str:
    """Hash FX graph structure using names, targets, and literal arguments.

    Tensor state is fingerprinted separately by ``hash_weights``. Source locations,
    fake-tensor identities, and process-specific object representations are excluded.
    """

    def target_name(value: Any) -> str:
        if isinstance(value, str):
            return value
        module = getattr(value, "__module__", "")
        name = getattr(value, "__qualname__", None)
        return f"{module}.{name}" if name else str(value)

    def argument(value: Any) -> Any:
        if hasattr(value, "op") and hasattr(value, "name"):
            return {"node": value.name}
        if isinstance(value, (tuple, list)):
            return [argument(item) for item in value]
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise ValueError("Graph argument keys must be strings")
            return {key: argument(item) for key, item in value.items()}
        if isinstance(value, slice):
            return {
                "slice": [
                    argument(value.start),
                    argument(value.stop),
                    argument(value.step),
                ]
            }
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        # Torch dtypes/devices are stable strings; arbitrary runtime objects are not.
        if type(value).__module__ == "torch" and type(value).__name__ in (
            "dtype",
            "device",
            "layout",
            "memory_format",
        ):
            return str(value)
        raise ValueError(f"Unsupported graph literal type: {type(value).__name__}")

    def describe_graph(value: Any) -> list[dict[str, Any]]:
        return [
            {
                "name": node.name,
                "op": node.op,
                "target": target_name(node.target),
                "args": argument(node.args),
                "kwargs": argument(node.kwargs),
            }
            for node in value.nodes
        ]

    source_module = graph if hasattr(graph, "graph") else None
    graph = getattr(graph, "graph", graph)
    nodes = describe_graph(graph)
    nested = {}
    if source_module is not None:
        nested = {
            name: describe_graph(module.graph)
            for name, module in source_module.named_modules()
            if name and hasattr(module, "graph")
        }
    return fingerprint({"nodes": nodes, "subgraphs": nested})


def hash_weights(state: Mapping[str, Any]) -> str:
    """Hash ordered tensor names, shapes, dtypes, and raw bytes, independent of device."""
    import torch

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.layout != torch.strided
            or tensor.is_quantized
        ):
            raise ValueError("Weight fingerprints require dense, non-quantized tensors")
        header = to_json(
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        data = (
            tensor.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def hash_inputs(args: Any, kwargs: Mapping[str, Any] | None = None) -> str:
    """Hash nested input data, structure, strides, and tensor alias relationships.

    Container types and insertion order are preserved because a model can observe
    them. Runtime storage addresses are replaced by encounter-order alias IDs.
    Custom input classes are rejected rather than fingerprinted by repr or pickle.
    """
    import torch

    storages: dict[Any, int] = {}

    def describe(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided or value.is_quantized:
                raise ValueError(
                    "Input fingerprints require dense, non-quantized tensors"
                )
            storage = value.untyped_storage()
            key = (str(value.device), storage._cdata)
            alias = storages.setdefault(key, len(storages))
            return {
                "kind": "tensor",
                "sha256": hash_weights({"value": value}),
                "alias": alias,
                "stride": list(value.stride()),
                "storage_offset": value.storage_offset(),
                "requires_grad": value.requires_grad,
            }
        if isinstance(value, tuple):
            return {"kind": "tuple", "items": [describe(item) for item in value]}
        if isinstance(value, list):
            return {"kind": "list", "items": [describe(item) for item in value]}
        if isinstance(value, dict):
            if not all(isinstance(key, str) for key in value):
                raise ValueError("Input dictionary keys must be strings")
            return {
                "kind": "dict",
                "items": [[key, describe(item)] for key, item in value.items()],
            }
        if value is None or isinstance(value, (str, bool, int, float)):
            return {"kind": type(value).__name__, "value": value}
        raise ValueError(f"Unsupported input fingerprint type: {type(value).__name__}")

    return fingerprint({"args": describe(args), "kwargs": describe(dict(kwargs or {}))})


def _version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return None


def collect_environment(job_id: str = "") -> EnvironmentReport:
    """Collect installed versions and readable GPU provenance without running a model."""
    unavailable: dict[str, str] = {}
    diagnostics: list[Diagnostic] = []
    torch_version = _version("torch")
    tensorrt_version = _version(
        "tensorrt", "tensorrt-cu12", "tensorrt-cu13", "tensorrt-rtx"
    )
    trt_version = _version("torch-tensorrt", "torch_tensorrt")
    runtime_version = None
    driver_version = None
    build_version = None
    nvidia_driver_version = None
    cuda_available = False
    gpus: list[GPUInfo] = []
    features: dict[str, Any] = {}
    try:
        import torch

        torch_version = str(torch.__version__)
        build_version = torch.version.cuda
        features["torch_cuda_build_version_source"] = "torch.version.cuda"
        cuda_available = torch.cuda.is_available()
        features["hip_version"] = getattr(torch.version, "hip", None)
        if cuda_available:
            try:
                get_runtime_version = getattr(
                    torch.cuda.cudart(), "cudaRuntimeGetVersion"
                )
                runtime_result = get_runtime_version()
                if isinstance(runtime_result, tuple) and len(runtime_result) == 2:
                    status, runtime_result = runtime_result
                    if status != 0:
                        raise RuntimeError("cudaRuntimeGetVersion failed")
                if type(runtime_result) is not int or runtime_result <= 0:
                    raise TypeError("Unexpected CUDA runtime version result")
                runtime_version = (
                    f"{runtime_result // 1000}.{(runtime_result % 1000) // 10}"
                )
                features["cuda_runtime_version_source"] = (
                    "torch.cuda.cudart.cudaRuntimeGetVersion"
                )
            except Exception as exc:
                unavailable["cuda_runtime_version"] = (
                    f"Runtime API query unavailable: {type(exc).__name__}"
                )
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                uuid = getattr(props, "uuid", None)
                gpus.append(
                    GPUInfo(
                        name=props.name,
                        uuid=str(uuid) if uuid else None,
                        compute_capability=f"{props.major}.{props.minor}",
                        total_memory_bytes=props.total_memory,
                        index=index,
                    )
                )
            try:
                library = ctypes.CDLL("libcuda.so.1")
                driver_api_version = ctypes.c_int()
                library.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
                library.cuDriverGetVersion.restype = ctypes.c_int
                if library.cuDriverGetVersion(ctypes.byref(driver_api_version)) == 0:
                    driver_version = f"{driver_api_version.value // 1000}.{(driver_api_version.value % 1000) // 10}"
                    features["cuda_driver_version_source"] = (
                        "libcuda.cuDriverGetVersion (CUDA driver API version)"
                    )
            except (OSError, AttributeError):
                pass
            try:
                import pynvml

                pynvml.nvmlInit()
                try:
                    version = pynvml.nvmlSystemGetDriverVersion()
                    nvidia_driver_version = (
                        version.decode("ascii")
                        if isinstance(version, bytes)
                        else str(version)
                    )
                    features["nvidia_driver_version_source"] = (
                        "pynvml.nvmlSystemGetDriverVersion"
                    )
                finally:
                    pynvml.nvmlShutdown()
            except Exception as exc:
                unavailable["nvidia_driver_version"] = (
                    f"Optional NVML query unavailable: {type(exc).__name__}"
                )
    except Exception as exc:
        unavailable["torch_runtime"] = type(exc).__name__
        diagnostics.append(
            Diagnostic(
                code="environment_probe_failed",
                message="Torch runtime discovery failed.",
            )
        )
    for name, value in (
        ("torch_tensorrt_version", trt_version),
        ("tensorrt_version", tensorrt_version),
        ("torch_cuda_build_version", build_version),
        ("cuda_runtime_version", runtime_version),
        ("cuda_driver_version", driver_version),
        ("nvidia_driver_version", nvidia_driver_version),
    ):
        if value is None:
            unavailable.setdefault(
                name, "not installed or not available from the runtime"
            )
    if not cuda_available:
        unavailable["gpus"] = "CUDA is unavailable"
    return EnvironmentReport(
        job_id=job_id,
        python_version=platform.python_version(),
        platform=platform.platform(),
        torch_version=torch_version,
        torch_tensorrt_version=trt_version,
        tensorrt_version=tensorrt_version,
        torch_cuda_build_version=build_version,
        cuda_runtime_version=runtime_version,
        cuda_driver_version=driver_version,
        nvidia_driver_version=nvidia_driver_version,
        cuda_available=cuda_available,
        gpus=gpus,
        features=features,
        unavailable=unavailable,
        diagnostics=diagnostics,
    )


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Replace one JSON file atomically after syncing its canonical contents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("JSON destinations must not be symlinks")
    content = to_json(value) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sync_directory(path: Path) -> None:
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def build_manifest(
    root: str | Path,
    job_id: str,
    request_fingerprint: str = "",
    environment: EnvironmentReport | None = None,
    lineage: dict[str, Any] | None = None,
    absent_stage_reasons: dict[str, str] | None = None,
    status: ReportStatus | str = ReportStatus.SUCCEEDED,
    failure: Failure | None = None,
) -> ArtifactManifest:
    root = Path(root).resolve(strict=True)
    entries: list[ArtifactEntry] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Manifest contents must not contain symlinks")
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == "manifest.json":
            continue
        if not path.resolve(strict=True).is_relative_to(root):
            raise ValueError("Artifact escaped the job directory")
        kind = (
            "report"
            if relative.endswith(".json")
            else "events" if relative.endswith(".jsonl") else "file"
        )
        entries.append(
            ArtifactEntry(
                artifact_id=relative,
                path=relative,
                sha256=hash_file(path),
                size_bytes=path.stat().st_size,
                kind=kind,
            )
        )
    return ArtifactManifest(
        job_id=job_id,
        entries=entries,
        environment=environment,
        request_fingerprint=request_fingerprint,
        lineage=lineage or {},
        absent_stage_reasons=absent_stage_reasons or {},
        status=ReportStatus(status),
        failure=failure,
    )


def verify_manifest(manifest: ArtifactManifest, root: str | Path) -> list[str]:
    """Return integrity problems; no manifest path may read outside the supplied root."""
    manifest.to_dict()  # Reject malformed paths even if manually constructed.
    root = Path(root).resolve(strict=True)
    problems: list[str] = []
    for entry in manifest.entries:
        path = root / entry.path
        # Check every component so a symlink inside the root is rejected too.
        cursor = root
        for component in Path(entry.path).parts:
            cursor = cursor / component
            if cursor.is_symlink():
                problems.append(f"{entry.artifact_id}: symlink")
                break
        else:
            try:
                resolved = path.resolve(strict=True)
                if not resolved.is_relative_to(root):
                    problems.append(f"{entry.artifact_id}: outside root")
                elif not resolved.is_file():
                    problems.append(f"{entry.artifact_id}: not a regular file")
                elif resolved.stat().st_size != entry.size_bytes:
                    problems.append(f"{entry.artifact_id}: size mismatch")
                elif hash_file(resolved) != entry.sha256:
                    problems.append(f"{entry.artifact_id}: hash mismatch")
            except (OSError, ValueError):
                problems.append(f"{entry.artifact_id}: missing or unreadable")
    return problems


def _rename_create_only(source: Path, destination: Path) -> None:
    # Linux exposes an atomic create-only rename. This also rejects empty existing
    # destination directories, which plain os.rename would otherwise replace.
    if platform.system() == "Linux":
        library = ctypes.CDLL(None, use_errno=True)
        rename = getattr(library, "renameat2", None)
        if rename is not None:
            rename.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            ]
            rename.restype = ctypes.c_int
            if (
                rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
                == 0
            ):
                return
            error = ctypes.get_errno()
            if error not in (errno.ENOSYS, errno.EINVAL):
                raise OSError(error, os.strerror(error), destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    # The exclusive claim held by JobDirectory prevents races between cooperating
    # writers on platforms without RENAME_NOREPLACE. Adapter root policy must also
    # exclude untrusted writers on those platforms.
    os.rename(source, destination)


class JobDirectory:
    """Own a temporary job directory and publish a complete immutable snapshot.

    Failures are finalized with their partial evidence instead of discarded. A
    supervisor should call finalize only after its worker has exited and closed files.
    """

    def __init__(
        self,
        output_root: str | Path,
        job_id: str,
        request: dict[str, Any] | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", job_id) or job_id in (
            ".",
            "..",
        ):
            raise ValueError("Invalid job ID")
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.job_id = job_id
        self.final_path = self.output_root / job_id
        self.request_fingerprint = fingerprint(request) if request is not None else ""
        self._claim = self.output_root / f".{job_id}.claim"
        self._finalized = False
        if self.final_path.exists() or self.final_path.is_symlink():
            raise FileExistsError(self.final_path)
        descriptor = os.open(self._claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        try:
            self.path = Path(
                tempfile.mkdtemp(
                    prefix=f".{job_id}.", suffix=".tmp", dir=self.output_root
                )
            )
            (self.path / "diagnostics").mkdir()
            (self.path / "artifacts").mkdir()
            if request is not None:
                atomic_write_json(self.path / "request.json", request)
        except BaseException:
            self._claim.unlink(missing_ok=True)
            raise

    def finalize(
        self,
        manifest: ArtifactManifest | None = None,
        status: ReportStatus | str = ReportStatus.SUCCEEDED,
        failure: Failure | None = None,
    ) -> Path:
        if self._finalized:
            raise RuntimeError("Job directory is already finalized")
        if manifest is None:
            manifest = build_manifest(
                self.path,
                self.job_id,
                self.request_fingerprint,
                status=status,
                failure=failure,
            )
        if manifest.job_id != self.job_id:
            raise ValueError("Manifest job ID does not match job directory")
        if (
            self.request_fingerprint
            and manifest.request_fingerprint != self.request_fingerprint
        ):
            raise ValueError(
                "Manifest request fingerprint does not match job directory"
            )
        problems = verify_manifest(manifest, self.path)
        if problems:
            raise ValueError(
                "Manifest integrity validation failed: " + "; ".join(problems)
            )
        contents = list(self.path.rglob("*"))
        if any(path.is_symlink() for path in contents):
            raise ValueError("Job contents must not contain symlinks")
        actual_paths = {
            path.relative_to(self.path).as_posix()
            for path in contents
            if path.is_file()
            and path.relative_to(self.path).as_posix() != "manifest.json"
        }
        if actual_paths != {entry.path for entry in manifest.entries}:
            raise ValueError("Manifest must include every file in the job directory")
        atomic_write_json(self.path / "manifest.json", manifest)
        _rename_create_only(self.path, self.final_path)
        _sync_directory(self.output_root)
        self._claim.unlink(missing_ok=True)
        self._finalized = True
        return self.final_path

    def abort(self, failure: Failure | None = None) -> Path:
        if failure is None:
            failure = Failure(
                stage=Stage.FINALIZATION,
                category=FailureCategory.CANCELLED,
                message="Job aborted",
                retry_unchanged=True,
                retry_after_change=False,
            )
        return self.finalize(status=status_for_failure(failure), failure=failure)

    def __enter__(self) -> JobDirectory:
        return self

    def __exit__(
        self, exc_type: Any, exc: BaseException | None, tb: Any
    ) -> Literal[False]:
        if not self._finalized:
            failure = classify_exception(exc, Stage.FINALIZATION) if exc else None
            self.finalize(
                status=(
                    status_for_failure(failure)
                    if failure is not None
                    else ReportStatus.SUCCEEDED
                ),
                failure=failure,
            )
        return False


__all__ = [
    "JobDirectory",
    "atomic_write_json",
    "build_manifest",
    "collect_environment",
    "fingerprint",
    "hash_file",
    "hash_graph",
    "hash_inputs",
    "hash_weights",
    "verify_manifest",
]
