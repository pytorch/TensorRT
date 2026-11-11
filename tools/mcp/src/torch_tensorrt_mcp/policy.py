"""Administrator-selected trusted factories and create-only artifact locations."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .storage import read_json

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_JOB = re.compile(r"^[0-9a-f]{32}$")
_OUTPUT_DIR = re.compile(
    r"^\d{8}_\d{6}_[a-z_]+_[A-Za-z][A-Za-z0-9_-]{0,63}_[0-9a-f]{8}$"
)


class PolicyError(ValueError):
    pass


def confined_path(root: Path, relative: str, *, existing: bool = True) -> Path:
    """Reject symlinks at every component, not merely escapes after resolve()."""
    candidate = Path(relative)
    if candidate.is_absolute():
        raise PolicyError(f"path must be relative to {root}: {relative!r}")
    if not candidate.parts:
        raise PolicyError("path must not be empty")
    invalid_parts = [part for part in candidate.parts if part in ("..", ".")]
    if invalid_parts:
        raise PolicyError(
            f"path contains forbidden traversal component {invalid_parts[0]!r}: "
            f"{relative!r}"
        )
    current = root
    for component in candidate.parts:
        current /= component
        if current.is_symlink():
            raise PolicyError(f"symlink path component is not permitted: {current}")
    resolved = current.resolve(strict=existing)
    if not resolved.is_relative_to(root):
        raise PolicyError(f"resolved path {resolved} is outside configured root {root}")
    return resolved


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Factory:
    path: str
    function: str
    parameters: tuple[str, ...] = ()

    def validate_parameters(self, values: dict[str, Any]) -> None:
        unexpected = set(values) - set(self.parameters)
        if unexpected:
            raise PolicyError(
                "factory arguments are not administrator-allowlisted: "
                + ", ".join(sorted(unexpected))
                + "; allowed arguments: "
                + (", ".join(self.parameters) if self.parameters else "<none>")
            )


@dataclass(frozen=True)
class Policy:
    workspace_root: Path
    output_root: Path
    models: dict[str, Factory]
    inputs: dict[str, Factory]
    max_response_bytes: int = 262144
    max_pending_jobs: int = 32
    max_timeout_seconds: float = 3600.0

    @classmethod
    def load(cls, config_path: Path | str) -> "Policy":
        path = Path(config_path).resolve(strict=True)
        if path.stat().st_size > 1048576:
            raise PolicyError(
                f"configuration {path} is {path.stat().st_size} bytes, while "
                "maximum size is 1048576 bytes"
            )
        data = json.loads(path.read_text())
        required = {"workspace_root", "output_root", "models", "inputs"}
        allowed = required | {
            "max_response_bytes",
            "max_pending_jobs",
            "max_timeout_seconds",
        }
        if not isinstance(data, dict):
            raise PolicyError(
                "administrator configuration must be a JSON object, "
                f"got {type(data).__name__}"
            )
        unexpected_keys = set(data) - allowed
        missing_keys = required - set(data)
        if unexpected_keys or missing_keys:
            details = []
            if missing_keys:
                details.append("missing: " + ", ".join(sorted(missing_keys)))
            if unexpected_keys:
                details.append("unexpected: " + ", ".join(sorted(unexpected_keys)))
            raise PolicyError(
                "invalid administrator configuration keys (" + "; ".join(details) + ")"
            )
        workspace = Path(data["workspace_root"]).resolve(strict=True)
        if not workspace.is_dir():
            raise PolicyError(
                f"workspace_root must be an existing directory: {workspace}"
            )
        configured_output = Path(data["output_root"])
        try:
            configured_output.mkdir(mode=0o700, parents=True, exist_ok=True)
            output = configured_output.resolve(strict=True)
        except OSError as exc:
            raise PolicyError(
                f"output_root could not be created: {configured_output}"
            ) from exc
        if not output.is_dir():
            raise PolicyError(f"output_root must be a directory: {output}")
        if output == workspace or output in workspace.parents:
            raise PolicyError(
                f"output_root {output} cannot equal or contain workspace_root {workspace}"
            )
        try:
            candidate_root = confined_path(
                output, ".lowering_pass_candidates", existing=False
            )
            candidate_root.mkdir(mode=0o700, exist_ok=True)
            candidate_root.chmod(0o700)
            candidate_root = confined_path(output, ".lowering_pass_candidates")
        except (OSError, PolicyError) as exc:
            raise PolicyError(
                f"private candidate directory could not be created under {output}"
            ) from exc
        if not candidate_root.is_dir():
            raise PolicyError(
                f"private candidate path must be a directory: {candidate_root}"
            )

        def factories(raw: Any) -> dict[str, Factory]:
            if not isinstance(raw, dict):
                raise PolicyError(
                    f"factory registry must be a JSON object, got {type(raw).__name__}"
                )
            result: dict[str, Factory] = {}
            for name, spec in raw.items():
                if not isinstance(name, str) or not _NAME.fullmatch(name):
                    raise PolicyError(
                        f"invalid factory name {name!r}; expected 1–64 characters, "
                        "starting with a letter and containing only letters, digits, "
                        "'_' or '-'"
                    )
                if not isinstance(spec, dict):
                    raise PolicyError(
                        f"factory {name!r} configuration must be a JSON object, "
                        f"got {type(spec).__name__}"
                    )
                factory_allowed = {"path", "function", "parameters"}
                factory_required = {
                    "path",
                    "function",
                }
                unexpected_keys = set(spec) - factory_allowed
                missing_keys = factory_required - set(spec)
                if unexpected_keys or missing_keys:
                    details = []
                    if missing_keys:
                        details.append("missing: " + ", ".join(sorted(missing_keys)))
                    if unexpected_keys:
                        details.append(
                            "unexpected: " + ", ".join(sorted(unexpected_keys))
                        )
                    raise PolicyError(
                        f"invalid configuration keys for factory {name!r} "
                        f"({'; '.join(details)})"
                    )
                source = confined_path(workspace, spec["path"])
                function = spec["function"]
                parameters = spec.get("parameters", [])
                if source.suffix != ".py" or not source.is_file():
                    raise PolicyError(
                        f"factory {name!r} path must name an existing Python file: "
                        f"{source}"
                    )
                if not isinstance(function, str) or not function.isidentifier():
                    raise PolicyError(
                        f"factory {name!r} function must be a valid Python identifier, "
                        f"got {function!r}"
                    )
                if not isinstance(parameters, list):
                    raise PolicyError(
                        f"factory {name!r} parameters must be a JSON array, "
                        f"got {type(parameters).__name__}"
                    )
                invalid_parameters = [
                    value
                    for value in parameters
                    if not isinstance(value, str)
                    or not value.isidentifier()
                    or value in ("seed", "lowering_pass_candidate_id")
                ]
                if invalid_parameters:
                    raise PolicyError(
                        f"factory {name!r} has invalid parameter names: "
                        + ", ".join(repr(value) for value in invalid_parameters)
                        + "; each must be a Python identifier and cannot be "
                        "'seed' or 'lowering_pass_candidate_id'"
                    )
                result[name] = Factory(
                    str(source.relative_to(workspace)),
                    function,
                    tuple(parameters),
                )
            return result

        max_response_bytes = data.get("max_response_bytes", 262144)
        max_pending_jobs = data.get("max_pending_jobs", 32)
        max_timeout = data.get("max_timeout_seconds", 3600.0)
        if (
            type(max_response_bytes) is not int
            or not 4096 <= max_response_bytes <= 16777216
        ):
            raise PolicyError(
                "max_response_bytes must be an integer from 4096 to 16777216, "
                f"got {max_response_bytes!r}"
            )
        if type(max_pending_jobs) is not int or not 1 <= max_pending_jobs <= 1024:
            raise PolicyError(
                "max_pending_jobs must be an integer from 1 to 1024, "
                f"got {max_pending_jobs!r}"
            )
        if type(max_timeout) not in (float, int) or not 0 < max_timeout <= 86400:
            raise PolicyError(
                "max_timeout_seconds must be a number greater than 0 and at most "
                f"86400, got {max_timeout!r}"
            )
        return cls(
            workspace_root=workspace,
            output_root=output,
            models=factories(data["models"]),
            inputs=factories(data["inputs"]),
            max_response_bytes=max_response_bytes,
            max_pending_jobs=max_pending_jobs,
            max_timeout_seconds=float(max_timeout),
        )

    def validate_request(self, operation: str, request: dict[str, Any]) -> None:
        if request["timeout_seconds"] > self.max_timeout_seconds:
            raise PolicyError(
                f"requested timeout_seconds {request['timeout_seconds']} exceeds "
                f"administrator limit {self.max_timeout_seconds}"
            )
        if operation in ("analyze", "compile"):
            lowering_pass_candidate_id = request.get("lowering_pass_candidate_id")
            if lowering_pass_candidate_id is not None:
                self.candidate_path(lowering_pass_candidate_id)
            self.validate_model_request(request)

    def validate_model_request(self, request: dict[str, Any]) -> None:
        """Validate factories without reapplying execution limits."""
        for kind, registry, name, parameters in (
            ("model", self.models, request["model"], request["model_parameters"]),
            ("inputs", self.inputs, request["inputs"], request["input_parameters"]),
        ):
            if name not in registry:
                raise PolicyError(
                    f"unknown administrator-configured {kind} factory {name!r}; "
                    "available factories: "
                    + (", ".join(sorted(registry)) if registry else "<none>")
                )
            factory = registry[name]
            factory.validate_parameters(parameters)

    def candidate_path(self, lowering_pass_candidate_id: str) -> Path:
        if not isinstance(lowering_pass_candidate_id, str) or not _JOB.fullmatch(
            lowering_pass_candidate_id
        ):
            raise PolicyError(
                "invalid candidate ID; expected exactly 32 lowercase hexadecimal "
                "characters"
            )
        relative = (
            ".lowering_pass_candidates/"
            f"lowering_pass_candidate_{lowering_pass_candidate_id}.py"
        )
        try:
            path = confined_path(self.output_root, relative)
        except FileNotFoundError as exc:
            raise PolicyError(
                "candidate source must be a regular file: "
                f"{self.output_root / relative}"
            ) from exc
        if not path.is_file():
            raise PolicyError(f"candidate source must be a regular file: {path}")
        return path

    def load_candidate(self, lowering_pass_candidate_id: str | None) -> Any | None:
        """Load one policy-confined lowering candidate in the execution worker."""
        if lowering_pass_candidate_id is None:
            return None
        path = self.candidate_path(lowering_pass_candidate_id)
        module_name = f"_torch_tensorrt_candidate_{lowering_pass_candidate_id}"
        module = sys.modules.get(module_name)
        if module is not None:
            return module
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise PolicyError(
                "could not create an import specification for candidate "
                f"{lowering_pass_candidate_id}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        return module

    def job_path(self, job_id: str) -> Path:
        if not _JOB.fullmatch(job_id):
            raise PolicyError(
                f"invalid job ID {job_id!r}; expected exactly 32 lowercase "
                "hexadecimal characters"
            )
        record_path = confined_path(
            self.output_root, f".jobs/{job_id}.json", existing=False
        )
        try:
            record = read_json(record_path)
        except (OSError, ValueError) as exc:
            raise PolicyError(f"job metadata is unavailable for {job_id}") from exc
        output_dir = record.get("output_dir")
        if (
            record.get("job_id") != job_id
            or not isinstance(output_dir, str)
            or not _OUTPUT_DIR.fullmatch(output_dir)
        ):
            raise PolicyError("job metadata has an invalid output directory")
        if not output_dir.endswith("_" + job_id[:8]):
            raise PolicyError("job output directory does not match its job ID")
        return confined_path(self.output_root, output_dir)

    def load_factory(self, factory: Factory) -> Any:
        """Called only inside an execution worker, after request authorization."""
        path = confined_path(self.workspace_root, factory.path)
        source_identity = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
        module_name = "_torch_tensorrt_factory_" + source_identity
        # Trusted sibling imports may happen at import time or inside a factory.
        # Keep these roots in the execution worker's path for both cases.
        for import_root in (self.workspace_root, path.parent):
            if str(import_root) not in sys.path:
                sys.path.insert(0, str(import_root))
        module = sys.modules.get(module_name)
        if module is not None:
            function = getattr(module, factory.function)
            if not callable(function):
                raise PolicyError(
                    f"configured factory function {factory.function!r} in {path} "
                    f"is not callable (got {type(function).__name__})"
                )
            return function
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise PolicyError(
                f"could not create an import specification for factory source {path}"
            )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        function = getattr(module, factory.function)
        if not callable(function):
            raise PolicyError(
                f"configured factory function {factory.function!r} in {path} "
                f"is not callable (got {type(function).__name__})"
            )
        return function
