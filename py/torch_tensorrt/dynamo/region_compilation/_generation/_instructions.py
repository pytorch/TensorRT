# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Packaged author guidance and bounded reads of explicitly registered evidence."""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Any

_MAX_RESOURCE_BYTES = 262144
_COMMON_RESOURCES = (
    "SKILL.md",
    "references/correctness.md",
    "references/evaluation.md",
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _snapshot(name: str, text: str, origin: str) -> dict[str, str]:
    if not text.strip():
        raise ValueError(f"Instruction resource {name!r} is empty")
    if len(text.encode("utf-8")) > _MAX_RESOURCE_BYTES:
        raise ValueError(f"Instruction resource {name!r} exceeds 256 KiB")
    return {"name": name, "text": text, "sha256": _sha256(text), "origin": origin}


def load_instructions(
    backend_name: str,
    instruction_resources: tuple[str, ...] = (),
    instructions: str | None = None,
    instruction_files: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    """Snapshot selected guidance without depending on a source checkout.

    Built-in resource names are relative to ``skills/kernel-author``. A custom
    backend may name ``package:relative/path.md`` resources or supply explicit
    instruction files.
    The digest binds the exact instructions used for this generation run.
    """
    root = resources.files(__package__).joinpath("skills", "kernel-author")
    snapshots = []
    for name in dict.fromkeys((*_COMMON_RESOURCES, *instruction_resources)):
        package, separator, resource = name.partition(":")
        selected_root = root
        if separator:
            if not all(part.isidentifier() for part in package.split(".")):
                raise ValueError(f"Invalid instruction resource package: {package!r}")
            selected_root = resources.files(package)
        else:
            resource = name
        resource_path = PurePosixPath(resource)
        if (
            not resource
            or resource_path.is_absolute()
            or ".." in resource_path.parts
            or "\\" in name
        ):
            raise ValueError(
                f"Instruction resource must be a package-relative path: {name!r}"
            )
        with selected_root.joinpath(*resource_path.parts).open("rb") as stream:
            data = stream.read(_MAX_RESOURCE_BYTES + 1)
        if len(data) > _MAX_RESOURCE_BYTES:
            raise ValueError(f"Instruction resource {name!r} exceeds 256 KiB")
        snapshots.append(
            _snapshot(name, data.decode("utf-8"), "package" if separator else "builtin")
        )
    for filename in instruction_files:
        file_path = Path(filename).expanduser().resolve()
        with file_path.open("rb") as stream:
            data = stream.read(_MAX_RESOURCE_BYTES + 1)
        if len(data) > _MAX_RESOURCE_BYTES:
            raise ValueError(f"Instruction resource {str(file_path)!r} exceeds 256 KiB")
        snapshots.append(_snapshot(str(file_path), data.decode("utf-8"), "file"))
    if instructions and instructions.strip():
        snapshots.append(_snapshot("user-instructions", instructions, "inline"))
    return _bundle(backend_name, snapshots)


def _bundle(backend_name: str, snapshots: list[dict[str, str]]) -> dict[str, Any]:
    bundle: dict[str, Any] = {
        "version": 1,
        "backend": backend_name,
        "resources": snapshots,
    }
    bundle["sha256"] = _sha256(json.dumps(bundle, sort_keys=True, ensure_ascii=False))
    bundle["text"] = "\n\n".join(
        f"INSTRUCTION RESOURCE: {entry['name']}\n{entry['text']}" for entry in snapshots
    )
    return bundle


def validate_instructions(bundle: dict[str, Any], backend_name: str) -> dict[str, Any]:
    """Validate and copy a saved bundle for exact replay without reading its paths."""
    if (
        not isinstance(bundle, dict)
        or set(bundle) != {"version", "backend", "resources", "sha256", "text"}
        or type(bundle.get("version")) is not int
        or bundle["version"] != 1
        or bundle["backend"] != backend_name
        or not isinstance(bundle["resources"], list)
        or not bundle["resources"]
    ):
        raise ValueError("Invalid saved instruction bundle or backend mismatch")
    snapshots = []
    for resource in bundle["resources"]:
        if (
            not isinstance(resource, dict)
            or set(resource) != {"name", "text", "origin", "sha256"}
            or any(not isinstance(value, str) for value in resource.values())
            or not resource["name"]
            or resource["origin"] not in {"builtin", "package", "file", "inline"}
        ):
            raise ValueError("Invalid saved instruction resource")
        snapshot = _snapshot(resource["name"], resource["text"], resource["origin"])
        if snapshot["sha256"] != resource["sha256"]:
            raise ValueError(
                f"Instruction resource digest mismatch: {resource['name']}"
            )
        snapshots.append(snapshot)
    restored = _bundle(backend_name, snapshots)
    if restored["sha256"] != bundle["sha256"]:
        raise ValueError("Saved instruction bundle digest mismatch")
    if restored["text"] != bundle["text"]:
        raise ValueError("Saved instruction text does not match its resources")
    return restored


class EvidenceCatalog:
    """Read immutable text snapshots by logical name, never arbitrary file paths.

    The controller registers known references and worker results. Providers may
    request pages; the controller owns read-count and wall-time budgets.
    """

    def __init__(self, *, page_chars: int = 8192) -> None:
        if type(page_chars) is not int or not 1 <= page_chars <= 65536:
            raise ValueError("Evidence page_chars must be an integer in [1, 65536]")
        self._page_chars = page_chars
        self._resources: dict[str, str] = {}

    def add(self, name: str, text: str) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Evidence resource names must be nonempty strings")
        if not isinstance(text, str):
            raise TypeError("Evidence resources must be text")
        if name in self._resources and self._resources[name] != text:
            raise ValueError(f"Evidence resource {name!r} is already registered")
        self._resources[name] = text

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {"resource": name, "chars": len(text), "sha256": _sha256(text)}
            for name, text in self._resources.items()
        ]

    def read(self, resource: str, offset: int = 0) -> dict[str, Any]:
        if not isinstance(resource, str) or resource not in self._resources:
            raise ValueError(f"Unknown evidence resource: {resource!r}")
        text = self._resources[resource]
        if type(offset) is not int or not 0 <= offset <= len(text):
            raise ValueError("Evidence offset must be an integer within the resource")
        end = min(len(text), offset + self._page_chars)
        return {
            "resource": resource,
            "offset": offset,
            "text": text[offset:end],
            "next_offset": end if end < len(text) else None,
            "sha256": _sha256(text),
        }
