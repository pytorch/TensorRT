# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Small helpers shared by the generation controller, worker, and CLI."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from filelock import FileLock

_LOGGER = logging.getLogger(__name__)
# Directories already reported as writable by others; each is reported once.
_REPORTED: set[Path] = set()

_CONTROL = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest(value: Any) -> str:
    """Hash canonical JSON; non-finite numbers are rejected."""
    return sha256_text(json.dumps(value, sort_keys=True, allow_nan=False))


def write_json(path: Path, value: Any) -> None:
    """Atomically replace ``path`` with indented JSON."""
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def strip_control(text: str) -> str:
    """Remove ANSI escape sequences and control characters from diagnostics."""
    return _CONTROL.sub("", text)


def scrub(text: str, directory: Path, placeholder: str = "<evaluation>") -> str:
    """Hide local paths and terminal control sequences from the author."""
    text = strip_control(text)
    for path, name in (
        (directory.resolve(), placeholder),
        (Path(os.path.expanduser("~")), "<home>"),
    ):
        # Whole paths only; a root home would otherwise rewrite every "/".
        if path.is_absolute() and path.parent != path:
            pattern = rf"(?<![\w.-]){re.escape(str(path))}(?![\w.-])"
            text = re.sub(pattern, name, text)
    return text


def private_directory(path: Path) -> Path:
    """Create ``path`` readable only by its owner, and refuse another user's.

    Runs hold prompts, provider logs, and kernels that are revalidated and then
    loaded, so a cache directory must not be shared with other users. An
    existing directory keeps its permissions, which its owner may have chosen;
    one that others can write is reported once per process.
    """
    from torch_tensorrt.region import RegionError

    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise RegionError(
            f"Kernel cache directory {path} belongs to another user; use a "
            "cache_dir that only you can write"
        )
    if info.st_mode & 0o022 and path not in _REPORTED:
        _REPORTED.add(path)
        _LOGGER.warning(
            "Kernel cache directory %s is writable by other users, who could "
            "replace its kernels; restrict it with chmod 700",
            path,
        )
    return path


def gpu_lock(root: Path, device: Any) -> FileLock:
    """Serialize evaluations that share one physical GPU under ``root``.

    The lock is keyed by the device UUID, so different logical indices of the
    same GPU (for example under CUDA_VISIBLE_DEVICES) share one lock.
    """
    import torch
    from filelock import FileLock

    uuid = torch.cuda.get_device_properties(torch.device(device)).uuid
    return FileLock(str(root / f"gpu-{uuid}.lock"))
