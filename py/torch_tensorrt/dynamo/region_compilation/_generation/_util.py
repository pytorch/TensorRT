# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Small helpers shared by the generation controller, worker, and CLI."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from filelock import FileLock

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


def gpu_lock(root: Path, device: Any) -> FileLock:
    """Serialize evaluations that share one physical GPU under ``root``.

    The lock is keyed by the device UUID, so different logical indices of the
    same GPU (for example under CUDA_VISIBLE_DEVICES) share one lock.
    """
    import torch
    from filelock import FileLock

    uuid = torch.cuda.get_device_properties(torch.device(device)).uuid
    return FileLock(str(root / f"gpu-{uuid}.lock"))
