"""Atomic, bounded JSON I/O helpers independent of the GPU runtime."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    fd, name = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path: Path, *, max_bytes: int = 16777216) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("expected a regular JSON file")
    with path.open("rb") as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("JSON file exceeds configured size limit")
    result = json.loads(data)
    if not isinstance(result, dict):
        raise ValueError("expected a JSON object")
    return result
