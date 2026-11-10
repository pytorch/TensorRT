# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Code edits: source files a planner may edit, inside fences.

A workload may declare :class:`EditableFile` objects for source files it
builds from, such as a lowering pass or region kernels. Lines containing
``EDITABLE-START`` and ``EDITABLE-END`` fence the parts a planner may change.
A planner proposes SEARCH/REPLACE edits: each search text must occur exactly
once inside the fences of the file's current version (its version in the
best config), and the edited file must still compile. The session stores
the edited file under its SHA-256 in the session directory and records that
digest in the config as ``edit.<name>``. An edited candidate is then
reproducible, de-duplicated and reviewable like any other config.

The worker passes the edited file's path to :meth:`Workload.build` as
``values["edit.<name>"]`` (``None`` for the original); :func:`load_module`
imports it.

What the fences protect: a proposal can only change fenced code, so it
cannot reach the measurement code, the reference outputs or the accuracy
bounds. The edited code is not sandboxed, though: it runs in the worker
process that also checks and times the outputs, and :func:`check_compiles`
only checks that it compiles. So put the fences around code whose changes
the measurement can judge (launch settings, kernel bodies), never around
code that computes or compares outputs, and review stored edited files
before deploying them. The reference outputs are computed once, in a
separate worker, from the original code.
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

FENCE_START = "EDITABLE-START"
FENCE_END = "EDITABLE-END"
EDIT_PREFIX = "edit."
_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,62}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
# The edits of one file may change at most this many lines (each edit counts
# the larger of its search and replace line counts). It keeps diffs
# reviewable and leaves less room for an agent to rewrite what it should tune.
MAX_EDITED_LINES = 400


@dataclass(frozen=True)
class EditableFile:
    """A source file the planner may edit inside its EDITABLE fences.

    ``name`` is a lowercase identifier; the config key is ``edit.<name>``.
    ``path`` is the original file. ``description`` tells the planner what the
    fenced code does and what may change.
    """

    name: str
    path: str
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME_PATTERN.fullmatch(self.name):
            raise ValueError(
                f"EditableFile name must be a lowercase identifier; got {self.name!r}"
            )
        # Resolved, so a session resumed from another directory finds the file.
        object.__setattr__(self, "path", str(Path(self.path).expanduser().resolve()))
        source = Path(self.path)
        if not source.is_file():
            raise ValueError(f"EditableFile {self.name}: no file at {self.path}")
        if not fenced_spans(source.read_text()):
            raise ValueError(
                f"EditableFile {self.name}: {self.path} has no "
                f"{FENCE_START}/{FENCE_END} block"
            )

    @property
    def key(self) -> str:
        """The config key that selects an edited version: ``edit.<name>``."""
        return EDIT_PREFIX + self.name

    def to_json(self) -> dict[str, Any]:
        """The editable file as recorded with the workload, with the original's SHA-256.

        The digest makes a session detect an edit to the original between
        resumes, as it detects any other change to the workload.
        """
        return {
            "name": self.name,
            "path": self.path,
            "description": self.description,
            "sha256": hashlib.sha256(Path(self.path).read_bytes()).hexdigest(),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "EditableFile":
        return cls(data["name"], data["path"], data["description"])


def is_edit_key(name: str) -> bool:
    """Whether a config key selects an edited file (``edit.<name>``), not a knob."""
    return name.startswith(EDIT_PREFIX)


def is_edit_digest(value: Any) -> bool:
    """Whether ``value`` is a SHA-256 (64 hex digits), the name of an edited file."""
    return isinstance(value, str) and bool(_SHA256_PATTERN.fullmatch(value))


def fenced_spans(source: str) -> list[tuple[int, int]]:
    """Character spans of the editable text between fence lines (fences excluded)."""
    spans = []
    offset = 0
    start: int | None = None
    for line in source.splitlines(keepends=True):
        if FENCE_START in line:
            if start is not None:
                raise ValueError(f"nested {FENCE_START}")
            start = offset + len(line)
        elif FENCE_END in line:
            if start is None:
                raise ValueError(f"{FENCE_END} without {FENCE_START}")
            spans.append((start, offset))
            start = None
        offset += len(line)
    if start is not None:
        raise ValueError(f"unterminated {FENCE_START}")
    return spans


def _fence_lines(source: str) -> list[str]:
    """The fence lines of ``source``, in order; edits must leave them unchanged."""
    return [
        line
        for line in source.splitlines(keepends=True)
        if FENCE_START in line or FENCE_END in line
    ]


def _fenced_matches(
    source: str, search: str, spans: Sequence[tuple[int, int]]
) -> list[int]:
    """Offsets of every occurrence of ``search``, overlapping ones included, inside a span."""
    found = []
    for start, end in spans:
        at = source.find(search, start)
        while at != -1 and at + len(search) <= end:
            found.append(at)
            at = source.find(search, at + 1)
    return found


def apply_edits(source: str, edits: Sequence[Mapping[str, Any]]) -> str:
    """Apply SEARCH/REPLACE ``edits`` in order; raise ValueError if any is not allowed.

    Each search text must occur exactly once inside the fenced blocks (it may
    also occur elsewhere; only the fenced occurrence is edited). The edits
    must leave the fence lines unchanged and every block non-empty.
    """
    if not edits:
        raise ValueError("an edited file needs at least one edit")
    fences = _fence_lines(source)
    changed = 0
    for index, edit in enumerate(edits):
        search, replace = edit.get("search"), edit.get("replace")
        if not isinstance(search, str) or not search:
            raise ValueError(f"edit {index}: 'search' must be non-empty text")
        if not isinstance(replace, str):
            raise ValueError(f"edit {index}: 'replace' must be text")
        if FENCE_START in replace or FENCE_END in replace:
            raise ValueError(f"edit {index}: the replacement may not contain fences")
        found = _fenced_matches(source, search, fenced_spans(source))
        if not found:
            if search in source:
                raise ValueError(f"edit {index} is not inside an EDITABLE region")
            raise ValueError(f"edit {index}: the search text does not occur")
        if len(found) > 1:
            raise ValueError(
                f"edit {index}: the search text occurs {len(found)} times inside "
                "the EDITABLE regions; it must occur exactly once"
            )
        at = found[0]
        changed += max(search.count("\n"), replace.count("\n")) + 1
        source = source[:at] + replace + source[at + len(search) :]
    try:
        spans = fenced_spans(source)
    except ValueError as error:
        raise ValueError(f"the edits break the EDITABLE fences: {error}") from None
    if _fence_lines(source) != fences:
        raise ValueError("the edits change an EDITABLE fence line")
    if any(not source[start:end].strip() for start, end in spans):
        raise ValueError("the edits empty an EDITABLE region")
    if changed > MAX_EDITED_LINES:
        raise ValueError(
            f"the edits change about {changed} lines; at most {MAX_EDITED_LINES}"
        )
    return source


def check_compiles(source: str, filename: str) -> None:
    """Refuse an edited Python file that does not compile."""
    try:
        compile(source, filename, "exec")
    except SyntaxError as error:
        raise ValueError(f"the edited file does not compile: {error}") from None


class EditedFileStore:
    """A session's edited files, stored as ``edited_files/<sha256>.py``."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory) / "edited_files"

    def save(self, source: str) -> str:
        """Store ``source`` (once); return its SHA-256, its ``edit.<name>`` value."""
        sha = hashlib.sha256(source.encode()).hexdigest()
        path = self.path(sha)
        if not path.is_file():
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(source)
            temporary.replace(path)
        return sha

    def path(self, sha: str) -> Path:
        """Where the edited file with this SHA-256 is stored."""
        if not is_edit_digest(sha):
            raise ValueError(f"not the SHA-256 digest of an edited file: {sha!r}")
        return self.directory / f"{sha}.py"

    def source(self, editable: EditableFile, config: Mapping[str, Any]) -> str:
        """The file's source under ``config``: its edited version, or the original."""
        sha = config.get(editable.key)
        return self.path(sha).read_text() if sha else Path(editable.path).read_text()


def editable_blocks(source: str) -> list[str]:
    """The text of each fenced block of ``source``: what a planner sees and may edit."""
    return [source[start:end] for start, end in fenced_spans(source)]


def load_module(path: str | Path, name: str) -> ModuleType:
    """Import the Python file at ``path`` as a fresh module named ``name``."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
