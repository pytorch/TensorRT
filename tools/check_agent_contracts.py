# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Keep existing v0.1 evidence schemas frozen across a proposed change.

Compare to an explicitly resolved base commit, rather than regenerated fixtures
or a moving branch. New schema files/versions are allowed. Existing v0.1 files
cannot be removed or changed, even if their current fixtures also changed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = "0.1"
SCHEMA_DIRECTORY = f"schemas/reporting/v{SCHEMA_VERSION}"


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ValueError(
            "Git could not resolve the requested base or read its schema files"
        )
    return result.stdout


def _reject_constant(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _normalize(source: str) -> str:
    value = json.loads(
        source, object_pairs_hook=_unique_object, parse_constant=_reject_constant
    )
    if not isinstance(value, dict):
        raise ValueError("Schema must be a JSON object")
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def check_contracts(
    root: Path, base_ref: str, *, require_merge_parent: bool = False
) -> tuple[str, int, list[str]]:
    """Return resolved base, frozen-file count, and compatibility problems."""
    root = Path(root).resolve(strict=True)
    if (
        not base_ref
        or base_ref.startswith("-")
        or any(character.isspace() for character in base_ref)
    ):
        raise ValueError("Provide one explicit Git base commit or revision")
    base = _git(
        root, "rev-parse", "--verify", "--end-of-options", base_ref + "^{commit}"
    ).strip()
    if len(base) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in base
    ):
        raise ValueError("Base revision did not resolve to a full commit ID")
    if require_merge_parent:
        # The checkout must be the PR merge commit, not its head or a moving base.
        _git(root, "rev-parse", "--verify", "HEAD^2")
        parent = _git(root, "rev-parse", "--verify", "HEAD^1").strip()
        if parent != base:
            raise ValueError(
                "Explicit base commit does not match the checked-out PR merge parent"
            )
    tracked = _git(
        root, "ls-tree", "-r", "-z", "--name-only", base, "--", SCHEMA_DIRECTORY
    )
    paths = sorted(
        path
        for path in tracked.split("\0")
        if PurePosixPath(path).parent == PurePosixPath(SCHEMA_DIRECTORY)
        and path.endswith(".json")
    )
    problems = []
    for relative in paths:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            problems.append(
                f"{relative}: frozen schema removed, replaced, or symlinked"
            )
            continue
        try:
            previous = _normalize(_git(root, "show", f"{base}:{relative}"))
            current = _normalize(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            problems.append(f"{relative}: cannot compare valid schema JSON")
            continue
        if previous != current:
            problems.append(
                f"{relative}: frozen v{SCHEMA_VERSION} schema changed; "
                "introduce a new schema version"
            )
    return base, len(paths), problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ref",
        required=True,
        help="explicit base commit/revision; PR CI uses the event's base SHA",
    )
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--require-merge-parent",
        action="store_true",
        help="verify HEAD is a PR merge whose first parent is the explicit base",
    )
    arguments = parser.parse_args()
    try:
        base, count, problems = check_contracts(
            arguments.repo,
            arguments.base_ref,
            require_merge_parent=arguments.require_merge_parent,
        )
    except (OSError, ValueError) as error:
        print(f"Contract compatibility check failed: {error}", file=sys.stderr)
        return 2
    if problems:
        print(
            f"Frozen v{SCHEMA_VERSION} evidence contracts are incompatible "
            "with the base commit:",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(
        f"Checked {count} existing v{SCHEMA_VERSION} schema files against "
        f"{base}; no frozen contracts changed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
