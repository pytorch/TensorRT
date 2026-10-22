# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Regression checks for the immutable v0.1 schema compatibility gate."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "check_agent_contracts", Path(__file__).with_name("check_agent_contracts.py")
)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def git(root, *arguments):
    return subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "user.name=Contract Test",
            "-c",
            "user.email=contract-test@example.invalid",
            *arguments,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, "init", "-q")
    schemas = tmp_path / checker.SCHEMA_DIRECTORY
    schemas.mkdir(parents=True)
    path = schemas / "analysis.json"
    path.write_text(
        json.dumps({"type": "object", "properties": {"status": {"type": "string"}}})
    )
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "Base schema")
    return tmp_path, path, git(tmp_path, "rev-parse", "HEAD")


def test_formatting_changes_and_new_versions_preserve_frozen_contracts(repository):
    root, path, base = repository
    path.write_text(
        '{\n  "properties": {"status": {"type": "string"}},\n  "type": "object"\n}\n'
    )
    new = root / "schemas/reporting/v0.2"
    new.mkdir()
    (new / "analysis.json").write_text('{"type":"array"}')
    resolved, count, problems = checker.check_contracts(root, base)
    assert resolved == base and count == 1 and not problems


def test_changing_schema_and_current_golden_together_still_fails(repository):
    root, path, base = repository
    path.write_text('{"type":"object","properties":{"status":{"type":"integer"}}}')
    golden = root / "tests/py/reporting/golden/v0.1"
    golden.mkdir(parents=True)
    (golden / "analysis.json").write_text('{"status":1}')
    _, _, problems = checker.check_contracts(root, base)
    assert (
        len(problems) == 1
        and f"frozen v{checker.SCHEMA_VERSION} schema changed" in problems[0]
    )


def test_removing_frozen_schema_fails(repository):
    root, path, base = repository
    path.unlink()
    assert "removed" in checker.check_contracts(root, base)[2][0]


def test_first_introduction_does_not_require_old_schemas(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "commit", "--allow-empty", "-qm", "Before contracts")
    base = git(tmp_path, "rev-parse", "HEAD")
    directory = tmp_path / checker.SCHEMA_DIRECTORY
    directory.mkdir(parents=True)
    (directory / "analysis.json").write_text('{"type":"object"}')
    assert checker.check_contracts(tmp_path, base) == (base, 0, [])


@pytest.mark.parametrize("base", ["", "-HEAD", "not-a-real-commit", "HEAD --other"])
def test_invalid_base_cannot_silently_disable_gate(repository, base):
    with pytest.raises(ValueError):
        checker.check_contracts(repository[0], base)


def test_pr_base_is_validated_against_merge_parent(repository):
    root, path, base = repository
    git(root, "checkout", "-qb", "feature")
    (root / "feature.txt").write_text("feature")
    git(root, "add", ".")
    git(root, "commit", "-qm", "Feature")
    feature = git(root, "rev-parse", "HEAD")
    git(root, "checkout", "-q", "--detach", base)
    git(root, "merge", "--no-ff", "-m", "PR merge", feature)
    assert checker.check_contracts(root, base, require_merge_parent=True)[2] == []
    with pytest.raises(ValueError, match="merge parent"):
        checker.check_contracts(root, feature, require_merge_parent=True)
