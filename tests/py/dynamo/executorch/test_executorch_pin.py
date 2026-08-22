"""Every ExecuTorch pin in the repository must agree with dev_dep_versions.yml.

Sites are discovered by search, not listed here. A list of paths would drift the same way
the pins it checks drifted.

Two spellings are correct, for different reasons, and which one a site needs depends on what
that site produces. See the comment above EXECUTORCH_REQUIREMENT in setup.py. Requiring the
right spelling per role is the point: installable metadata that pins exactly would reject a
compatible patch release, and a build input that takes a range could resolve an ExecuTorch
the artifact was not compiled against.
"""

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
VERSIONS = REPO_ROOT / "dev_dep_versions.yml"

REQUIREMENT = re.compile(r"executorch(?:==|>=)[0-9][^\"'\s,`]*(?:,<[0-9.]+)?")

# The bazel repository puts the commit on its own line, so this one has to run against file
# contents rather than a git grep line.
BAZEL_COMMIT = re.compile(
    r'name\s*=\s*"executorch".*?commit\s*=\s*"([0-9a-f]{40})"', re.DOTALL
)

# Anywhere else the commit appears it is named on the same line, as a shell default or prose.
NAMED_COMMIT = re.compile(r"executorch[_a-z]*[^0-9a-f]*([0-9a-f]{40})", re.IGNORECASE)

# Requirements that end up in metadata someone resolves at install time. A patch release off
# the same branch has to stay installable, so these take the range. Everything else pins
# exactly: a build input compiled against one wheel, or a comment labelling a source sha.
RANGE_SITES = frozenset(
    {
        "setup.py",
        "justfile",
        "tests/ci/runner.py",
    }
)

# A step that exists to reproduce what a user runs belongs to the range group even inside a
# file that otherwise pins build inputs, so the marker travels with the line rather than
# with the path.
USER_WORKFLOW_MARKER = "verify the end user's workflow"


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout


def _versions() -> dict:
    # Matches how setup.py and versions.py read this file, without needing yaml here.
    text = VERSIONS.read_text()
    return dict(re.findall(r'^(__\w+__): "([^"]+)"', text, re.MULTILINE))


def _wants_range(path: str, number: int) -> bool:
    if path in RANGE_SITES:
        return True

    # Scan upward rather than reading one fixed line, so reformatting the workflow cannot
    # silently reclassify the requirement and fail the build for an unrelated reason.
    lines = (REPO_ROOT / path).read_text().splitlines()
    for line in reversed(lines[: number - 1]):
        if not line.strip():
            continue
        return USER_WORKFLOW_MARKER in line
    return False


def _expected(path: str, number: int, version: str) -> str:
    if not _wants_range(path, number):
        return f"executorch=={version}"

    # Assumes X.Y.Z, which is what ExecuTorch releases and what this file records.
    major, minor, _ = version.split(".")
    return f"executorch>={version},<{major}.{int(minor) + 1}"


def test_every_requirement_matches_the_pin() -> None:
    version = _versions()["__executorch_version__"]

    wrong = []
    found = 0
    for line in _git("grep", "-nI", "-E", r"executorch(==|>=)[0-9]").splitlines():
        path, number, text = line.split(":", 2)
        if path == VERSIONS.name:
            continue
        expected = _expected(path, int(number), version)
        for actual in REQUIREMENT.findall(text):
            found += 1
            if actual != expected:
                wrong.append(f"{path}:{number} has {actual}, expected {expected}")

    assert found, "no ExecuTorch requirement found, so this test is not looking"
    assert not wrong, "\n  ".join(["", *wrong])


def test_every_source_commit_matches_the_pin() -> None:
    commit = _versions()["__executorch_commit__"]

    wrong = []
    found = 0
    for path in _git("grep", "-lI", "-E", 'name = "executorch"').split():
        for match in BAZEL_COMMIT.finditer((REPO_ROOT / path).read_text()):
            found += 1
            if match.group(1) != commit:
                wrong.append(f"{path} compiles {match.group(1)}")

    for line in _git("grep", "-nI", "-iE", NAMED_COMMIT.pattern).splitlines():
        path, number, text = line.split(":", 2)
        if path == VERSIONS.name:
            continue
        for actual in NAMED_COMMIT.findall(text):
            found += 1
            if actual != commit:
                wrong.append(f"{path}:{number} uses {actual}")

    assert found, "no ExecuTorch source commit found, so this test is not looking"
    assert not wrong, f"pin says {commit}:\n  " + "\n  ".join(wrong)
