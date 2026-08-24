"""Every ExecuTorch pin in the repository must agree with dev_dep_versions.yml.

Sites are discovered by search, not listed here. A list of paths would drift the same way
the pins it checks drifted.

Two spellings are correct, for different reasons, and which one a site needs depends on what
that site produces. See the comment above EXECUTORCH_REQUIREMENT in setup.py. Requiring the
right spelling per role is the point: installable metadata that pins exactly would reject a
compatible patch release, and a build input that takes a range could resolve an ExecuTorch
the artifact was not compiled against.

Agreeing with the file is necessary but not sufficient, so
test_the_pinned_commit_is_the_pinned_wheels_own_source closes the gap the others leave: they
only prove the repository is self-consistent, which it would be even if the wheel and the
commit named two different ExecuTorch trees.
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
VERSIONS = REPO_ROOT / "dev_dep_versions.yml"

# Every PEP 440 operator, not just the two this repository happens to use, and an optional
# space before it. A site added with a compatible-release or bare-inequality operator is a site
# that drifted from the pin, and it should be visible to the search rather than silently
# exempt. Operators are named through the pattern rather than spelled out in prose here,
# because the search below reads this file too and an example would read as such a site.
REQUIREMENT = re.compile(
    r"executorch\s*(?:===|==|>=|<=|~=|!=|<|>)\s*[0-9][^\"'\s,`]*(?:,\s*<[0-9.]+)?"
)

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
# Only literal requirements land here. setup.py and tests/ci/runner.py derive theirs from
# dev_dep_versions.yml, so the search below no longer sees them and
# test_derived_requirements_match_the_pin covers them instead.
#
# Empty today: the only literal range left was the justfile's install recipe, and it installs
# the wheel the delegate is compiled against, so it pins exactly like the rest.
RANGE_SITES: frozenset[str] = frozenset()

# A step that exists to reproduce what a user runs belongs to the range group even inside a
# file that otherwise pins build inputs, so the marker travels with the line rather than
# with the path.
USER_WORKFLOW_MARKER = "verify the end user's workflow"

# The files expected to pin ExecuTorch to a version, excluding dev_dep_versions.yml itself.
# Asserted as a set of files rather than a line count because a site that loses its version stops
# matching the search rather than reporting a mismatch, and because the line count legitimately
# differs between this branch and the stacked runtime-wheel change.
_EXPECTED_REQUIREMENT_FILES = {
    ".github/workflows/executorch-build-linux.yml",
    ".github/workflows/executorch-test-linux.yml",
    "MODULE.bazel",
    "docker/MODULE.bazel.docker",
    "docker/MODULE.bazel.ngc",
    "justfile",
    "py/torch-tensorrt-executorch-runtime/README.md",
    "py/torch-tensorrt-executorch-runtime/pyproject.toml",
    "toolchains/ci_workspaces/MODULE.bazel.tmpl",
}


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


def _release_line(version: str) -> tuple[str, str]:
    """Split a pin into its major and minor, for either a release or a nightly.

    ``1.4.1`` and ``1.5.0.dev20260822`` both belong to the release line their first two
    fields name, so the range a site gets is derived from those and nothing else. Splitting
    on every dot instead assumes three fields and raises on the nightly form.
    """
    major, minor = version.split(".")[:2]
    return major, minor


def _expected(path: str, number: int, version: str) -> str:
    if not _wants_range(path, number):
        return f"executorch=={version}"

    major, minor = _release_line(version)
    # Only the top-level setup.py carries the Linux marker. It is the site uv resolves for the
    # win32 required-environment, where PyPI's candidates stop below this floor.
    marker = "; platform_system == 'Linux'" if path == "setup.py" and number > 1 else ""
    return f"executorch>={version},<{major}.{int(minor) + 1}{marker}"


def test_every_requirement_matches_the_pin() -> None:
    version = _versions()["__executorch_version__"]

    wrong = []
    found = 0
    seen = set()
    for line in _git(
        "grep", "-nI", "-E", r"executorch ?(===|==|>=|<=|~=|!=|<|>) ?[0-9]"
    ).splitlines():
        path, number, text = line.split(":", 2)
        if path == VERSIONS.name:
            continue
        expected = _expected(path, int(number), version)
        for actual in REQUIREMENT.findall(text):
            found += 1
            seen.add(path)
            if actual != expected:
                wrong.append(f"{path}:{number} has {actual}, expected {expected}")

    assert found, "no ExecuTorch requirement found, so this test is not looking"
    # The set of files, not just "nonzero": a site that drops its version entirely stops matching
    # the search and silently leaves the result set, which is exactly how a lost pin would look.
    assert seen == _EXPECTED_REQUIREMENT_FILES, (
        "the set of files pinning ExecuTorch changed.\n"
        f"  no longer pinning: {sorted(_EXPECTED_REQUIREMENT_FILES - seen)}\n"
        f"  newly pinning:     {sorted(seen - _EXPECTED_REQUIREMENT_FILES)}\n"
        "A file that lost its version does not appear in the search at all, so check for one "
        "that now names bare `executorch` before updating the expected set."
    )
    assert not wrong, "\n  ".join(["", *wrong])


def _setup_py_requirement(version: str) -> str:
    # setup.py cannot be imported here, importing it starts a build, so lift out the
    # statements that derive the requirement and evaluate only those.
    derived = {"_executorch_major", "_executorch_minor", "EXECUTORCH_REQUIREMENT"}
    statements = [
        node
        for node in ast.parse((REPO_ROOT / "setup.py").read_text()).body
        if isinstance(node, ast.Assign)
        and derived
        & {
            name.id
            for target in node.targets
            for name in ast.walk(target)
            if isinstance(name, ast.Name)
        }
    ]
    assert statements, "setup.py no longer derives EXECUTORCH_REQUIREMENT"

    namespace: dict = {"__executorch_version__": version}
    exec(compile(ast.Module(statements, []), "setup.py", "exec"), namespace)
    return namespace["EXECUTORCH_REQUIREMENT"]


def _runner_requirement(root: Path) -> str:
    # A subprocess rather than an import, so pointing the runner at another tree cannot
    # leave a reloaded module behind for whatever runs next.
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "from tests.ci.runner import _executorch_requirement; "
            "print(_executorch_requirement())",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "TRT_REPO_ROOT": str(root)},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_derived_requirements_match_the_pin() -> None:
    # setup.py and tests/ci/runner.py build their requirement from the pin, so the search
    # above cannot see them. Check the strings they produce instead.
    #
    # They want different shapes. setup.py declares what users may install, so it is a range
    # over the release line. runner.py installs the wheel CI tests the delegate against, and
    # the delegate is compiled from the commit the pin names, so it has to be exact: the
    # nightly channel gains a member every day and a range there silently unpairs the two.
    version = _versions()["__executorch_version__"]
    major, minor = _release_line(version)

    # The Linux marker is part of the requirement: the extra has to resolve for the win32 entry
    # in pyproject.toml's uv required-environments, where the only candidates are PyPI's and they
    # stop below this floor, so without it `uv lock` fails outright.
    assert _setup_py_requirement(version) == (
        f"executorch>={version},<{major}.{int(minor) + 1}; platform_system == 'Linux'"
    )
    assert _runner_requirement(REPO_ROOT) == f"executorch=={version}"


def test_the_runner_follows_the_row_s_cuda_version(monkeypatch) -> None:
    """Call the runner and read the URL it builds, rather than matching its source.

    The executorch suite is nightly-only, and the nightly matrix runs cu132 rows as well as cu130
    ones, so a fixed channel would install a CUDA 13.0 ExecuTorch into a CUDA 13.2 job. PRs pin to
    cu130, which is why watching PR CI cannot catch it. A source-text assertion could not catch it
    either: keeping the ``os.environ.get`` line while hardcoding the URL passes one.
    """
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    try:
        from ci import runner
    finally:
        sys.path.pop(0)

    def channel_for(cu_version: str | None) -> str:
        if cu_version is None:
            monkeypatch.delenv("CU_VERSION", raising=False)
        else:
            monkeypatch.setenv("CU_VERSION", cu_version)
        commands = runner._setup_commands("executorch")
        urls = [
            argument
            for command, _ in commands
            for argument in command
            if "download.pytorch.org" in argument
        ]
        assert len(urls) == 1, f"expected one index URL, got {urls}"
        return urls[0]

    assert channel_for("cu132").endswith("/nightly/cu132")
    assert channel_for("cu130").endswith("/nightly/cu130")
    # Unset is a local run, and matches the index pyproject.toml resolves against by default.
    assert channel_for(None).endswith("/nightly/cu130")


def test_derived_requirements_roll_the_minor_over(tmp_path: Path) -> None:
    # The upper bound is a version, not a decimal: 1.9 has to become 1.10, not 1.1.
    # Spelled through a variable because the search above reads this file too, and a
    # written-out requirement here would read as a site that drifted from the pin.
    version = "1.9.0"
    (tmp_path / "dev_dep_versions.yml").write_text(
        f'__executorch_version__: "{version}"\n'
    )

    assert (
        _setup_py_requirement(version)
        == f"executorch>={version},<1.10; platform_system == 'Linux'"
    )
    # No upper bound to roll over, but it must still track the pin it is given.
    assert _runner_requirement(tmp_path) == f"executorch=={version}"


def test_the_pinned_commit_is_the_pinned_wheels_own_source() -> None:
    """The two pins must name one ExecuTorch, not two that happen to be close.

    ``__executorch_version__`` selects the wheel the delegate is built to sit alongside, and
    ``__executorch_commit__`` selects the tree it compiles from. Nothing about the two strings
    forces them to agree, and a mismatch is invisible: both pins look plausible, the build
    succeeds, and the delegate is compiled from one ExecuTorch while running against another.
    Every published wheel records the commit it was built from, so the pairing is checkable
    rather than a convention.

    Skipped rather than failed whenever the installed wheel is not the one the pin names — not
    installed at all, a different member of a floating range, or built without git provenance.
    None of those say anything about whether the two pins agree, and this file stays readable
    offline.
    """
    versions = _versions()
    expected_commit = versions["__executorch_commit__"]
    expected_version = versions["__executorch_version__"]

    try:
        from executorch.version import __version__ as installed_version
        from executorch.version import git_version as installed_commit
    except ImportError:
        pytest.skip("executorch is not installed, so the pinned wheel cannot be read")

    if installed_commit is None:
        # ExecuTorch records this as Optional[str] and writes None when it is built outside a
        # git checkout. Such a wheel carries no provenance to compare, which is not the pins
        # disagreeing.
        pytest.skip(
            f"the installed ExecuTorch {installed_version} records no source commit, "
            "so the pairing cannot be checked against it"
        )

    # The wheel carries a local version label naming its CUDA build (`+cu132`), which the pin
    # deliberately omits so one pin serves every CUDA row. Compare the part they share.
    if installed_version.split("+")[0] != expected_version:
        # No evidence either way rather than a mismatch to report: only the wheel the pin names
        # carries the commit the pin should agree with. Every CI install path that builds or
        # tests the delegate requests the pin exactly -- the one deliberate range is the
        # end-user install rehearsal in executorch-build-linux.yml -- so arriving here usually
        # means the environment was built some other way, and that wheel's commit says nothing
        # about whether the two pins agree.
        pytest.skip(
            f"the installed ExecuTorch is {installed_version}, not the pinned "
            f"{expected_version}, so its commit says nothing about whether the pins agree"
        )

    assert installed_commit == expected_commit, (
        f"ExecuTorch {installed_version} was built from {installed_commit}, but "
        f"__executorch_commit__ pins {expected_commit}. The delegate would compile against "
        "one ExecuTorch and link another."
    )


def test_every_source_commit_matches_the_pin() -> None:
    commit = _versions()["__executorch_commit__"]

    wrong = []
    found = 0
    seen = set()
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
