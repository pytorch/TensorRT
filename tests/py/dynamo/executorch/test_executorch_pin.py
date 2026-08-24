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
import shlex
import subprocess
import sys
from collections import Counter
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

# A step that exists to reproduce what a user runs belongs to the range group even inside a
# file that otherwise pins build inputs, so the marker travels with the line rather than
# with the path.
# An explicit opt-out token rather than prose. "verify the end user's workflow" is a sentence
# someone can write, or paste, above a requirement without meaning to license a range there.
USER_WORKFLOW_MARKER = "pin-check: range-ok"

# The files expected to pin ExecuTorch, mapped to how many sites each must carry, excluding
# dev_dep_versions.yml itself. A count per file rather than just the set of files, because a site
# that loses its version stops matching the search entirely rather than reporting a mismatch, and
# two of these files carry more than one site, so a set of paths let either quietly drop one. A
# minimum rather than an exact count, since the stacked runtime-wheel change removes one README
# site and an exact count could not hold on both branches.
_EXPECTED_REQUIREMENT_SITES = {
    ".github/workflows/executorch-build-linux.yml": 2,
    ".github/workflows/executorch-test-linux.yml": 1,
    "MODULE.bazel": 1,
    "docker/MODULE.bazel.docker": 1,
    "docker/MODULE.bazel.ngc": 1,
    "justfile": 1,
    # Two: the install command and the prose sentence below it.
    "py/torch-tensorrt-executorch-runtime/README.md": 2,
    "py/torch-tensorrt-executorch-runtime/pyproject.toml": 1,
    "toolchains/ci_workspaces/MODULE.bazel.tmpl": 1,
}

# Same idea for the source commit the delegate compiles from. Five sites, not four: the
# reference-runner README names the ref as a shell default, which the old nonzero check could
# not distinguish from the four MODULE.bazel files.
_EXPECTED_COMMIT_SITES = {
    "MODULE.bazel": 1,
    "docker/MODULE.bazel.docker": 1,
    "docker/MODULE.bazel.ngc": 1,
    "toolchains/ci_workspaces/MODULE.bazel.tmpl": 1,
    "examples/executorch_reference_runner/README.md": 1,
}


def _assert_every_site_present(
    seen: "Counter[str]", expected: dict[str, int], what: str
) -> None:
    """Require every expected file to still carry at least its expected number of sites.

    A site that drops its version or commit stops matching the search rather than reporting a
    mismatch, so counting is the only way to notice it left.
    """
    short = {
        path: (count, seen.get(path, 0))
        for path, count in expected.items()
        if seen.get(path, 0) < count
    }
    unexpected = sorted(set(seen) - set(expected))
    assert not short and not unexpected, (
        f"the set of sites {what} changed.\n"
        + "".join(
            f"  {path} carries {actual} of {want} expected sites\n"
            for path, (want, actual) in sorted(short.items())
        )
        + "".join(f"  {path} is new and unaccounted for\n" for path in unexpected)
        + "A site that lost its pin does not appear in the search at all, so look for one that "
        "now names a bare reference before updating the expected counts."
    )


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, capture_output=True, text=True
    ).stdout


def _versions() -> dict:
    # Matches how setup.py and versions.py read this file, without needing yaml here.
    text = VERSIONS.read_text()
    return dict(re.findall(r'^(__\w+__): "([^"]+)"', text, re.MULTILINE))


def _wants_range(path: str, number: int) -> bool:
    # Scan upward past blanks and comment lines, so an explanatory line between the opt-out and
    # the requirement neither reclassifies the site nor fails the build for a cosmetic reason.
    # Only a comment carrying the token licenses a range; the first line of real content stops
    # the scan, so the opt-out cannot leak onto an unrelated requirement further down.
    lines = (REPO_ROOT / path).read_text().splitlines()
    for line in reversed(lines[: number - 1]):
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            return False
        if USER_WORKFLOW_MARKER in stripped:
            return True
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
    seen: Counter[str] = Counter()
    for line in _git(
        "grep", "-nI", "-E", r"executorch ?(===|==|>=|<=|~=|!=|<|>) ?[0-9]"
    ).splitlines():
        path, number, text = line.split(":", 2)
        if path == VERSIONS.name:
            continue
        expected = _expected(path, int(number), version)
        for actual in REQUIREMENT.findall(text):
            found += 1
            seen[path] += 1
            if actual != expected:
                wrong.append(f"{path}:{number} has {actual}, expected {expected}")

    assert found, "no ExecuTorch requirement found, so this test is not looking"
    _assert_every_site_present(seen, _EXPECTED_REQUIREMENT_SITES, "pinning ExecuTorch")
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


def test_derived_requirements_match_the_pin(monkeypatch) -> None:
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

    # The argument list CI actually runs, not just the string the helper derives. Dropping the
    # requirement from the setup command left every one of these tests green: the step still
    # succeeds, having installed no ExecuTorch, and the suite then skips on importorskip.
    monkeypatch.syspath_prepend(str(REPO_ROOT / "tests"))
    from ci.runner import _executorch_requirement, _setup_commands

    argv = [arg for command, _cwd in _setup_commands("executorch") for arg in command]
    assert argv.count(_executorch_requirement()) == 1, (
        "the executorch setup step does not install the pinned ExecuTorch exactly once: "
        f"{argv}"
    )

    # And the extra every documented `pip install "torch-tensorrt[executorch]"` relies on.
    # Emptying it left these tests green too. Read as source rather than imported, because
    # importing the top-level setup.py executes it.
    setup_tree = ast.parse((REPO_ROOT / "setup.py").read_text(encoding="utf-8"))
    extras = next(
        node.value
        for node in ast.walk(setup_tree)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "EXTRAS_REQUIRE" for t in node.targets)
    )
    # The published extras have to exist, or the loop below iterates nothing and deleting both
    # keys is indistinguishable from them being correct. Only these two: an unrelated future extra
    # has no reason to name ExecuTorch, and requiring it of every key made this test the one that
    # turns red when someone adds "debug".
    published = {"executorch", "all"}
    present = {key.value for key in extras.keys if isinstance(key, ast.Constant)}
    assert published <= present, (
        f"setup.py must publish the {sorted(published)} extras, but EXTRAS_REQUIRE has "
        f"{sorted(present)}. Every documented install command names one of them."
    )
    for key, value in zip(extras.keys, extras.values):
        if getattr(key, "value", None) not in published:
            continue
        named = [
            element.id
            for element in getattr(value, "elts", [])
            if isinstance(element, ast.Name)
        ]
        assert named.count("EXECUTORCH_REQUIREMENT") == 1, (
            f"extra {getattr(key, 'value', key)!r} does not reference "
            f"EXECUTORCH_REQUIREMENT exactly once: {named}"
        )

    # docgen builds its overlay with a shell substitution, so neither the literal search nor the
    # two helpers above can see it: `$(` is not a digit. Run the command it embeds and compare
    # what it prints, which fails if the line is deleted or the key is renamed.
    workflow = (REPO_ROOT / ".github/workflows/docgen.yml").read_text(encoding="utf-8")
    embedded = re.search(r'"executorch==\$\((python3 -c \'[^\']+\')\)"', workflow)
    assert embedded, (
        ".github/workflows/docgen.yml no longer pins ExecuTorch alongside the extra. It installs "
        "with --pre from the nightly channel, so without the pin it resolves through the range "
        "and takes whichever dev build is newest that day."
    )
    printed = subprocess.run(
        # The interpreter running the test, not the workflow's bare `python3`, which need not
        # have pyyaml here. The argument list is the workflow's own.
        [sys.executable, *shlex.split(embedded.group(1))[1:]],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert (
        printed == version
    ), f"docgen would install executorch=={printed}, pin says {version}"


def test_the_runner_follows_the_row_s_cuda_version(monkeypatch) -> None:
    """Call the runner and read the URL it builds, rather than matching its source.

    The executorch suite is nightly-only, and the nightly matrix runs cu132 rows as well as cu130
    ones, so a fixed channel would install a CUDA 13.0 ExecuTorch into a CUDA 13.2 job. PRs pin to
    cu130, which is why watching PR CI cannot catch it. A source-text assertion could not catch it
    either: keeping the ``os.environ.get`` line while hardcoding the URL passes one.
    """
    monkeypatch.syspath_prepend(str(REPO_ROOT / "tests"))
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

    Skipped rather than failed whenever the installed wheel is not the one the pin names, not
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
    seen: Counter[str] = Counter()
    for path in _git("grep", "-lI", "-E", 'name = "executorch"').split():
        for match in BAZEL_COMMIT.finditer((REPO_ROOT / path).read_text()):
            found += 1
            seen[path] += 1
            if match.group(1) != commit:
                wrong.append(f"{path} compiles {match.group(1)}")

    for line in _git("grep", "-nI", "-iE", NAMED_COMMIT.pattern).splitlines():
        path, number, text = line.split(":", 2)
        if path == VERSIONS.name:
            continue
        for actual in NAMED_COMMIT.findall(text):
            found += 1
            seen[path] += 1
            if actual != commit:
                wrong.append(f"{path}:{number} uses {actual}")

    assert found, "no ExecuTorch source commit found, so this test is not looking"
    _assert_every_site_present(seen, _EXPECTED_COMMIT_SITES, "naming the source commit")
    assert not wrong, f"pin says {commit}:\n  " + "\n  ".join(wrong)


@pytest.mark.unit
@pytest.mark.parametrize("setup_rc,expected", [(0, 0), (7, 7)])
def test_a_failed_setup_step_stops_the_suite(monkeypatch, setup_rc, expected):
    """A setup step that fails must fail the run, not warn and continue into pytest.

    Most of the ExecuTorch suite gates on ``pytest.importorskip``, so an install that fails makes
    those files skip while everything else passes: the run reports success precisely when the
    thing it exists to test is absent. That matters here because the pin names a nightly build,
    which the channel eventually prunes. Replacing the ``return rc`` with ``continue`` kept every
    other test in this file green, so assert on ``run_suite`` itself.
    """
    monkeypatch.syspath_prepend(str(REPO_ROOT / "tests"))
    from ci import runner

    calls: list[list[str]] = []

    class Completed:
        def __init__(self, argv):
            # The setup step is the pip install; anything else is pytest, which must not run
            # at all once setup has failed.
            self.returncode = setup_rc if "pip" in argv else 0

    def record(argv, **kwargs):
        calls.append(argv)
        return Completed(argv)

    monkeypatch.setattr(runner.subprocess, "run", record)
    suite = next(s for s in runner.SUITES if s.name == "executorch")
    rc = runner.run_suite(suite, "standard")

    assert rc == expected, f"run_suite returned {rc}, expected {expected}"
    ran_pytest = any("pytest" in " ".join(argv) for argv in calls)
    assert ran_pytest is (setup_rc == 0), (
        "pytest ran even though a setup step failed"
        if ran_pytest
        else "pytest never ran even though every setup step succeeded"
    )


@pytest.mark.unit
@pytest.mark.xfail(
    reason="uv.lock is regenerated by uv-update.yml on push to main, not by hand",
    strict=True,
)
def test_the_lockfile_records_the_same_executorch_range_as_setup_py():
    """``uv.lock`` caches what ``setup.py`` declares, so it drifts when the pin moves.

    An xfail, not a failure: ``.github/workflows/uv-update.yml`` regenerates the lock on pushes to
    main that touch ``setup.py``, and only that workflow runs ``uv sync --locked``, so a stale lock
    breaks nothing here. Regenerating it by hand is worse than leaving it -- the resolved entry and
    its hashes come from a resolver run against the nightly index, which cannot be faked in an
    editor. This exists so the drift is visible and so it turns into a real failure, via XPASS, the
    moment the lock is refreshed. The literal pin search cannot see this file: it writes
    ``specifier = ">=1.4.1,<1.5"``, with no ``executorch==`` for the grep to match.
    """
    lock = REPO_ROOT / "uv.lock"
    if not lock.is_file():
        pytest.skip("no uv.lock in this checkout")

    recorded = set(
        re.findall(
            r'\{ name = "executorch", marker = "[^"]*", specifier = "([^"]+)" \}',
            lock.read_text(encoding="utf-8"),
        )
    )
    if not recorded:
        pytest.skip("uv.lock records no executorch requirement")

    version = _versions()["__executorch_version__"]
    major, minor = _release_line(version)
    expected = f">={version},<{major}.{int(minor) + 1}"
    assert recorded == {expected}, (
        f"uv.lock records executorch {sorted(recorded)} but the pin derives {expected!r}. "
        "Run `uv lock --refresh` and commit the result."
    )


@pytest.mark.unit
def test_every_printed_install_instruction_names_the_nightly_channel():
    """Every ``[executorch]`` install instruction has to carry the nightly index.

    ExecuTorch is published only to the nightly CUDA channel, so an instruction without
    ``--extra-index-url`` resolves nothing and the user gets a bare "no matching distribution".
    The property had regressed and been re-fixed three times across this change with nothing
    asserting it, which is the signature of a property no test covers.

    Whole files rather than single lines: every one of these instructions wraps, so the extra and
    the index land on different lines and a line-oriented check sees neither together. Tracked
    files only, so a stale build directory cannot fail this.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")

    # Two shapes need the channel: an instruction naming the [executorch] extra, and the CI
    # install of a locally built torch-tensorrt wheel, whose ExecuTorch dependency resolves from
    # the same index. The second is the site that regressed most often and carries no extra.
    extra = re.compile(
        r"""torch[-_]tensorrt\[[^]]*executorch[^]]*\]|torch_tensorrt\*\.whl"""
    )
    missing = []
    for name in tracked:
        if not name or not name.endswith(
            (".py", ".sh", ".md", ".yml", ".yaml", ".rst", ".txt")
        ):
            continue
        # This file states the rule; it is not itself an instruction.
        if name == "tests/py/dynamo/executorch/test_executorch_pin.py":
            continue
        # docs/ is Sphinx output committed to the tree. Its sources live in docsrc/, which is
        # where a correction has to go, so flagging the generated copy sends the fix to a file
        # the next docs build overwrites.
        if name.startswith("docs/"):
            continue
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in extra.finditer(text):
            # The instruction is the pip invocation, so bound the window at the surrounding
            # blank-line-separated block rather than guessing a fixed number of lines.
            start = text.rfind("\n\n", 0, match.start()) + 1
            end = text.find("\n\n", match.end())
            block = text[start : end if end != -1 else len(text)]
            if "download.pytorch.org/whl/nightly" not in block:
                line = text.count("\n", 0, match.start()) + 1
                missing.append(f"{name}:{line}")

    assert not missing, (
        "these ExecuTorch install instructions do not name the nightly channel, so they "
        f"resolve no ExecuTorch at all: {missing}"
    )


@pytest.mark.unit
def test_the_pin_check_runs_in_ci():
    """This file has to be invoked by something, or its assertions never execute.

    Two suites deselect it by name so it does not need an installed ExecuTorch on a GPU runner,
    which leaves the lint job as the only path that runs it. Deleting that step is invisible
    otherwise: every test here still passes locally while nothing runs them in CI.
    """
    workflow = (REPO_ROOT / ".github/workflows/linter.yml").read_text(encoding="utf-8")
    assert (
        "test_executorch_pin.py" in workflow
    ), "no CI job invokes this file, so nothing here runs on a pull request"
    # And it needs pytest, which neither requirements.txt nor dependency-groups.lint provides.
    # Without this the step exits 1 on "No module named pytest" before running any assertion.
    assert re.search(
        r"uv pip install --system[^\n]*\bpytest\b", workflow
    ), "the job that runs this file does not install pytest, so the step cannot execute"
    assert re.search(
        r"uv pip install --system[^\n]*\bpyyaml\b", workflow
    ), "the job that runs this file does not install pyyaml, which _pinned_versions() shells out to"
