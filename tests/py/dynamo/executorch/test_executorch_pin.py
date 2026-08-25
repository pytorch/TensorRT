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
import pathlib
import re
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
# The whole specifier set, not just its first clause. Capturing up to the first comma compared
# equal on "executorch==PIN,!=PIN", a specifier that excludes the very version it appears to pin,
# and rejected the legal PEP 508 spelling with spaces around the operator.
REQUIREMENT = re.compile(
    r"executorch\s*(?:===|==|>=|<=|~=|!=|<|>)\s*[^\"'\s`,]+"
    r"(?:\s*,\s*(?:===|==|>=|<=|~=|!=|<|>)\s*[^\"'\s`,]+)*"
)


def _requirement_disagrees(actual: str, expected: str, version: str) -> str:
    """Why ``actual`` is not the pinned requirement, or an empty string if it is.

    Compares parsed specifier sets rather than matched text. Raw text equality could not see a
    clause the pattern did not capture, and treated whitespace the specification allows as drift.
    """
    from packaging.requirements import InvalidRequirement, Requirement

    try:
        parsed = Requirement(actual)
        wanted = Requirement(expected)
    except InvalidRequirement as error:
        return f"which is not a valid requirement ({error})"
    if parsed.name != wanted.name:
        return f"which names {parsed.name}, not {wanted.name}"
    if not parsed.specifier.contains(version, prereleases=True):
        return f"whose specifier excludes the pinned {version}"
    if set(parsed.specifier) != set(wanted.specifier):
        return f"expected {expected}"
    return ""


# The bazel repository puts the commit on its own line, so this one has to run against file
# contents rather than a git grep line.
BAZEL_COMMIT = re.compile(
    r'name\s*=\s*"executorch".*?commit\s*=\s*"([0-9a-f]{40})"', re.DOTALL
)

# Anywhere else the commit appears it is named on the same line, as a shell default or prose.
NAMED_COMMIT = re.compile(r"executorch[_a-z]*[^0-9a-f]*([0-9a-f]{40})", re.IGNORECASE)

PAIRING_TEST = "test_the_pinned_commit_is_the_pinned_wheels_own_source"
# A requirement takes the install-time range only when a comment carrying this exact token sits
# above it, so a site that reproduces a user's workflow can stay installable across a patch
# release. An explicit opt-out token rather than prose: "verify the end user's workflow" is a
# sentence someone can paste above a requirement without meaning to license a range there.
USER_WORKFLOW_MARKER = "pin-check: range-ok"

# A pip install of the plain torch-tensorrt wheel on a platform that has no ExecuTorch dev wheel
# carries this token. win32 is the case: the glob it installs also matches the Linux-only runtime
# wheel, so the channel scan reaches it, but ExecuTorch publishes no win32 nightly and the
# [executorch] extra is Linux-only. An explicit token rather than inference, so the exemption is
# deliberate and cannot be granted by accident to a Linux install that simply lost its index.
NO_NIGHTLY_MARKER = "pin-check: no-nightly"

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
    # One: the fenced install command. The prose sentence below it is documentation, checked for
    # pin agreement but not counted, so it cannot stand in for the command if that loses its pin.
    "py/torch-tensorrt-executorch-runtime/README.md": 1,
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


def _has_marker_above(path: str, number: int, marker: str) -> bool:
    """Whether a comment carrying ``marker`` sits directly above line ``number``.

    Scans upward past blank and comment lines; the first line of real content stops the scan, so
    the marker cannot leak from an unrelated command far above onto this one.
    """
    lines = (REPO_ROOT / path).read_text(encoding="utf-8").splitlines()
    for line in reversed(lines[: number - 1]):
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("#"):
            return False
        if marker in stripped:
            return True
    return False


def _wants_range(path: str, number: int) -> bool:
    # Only a comment carrying the token licenses a range; the first line of real content stops the
    # scan, so the opt-out cannot leak onto an unrelated requirement further down.
    return _has_marker_above(path, number, USER_WORKFLOW_MARKER)


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
    return f"executorch>={version},<{major}.{int(minor) + 1}"


# Nightly channels that actually carry the pinned ExecuTorch line. cu124 and cu128 exist on the
# index but are frozen at a CPU-only 0.5.0.dev build, so a recipe pointed at them resolves nothing
# the pin can use. Only these three serve the 1.5.0.dev wheels this change installs.
_PUBLISHED_NIGHTLY_CHANNELS = frozenset({"cu126", "cu130", "cu132"})

# Tracked files with no suffix that still carry install commands. justfile writes the nightly
# ExecuTorch install for local builds, so the printed-install walk has to read it by name.
_EXTENSIONLESS_INSTALL_FILES = frozenset({"justfile"})

# Index-URL variables whose value legitimately arrives from the CI environment and so has no
# assignment in the tree to resolve. An unresolved variable is accepted only if it is one of
# these; every other unresolved name, including a typo of a real one, fails the channel check
# rather than passing on sight. Empty today: every ExecuTorch install channels through either a
# literal nightly URL or ${EXECUTORCH_INDEX_URL}, which is assigned in executorch-build-linux.yml
# and therefore resolvable. Kept as the explicit seam a future environment-provided index goes
# through.
_ENVIRONMENT_INDEX_VARIABLES: frozenset[str] = frozenset()

# The bazel repositories annotate their pinned commit with the wheel it corresponds to, in a
# comment, because bazel fetches by commit and has no requirement string to carry. Those are the
# only comment sites that count as pins, and the commit beside them is checked separately.
_ANNOTATED_COMMIT_SITES = frozenset(
    {
        "MODULE.bazel",
        "docker/MODULE.bazel.docker",
        "docker/MODULE.bazel.ngc",
        "toolchains/ci_workspaces/MODULE.bazel.tmpl",
    }
)


def _is_commented_out(path: str, text: str) -> bool:
    """Whether this requirement sits in a comment rather than in live configuration.

    A comment is not a pin: a site could be gutted to a bare ``executorch`` while the exact pin
    lived on in a comment in the same file, which kept the per-file minimum satisfied and left the
    real requirement unpinned.
    """
    if path in _ANNOTATED_COMMIT_SITES:
        return False
    stripped = text.strip()
    # No exemption for prose. Returning False for .md/.rst/.txt defeated the threat named above,
    # because it made a comment count as a pin in exactly the files where install commands live: a
    # README install line gutted to a bare "executorch" passed as long as a decoy "# executorch==<pin>"
    # sat beside it. A "#" inside a fenced shell block is a shell comment, the same as anywhere else.
    return stripped.startswith(("#", "//", "/*", "*"))


def _without_trailing_comment(path: str, text: str) -> str:
    """``text`` up to a trailing ``#`` or ``//`` comment, unless the site annotates its pin there."""
    if path in _ANNOTATED_COMMIT_SITES:
        return text
    for marker in ("#", "//"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text


def _counts_toward_minimum(path: str, number: int) -> bool:
    """Whether a requirement at this line counts toward the per-file minimum.

    In a prose file a requirement in running text is documentation, not a live pin. Gutting the
    fenced install command to a bare ``executorch`` while a sentence below still spelled the pin
    kept the per-file count satisfied and left the command unpinned, so only a requirement inside
    a fenced code block counts for markdown. Every other file counts every live line; the
    reStructuredText sites are guarded by the install-instruction test instead.
    """
    if not path.endswith(".md"):
        return True
    fenced = False
    for current, content in enumerate(
        (REPO_ROOT / path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if current == number:
            return fenced
        if content.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
    return False


def _strip_whole_line_comments(text: str) -> str:
    """Blank out whole-line ``#`` comments, preserving line count so DOTALL spans stay aligned.

    The Bazel commit match walks from ``name = "executorch"`` to the ``commit = "..."`` line with
    DOTALL, so a commit commented out and replaced by a live ``branch = "main"`` still matched the
    commented copy and the build floated to a branch while this test stayed green.
    """
    return "\n".join(
        "" if line.lstrip().startswith("#") else line for line in text.splitlines()
    )


def _resolve_shell_assignment(text: str, variable: str, before: int) -> str | None:
    """The last literal ``VAR=...`` assignment of ``variable`` in ``text`` before offset ``before``.

    An install that channels through ``--extra-index-url "${VAR}"`` proves nothing on its own: the
    value is whatever ``VAR`` was last set to. Repointing that assignment at PyPI, or dropping its
    ``nightly/`` segment, left the install counted as channelled while it resolved nothing. Resolve
    the assignment so the channel is checked where it is actually set. Returns ``None`` when no
    assignment is found, meaning the value comes from the environment and cannot be resolved here.
    """
    assignment = re.compile(
        rf"""^\s*(?:export\s+)?{re.escape(variable)}=["']?([^"'\n]*)""", re.MULTILINE
    )
    resolved = None
    for match in assignment.finditer(text):
        if match.start() >= before:
            break
        resolved = match.group(1)
    return resolved


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
        if _is_commented_out(path, text):
            # A comment is not a pin. Counting raw matches meant a site could be gutted to a bare
            # "executorch" while the exact pin lived on in a comment in the same file, keeping the
            # per-file minimum satisfied.
            continue
        expected = _expected(path, int(number), version)
        # A trailing comment is not a pin either. Skipping whole-line comments was not enough: a
        # live install gutted to a bare "executorch" with a decoy "# executorch==<pin>" after it on
        # the same line kept the per-file count satisfied and left the install unpinned. The
        # annotated commit sites write their pin as a whole-line comment, which is handled above,
        # so nothing legitimate is lost here.
        counts = _counts_toward_minimum(path, int(number))
        for actual in REQUIREMENT.findall(_without_trailing_comment(path, text)):
            found += 1
            if counts:
                seen[path] += 1
            reason = _requirement_disagrees(actual, expected, version)
            if reason:
                wrong.append(f"{path}:{number} has {actual}, {reason}")

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
    # Anchor to a live line: leading whitespace only, no "#". A commented-out install still
    # carries the pattern, so a plain search stayed green when the whole step was disabled.
    embedded = re.search(
        r'^[ \t]*"executorch==\$\((python3 -c \'[^\']+\')\)"',
        workflow,
        re.MULTILINE,
    )
    assert embedded, (
        ".github/workflows/docgen.yml no longer pins ExecuTorch alongside the extra on a live "
        "line. It installs with --pre from the nightly channel, so without the pin it resolves "
        "through the range and takes whichever dev build is newest that day."
    )
    # Compared as text, not executed. Running it meant whatever that line said got executed on
    # every pull request: rewriting the one-liner to write a file left the test green and the file
    # written. It has to read __executorch_version__ out of dev_dep_versions.yml and print nothing
    # else, which is the property that makes the shell substitution equal the pin.
    command = embedded.group(1)
    reads_the_pin = re.fullmatch(
        r"""python3 -c 'import yaml;print\(yaml\.safe_load\(open\("dev_dep_versions\.yml"\)\)"""
        r"""\["__executorch_version__"\]\)'""",
        command,
    )
    assert reads_the_pin, (
        "the docgen pin no longer reads __executorch_version__ out of dev_dep_versions.yml, so "
        f"what it installs is no longer the pin: {command}"
    )


_RUNTIME_SETUP_PY = "py/torch-tensorrt-executorch-runtime/setup.py"


def _runtime_install_requires() -> dict[str, str]:
    """The runtime wheel's ``install_requires`` entries, each mapped to its source text.

    Read as source, not imported: importing this setup.py runs a Bazel build. The values are
    f-strings built at build time from the installed distributions, so the source segment is what
    the check compares, not a resolved string.
    """
    tree = ast.parse((REPO_ROOT / _RUNTIME_SETUP_PY).read_text(encoding="utf-8"))
    source = (REPO_ROOT / _RUNTIME_SETUP_PY).read_text(encoding="utf-8")
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup"
    )
    requires = next(
        keyword.value for keyword in call.keywords if keyword.arg == "install_requires"
    )
    entries = {}
    for element in requires.elts:
        text = ast.get_source_segment(source, element)
        distribution = re.match(r'f?"([A-Za-z0-9_.-]+)', text)
        if distribution:
            entries[distribution.group(1)] = text
    return entries


def test_the_runtime_wheel_pins_executorch_to_the_public_pin() -> None:
    """The runtime wheel's own ExecuTorch requirement has to pin the pinned version, stripped.

    This is the one requirement whose native code is compiled against the ExecuTorch runtime, so a
    wheel that requires a different ExecuTorch than it was built against loads a mismatched runtime.
    The literal search above cannot see it: setup.py builds the string from the installed
    distribution, so ``executorch==`` is followed by a brace, not a digit. Loosening it to a bare
    ``executorch`` left every other test green.
    """
    entries = _runtime_install_requires()
    assert (
        "executorch" in entries
    ), f"{_RUNTIME_SETUP_PY} install_requires no longer pins executorch: {sorted(entries)}"
    # The value is built from installed_version("executorch"), the same source torch and
    # torch-tensorrt use, and stripped of its local label the same way. Compare the source text so
    # a bare name, a hardcoded version, or a different version source is rejected.
    assert (
        entries["executorch"] == 'f"executorch=={public_version(executorch_version)}"'
    ), (
        f"{_RUNTIME_SETUP_PY} must pin executorch to public_version(executorch_version), so the "
        f"wheel requires the ExecuTorch it was compiled against, but it declares "
        f"{entries['executorch']}"
    )


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


def _load_utils_channel_helpers(fake_cuda: str | None):
    """Exec ``executorch_install_channel``/``executorch_install_command`` with a stub torch.

    ``py/torch_tensorrt/_utils.py`` imports ``tensorrt`` and the built ``torch_tensorrt``, neither
    installed on the lint runner, so it cannot be imported here. Extract just the two functions and
    exec them against a fake ``torch`` whose ``version.cuda`` is ``fake_cuda``, which is all they
    read. This keeps the test on the source that ships rather than a copy of its logic.
    """
    source = (REPO_ROOT / "py/torch_tensorrt/_utils.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {"executorch_install_channel", "executorch_install_command"}
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {f.name for f in functions} == wanted, (
        "py/torch_tensorrt/_utils.py must define executorch_install_channel and "
        f"executorch_install_command; found {sorted(f.name for f in functions)}"
    )

    class _Version:
        cuda = fake_cuda

    namespace: dict[str, object] = {"torch": type("torch", (), {"version": _Version})}
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, "<utils-extract>", "exec"), namespace)  # noqa: S102
    return (
        namespace["executorch_install_channel"],
        namespace["executorch_install_command"],
    )


def test_the_executorch_install_message_names_the_torch_channel() -> None:
    """The three ExecuTorch install messages derive their channel from the running torch.

    ExecuTorch ships a distinct wheel per CUDA channel, so a message that hardcodes cu130 tells a
    CUDA 13.2 user to install a CUDA 13.0 build. The messages route through
    ``executorch_install_command`` so the channel is computed once from ``torch.version.cuda``.
    A source-text assertion cannot see the value the format string produces, so exercise the helper
    across both published 13.x channels and the no-CUDA fallback.
    """
    channel, command = _load_utils_channel_helpers("13.2")
    assert channel() == "cu132"
    message = command()
    assert "download.pytorch.org/whl/nightly/cu132" in message, message
    # Raised from inside an installed torch_tensorrt, so pip treats the requirement as satisfied
    # and exits 0 without the extra unless --upgrade forces a re-resolve. --pre selects the dev pin.
    assert "--upgrade" in message and "--pre" in message, message

    channel_130, command_130 = _load_utils_channel_helpers("13.0")
    assert channel_130() == "cu130"
    assert "nightly/cu130" in command_130()

    # No CUDA build reports a placeholder rather than a channel the user cannot install from.
    channel_none, command_none = _load_utils_channel_helpers(None)
    assert channel_none() == "cuXYZ"
    assert "nightly/cuXYZ" in command_none()

    # Every message site has to delegate to the shared command, checked one site at a time. A
    # whole-file substring pass cannot do that: with two sites in a file, "the helper is called
    # somewhere" is satisfied by whichever site is still correct, so hardcoding a channel in the
    # other one goes unnoticed. Worse, forbidding the literal cu130 only catches hardcoding the
    # RIGHT channel, while cu126 or cu132 sail through and misdirect a user on that CUDA build.
    #
    # A "site" is a raise of ImportError whose message mentions installing ExecuTorch. Each one must
    # obtain its command by calling the helper rather than by carrying a literal index URL.
    for path in (
        "py/torch_tensorrt/_compile.py",
        "py/torch_tensorrt/executorch/__init__.py",
    ):
        text = (REPO_ROOT / path).read_text(encoding="utf-8")
        tree = ast.parse(text)
        sites = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue
            rendered = ast.unparse(node)
            if (
                "download.pytorch.org" not in rendered
                and "install" not in rendered.lower()
            ):
                continue
            if "executorch" not in rendered.lower():
                continue
            calls_helper = any(
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == "executorch_install_command"
                for inner in ast.walk(node)
            )
            if not calls_helper:
                # Only complain when the site actually spells an index URL itself. A raise that
                # names no channel at all is a different message, not a hardcoded one.
                assert "download.pytorch.org" not in rendered, (
                    f"{path} raises an ExecuTorch install message that spells its own index URL "
                    "instead of calling executorch_install_command(), so the channel it prints "
                    "cannot follow the running torch:\n"
                    f"{rendered}"
                )
                continue
            sites += 1
        assert sites, (
            f"{path} has no ExecuTorch install message built with executorch_install_command(); "
            "either a site was removed or this scan no longer recognises it"
        )


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
        source = _strip_whole_line_comments((REPO_ROOT / path).read_text())
        for match in BAZEL_COMMIT.finditer(source):
            found += 1
            seen[path] += 1
            if match.group(1) != commit:
                wrong.append(f"{path} compiles {match.group(1)}")

    for line in _git("grep", "-nI", "-iE", NAMED_COMMIT.pattern).splitlines():
        path, number, text = line.split(":", 2)
        if path == VERSIONS.name:
            continue
        # A commit in a comment is not a pin. Grepping raw lines let EXECUTORCH_REF float to a
        # branch with the real SHA left behind in a "# ..." comment on the same file, which kept
        # this scan green. The Bazel walk above already strips comments; do the same here.
        if _is_commented_out(path, text):
            continue
        for actual in NAMED_COMMIT.findall(_without_trailing_comment(path, text)):
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
def test_the_lockfile_records_the_same_executorch_range_as_setup_py():
    """``uv.lock`` caches what ``setup.py`` declares, so it drifts when the pin moves.

    A stale lock breaks nothing here, because only ``uv-update.yml`` runs ``uv sync --locked``,
    and its resolved hashes come from a resolver run against the nightly index that cannot be
    faked in an editor. So this accepts two states: the range the pin derives, and a range that
    predates the pin.

    It used to be a strict xfail, which meant the moment anyone refreshed the lock the assertion
    passed and pytest reported that pass as a failure. The lint step runs this file on every pull
    request, so that would have turned the lint job red repo-wide for a file none of those pull
    requests touched. Nor is the lock only machine-generated: it was hand-refreshed twice inside
    ordinary version-bump changes on 2026-08-23.

    The literal pin search cannot see this file: it writes ``specifier = ">=1.4.1,<1.5"``, with no
    ``executorch==`` for the grep to match.
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
    if recorded == {expected}:
        return

    # Behind the pin is the expected resting state until the lock is regenerated. Ahead of it is
    # not: that means the lock names an ExecuTorch this repository does not pin.
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    # Ahead means the range's own lower bound is above the pin. Probing the specifier with sample
    # versions was fragile in both directions: an upper-bound test missed an open-ended ">=1.7",
    # and a low sentinel called the ordinary behind-the-pin state a failure.
    pinned = Version(version)
    ahead = [
        entry
        for entry in sorted(recorded)
        if any(
            clause.operator in {">=", ">", "==", "~=", "==="}
            and Version(clause.version.rstrip(".*") or "0") > pinned
            for clause in SpecifierSet(entry)
        )
    ]
    assert not ahead, (
        f"uv.lock records executorch {ahead}, which is ahead of the pinned {version}. The pin "
        "derives "
        + repr(expected)
        + ", so run `uv lock --refresh` and commit the result."
    )


_INSTALL_INVOCATION = re.compile(
    r"(?:python[0-9.]*\s+-m\s+pip|uv\s+pip|\bpip)\s+(?:install|wheel)\b"
)


def _blank_comments_preserving_length(block: str) -> str:
    """``block`` with comment text replaced by spaces, keeping every offset and newline in place.

    Used only to locate pip keywords without a ``# pip install ...`` in a comment starting a false
    invocation. Length is preserved so an offset into the original block indexes the same character
    here. Whole-line ``#`` comments blank entirely; a trailing `` #`` comment blanks from the hash,
    but a ``#cu130`` URL fragment (no preceding space) is left intact.
    """
    out = []
    for physical in block.splitlines(keepends=True):
        newline = "\n" if physical.endswith("\n") else ""
        body = physical[:-1] if newline else physical
        if body.lstrip().startswith("#"):
            body = " " * len(body)
        else:
            hash_at = re.search(r"(?:^|\s)#", body)
            if hash_at:
                cut = hash_at.start()
                body = body[:cut] + " " * (len(body) - cut)
        out.append(body + newline)
    return "".join(out)


def _install_invocation_window(block: str, match_offset: int) -> str:
    """The slice of ``block`` belonging to the one pip/uv invocation that owns the match.

    The channel and the requirement it channels have to belong to the *same* invocation. A window
    bounded only at blank lines was too wide: it spanned every step of a contiguous YAML job and
    every line of a shell if/else, so a ``--extra-index-url`` from a neighbouring ``pip install``,
    an ``echo``, or prose satisfied the check for an install that carried none of its own. Two real
    holes this closed: the win32 branch of ``install-torch-tensorrt.sh`` borrowed the else branch's
    URL, and the ``.[executorch]`` step in ``docgen.yml`` borrowed the *Install base deps* step's.

    An invocation runs from its ``pip``/``uv pip`` keyword to the next such keyword in the block, or
    to the block's end. That single boundary spans a backslash-continued shell command, a YAML
    ``run:`` body and a Python error message built from adjacent string fragments alike, because
    none of those start a second invocation between the keyword and the URL. ``match_offset`` is a
    ``block``-relative offset into the original (un-stripped) text, so a requirement that appears
    twice in one block resolves to its own invocation rather than the first copy's. When no
    invocation keyword precedes the match the whole block is returned, leaving non-install prose
    matches to the caller's other filters.
    """
    scan = _blank_comments_preserving_length(block)
    starts = [m.start() for m in _INSTALL_INVOCATION.finditer(scan)]
    preceding = [s for s in starts if s <= match_offset]
    if not preceding:
        return block
    begin = preceding[-1]
    following = [s for s in starts if s > match_offset]
    finish = following[0] if following else len(block)
    return block[begin:finish]


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
    # The local-path spelling counts too. Matching only the named-distribution form left the four
    # sites that write "pip install .[executorch]" unguarded: the nightly index could be deleted
    # from all four with this test green.
    # A fourth shape: a direct "executorch==<pin>" or "executorch>=<pin>" in a pip command. The
    # runtime README build recipe installs ExecuTorch this way, and its nightly index could be
    # deleted with this test green because none of the three shapes above match a bare
    # distribution name. Gated on a pip context below so a requirement in pyproject or a comment
    # is not mistaken for an install instruction.
    # The built-wheel shape covers both the plain "torch_tensorrt*.whl" and the runtime wheel
    # "torch_tensorrt_executorch_runtime-*.whl": the latter's install_requires names the same
    # nightly ExecuTorch, so its documented install needs the channel too, and matching only the
    # plain glob left the runtime README's Use command unscanned. It is gated on a real
    # "pip install" without "--no-deps" below, so naming the file in an "ls" or a heredoc, or a
    # "--no-deps" install that fetches nothing, is not mistaken for a dependency-resolving install.
    # A fifth shape: a bare "pip install executorch" with no version operator. It resolves the
    # stable 1.4.1 from PyPI, the version this change moves away from, and carries no operator so
    # the shapes above miss it. Matched as a standalone distribution token and gated on a
    # pip-install context below, so the word in a path, an import, a filename, or prose is not
    # mistaken for an install. Like the named-distribution form it needs both the channel and
    # --pre, since a bare name without --pre picks the stable release even off the nightly index.
    extra = re.compile(
        r"""torch[-_]tensorrt\[[^]]*executorch[^]]*\]"""
        r"""|(?<![\w./-])\.\[[^]]*executorch[^]]*\]"""
        r"""|torch_tensorrt(?:_executorch_runtime-)?\*\.whl"""
        r"""|(?<![\w./-])executorch(?:_[a-z]+)?\s*(?:==|>=)"""
        r"""|(?<![\w./\[-])executorch(?![\w./\[=<>_-])"""
    )
    missing = []
    for name in tracked:
        base = name.rsplit("/", 1)[-1]
        # Select by suffix, plus a few extensionless files that carry install commands. The
        # justfile in particular writes a real "uv pip install ... executorch==<pin>" with its
        # nightly index; filtering on suffix alone never read it, so both the index and the pin
        # could be dropped from it with this test green.
        if not base or not (
            name.endswith((".py", ".sh", ".md", ".yml", ".yaml", ".rst", ".txt"))
            or base in _EXTENSIONLESS_INSTALL_FILES
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
            line = text.count("\n", 0, match.start()) + 1
            # Bound the window at the surrounding blank-line-separated block first, then narrow to
            # the single pip invocation that owns the match. The block alone was too wide: a
            # contiguous YAML job or a shell if/else is one block, so a --extra-index-url, --pre or
            # channel from a neighbouring command satisfied an install that carried none itself.
            block_start = text.rfind("\n\n", 0, match.start()) + 1
            block_end = text.find("\n\n", match.end())
            block = text[block_start : block_end if block_end != -1 else len(text)]
            block = _install_invocation_window(block, match.start() - block_start)
            # Strip comments after windowing: a shell comment naming the channel or --pre is prose,
            # not an argument pip sees, so a gutted install line with a decoy comment beside it must
            # not satisfy the check. Whole-line comments drop entirely; a trailing "#" comment is
            # cut, but not the "#cu130" fragment of a URL, which carries no space.
            block = "\n".join(
                re.sub(r"(?:^|\s)#.*$", "", bl)
                for bl in block.splitlines()
                if not bl.lstrip().startswith("#")
            )
            # A plain torch-tensorrt wheel install on a platform with no ExecuTorch dev wheel is
            # exempt. win32 installs a glob that also matches the Linux-only runtime wheel, so the
            # scan reaches it, but ExecuTorch publishes no win32 nightly. The marker has to sit
            # directly above the invocation, and a separate test keeps it inside a win32 guard, so a
            # gutted Linux install cannot claim it.
            if _has_marker_above(name, line, NO_NIGHTLY_MARKER):
                continue
            is_direct_install = bool(
                re.fullmatch(r"executorch(?:_[a-z]+)?\s*(?:==|>=)", match.group(0))
            )
            # A bare "executorch==" only counts as an instruction inside a pip command. Anywhere
            # else it is a dependency declaration or prose, guarded by other tests.
            if is_direct_install:
                if not re.search(r"\bpip\s+(?:install|wheel)\b", block):
                    continue
                # The version after the operator has to be a literal. "executorch==${OLD}" or
                # "executorch==$(...)" clears the pip-context gate yet pins nothing: pip installs
                # whatever the expansion yields, which floats off the pin. The one legitimate
                # non-literal is docgen's, which reads __executorch_version__ out of
                # dev_dep_versions.yml; a separate test proves that command equals the pin.
                after = text[match.end() : match.end() + 80].lstrip()
                if not after[:1].isdigit() and not after.startswith(
                    "$(python3 -c 'import yaml;"
                ):
                    missing.append(
                        f"{name}:{line} installs executorch at a non-literal version, which pins "
                        "nothing: pip resolves whatever the expansion yields off the nightly index"
                    )
                    continue
            is_bare_name = match.group(0) == "executorch"
            if is_bare_name:
                # A bare distribution name is an install only inside a pip command; the same word
                # in a path, an import, or prose is not.
                if not re.search(r"\bpip\s+(?:install|wheel)\b", block):
                    continue
                # The word also appears in prose that shares a block with a real
                # "torch_tensorrt[executorch]" install, so require the bare token to sit on the pip
                # command line itself before treating it as the install target. A bare target there
                # pins nothing even with --pre and the channel: pip resolves the newest nightly, not
                # this pin. Nothing in the tree installs executorch bare, so it is always a defect.
                command_line = text.splitlines()[line - 1]
                if re.search(r"\bpip\s+(?:install|wheel)\b", command_line):
                    missing.append(
                        f"{name}:{line} installs executorch by bare name, which pins nothing: pip "
                        "resolves the newest nightly rather than the pinned version"
                    )
                continue
            is_built_wheel = bool(
                re.fullmatch(
                    r"torch_tensorrt(?:_executorch_runtime-)?\*\.whl", match.group(0)
                )
            )
            # This glob names a local file, so it pulls the nightly ExecuTorch dependency only in a
            # "pip install" that resolves dependencies. Naming the file elsewhere (an "ls", a
            # heredoc, "pip wheel", or a "--no-deps" install) fetches no ExecuTorch, so no channel
            # applies.
            if is_built_wheel and (
                not re.search(r"\bpip\s+install\b", block)
                or re.search(r"(?:^|\s)--no-deps(?:\s|$)", block)
            ):
                continue
            channel = re.search(
                r"download\.pytorch\.org/whl/nightly(?:/(cu\d+))?", block
            )
            # CI passes the channel through a variable rather than a literal URL. Capture the
            # variable name so its assignment can be resolved: accepting the reference on sight let
            # the assignment be repointed at PyPI, or stripped of its nightly segment, with the
            # install still counted as channelled.
            variable_index = re.search(
                r"--(?:extra-index-url|index-url)\s+\"?\$\{?([A-Za-z_][A-Za-z0-9_]*)",
                block,
            )
            if not channel and variable_index:
                # Resolve the variable's last assignment before this install and check the channel
                # there. An unresolved variable is accepted only when it is a known workflow input
                # whose value arrives from the CI environment; every other unresolved name, a typo
                # among them, fails rather than passing on sight.
                assignment = _resolve_shell_assignment(
                    text, variable_index.group(1), match.start()
                )
                if assignment is None:
                    if variable_index.group(1) in _ENVIRONMENT_INDEX_VARIABLES:
                        continue
                    missing.append(
                        f"{name}:{line} channels through ${{{variable_index.group(1)}}}, which has "
                        "no assignment in the tree and is not a known CI index input"
                    )
                    continue
                channel = re.search(
                    r"download\.pytorch\.org/whl/nightly(?:/(cu\d+))?", assignment
                )
                if not channel:
                    missing.append(
                        f"{name}:{line} installs from ${{{variable_index.group(1)}}}, set to "
                        f"{assignment!r}, which is not the nightly channel"
                    )
                    continue
            if not channel:
                missing.append(f"{name}:{line} names no nightly channel")
                continue
            # A substring proves a string sits nearby, not that it resolves anything. Rewriting
            # every channel in the tree to a nonexistent cu999 left this green. Not compared
            # against __cuda_version__: the runtime error messages derive their channel from the
            # user's torch build, and the documented recipes name a concrete published channel that
            # carries the pinned ExecuTorch, so a literal cuXYZ here is checked only for being one
            # the project publishes.
            suffix = channel.group(1)
            if suffix and suffix not in _PUBLISHED_NIGHTLY_CHANNELS:
                missing.append(
                    f"{name}:{line} installs from nightly/{suffix}, which the project does not "
                    f"publish for; expected one of {sorted(_PUBLISHED_NIGHTLY_CHANNELS)}"
                )
            # The named-distribution form needs --pre. "torch-tensorrt[executorch]" with no
            # version pin resolves to the stable PyPI wheel, which carries no executorch extra at
            # all, so the command exits 0 with a warning and installs nothing the feature needs.
            # The ".[executorch]" and built-wheel forms already pin executorch to a dev version,
            # which enables prerelease selection on their own, so they do not need it.
            named_distribution = re.fullmatch(
                r"torch[-_]tensorrt\[[^]]*executorch[^]]*\]", match.group(0)
            )
            if named_distribution and not re.search(r"(?:^|\s)--pre(?:\s|$)", block):
                missing.append(
                    f"{name}:{line} installs {match.group(0)} without --pre, so pip resolves the "
                    "stable release with no executorch extra rather than the nightly prerelease"
                )

    assert not missing, (
        "these ExecuTorch install instructions do not name the nightly channel, so they "
        f"resolve no ExecuTorch at all: {missing}"
    )


@pytest.mark.unit
def test_the_no_nightly_marker_only_exempts_a_win32_install():
    """The ``no-nightly`` exemption is legitimate only where no ExecuTorch dev wheel exists.

    The channel scan skips an install carrying ``pin-check: no-nightly`` above it. That is correct
    for win32, whose wheel glob also matches the Linux-only runtime wheel while ExecuTorch ships no
    win32 nightly. Without this test the marker is a blanket silencer: strip the index from the
    Linux install, paste the marker above it, and the channel scan stays green. Requiring the marker
    to sit inside a ``win32`` platform guard keeps the exemption tied to the one case it describes.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")

    misplaced = []
    for name in tracked:
        if not name or name == "tests/py/dynamo/executorch/test_executorch_pin.py":
            continue
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if NO_NIGHTLY_MARKER not in text:
            continue
        lines = text.splitlines()
        control = re.compile(r"^\s*(?:if\b|elif\b|else\b|fi\b)")
        for index, content in enumerate(lines):
            if NO_NIGHTLY_MARKER not in content:
                continue
            # The exempted install sits just below the marker, so the branch it lives in is the
            # nearest control-flow keyword above it. Requiring that keyword to be the win32 guard
            # ties the exemption to the one platform it describes: a marker pasted onto a Linux
            # "else" install resolves to that "else", not to "if ... win32", and is rejected.
            branch = next(
                (lines[j] for j in range(index - 1, -1, -1) if control.match(lines[j])),
                "",
            )
            if "win32" not in branch:
                misplaced.append(
                    f"{name}:{index + 1} carries {NO_NIGHTLY_MARKER!r} outside a win32 branch, "
                    "so it would exempt a Linux install that simply lost its index"
                )

    assert not misplaced, (
        "the no-nightly exemption is only valid inside a win32 branch: " f"{misplaced}"
    )


@pytest.mark.unit
def test_the_pin_check_runs_in_ci():
    """This file has to be invoked by something, or its assertions never execute.

    Two suites deselect it by name so it does not need an installed ExecuTorch on a GPU runner,
    which leaves the lint job as the only path that runs it. Deleting that step is invisible
    otherwise: every test here still passes locally while nothing runs them in CI.
    """
    # Parse the workflow and assert inside the owning job. Searching the file as one blob could
    # not tell which job it was reading, so an identical install line in a sibling job that has no
    # pin check satisfied it, and deleting the real one stayed green. A commented-out step also
    # vanishes from the parse, where a text search still finds it.
    import yaml

    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/linter.yml").read_text(encoding="utf-8")
    )
    # The workflow must actually run on pull requests. PyYAML reads the unquoted "on" key as the
    # boolean True (the YAML 1.1 "Norway problem"), so accept either spelling, then require a
    # pull_request trigger. Reducing "on:" to workflow_dispatch left every string in place while
    # the workflow never fired on a pull request.
    triggers = workflow.get("on", workflow.get(True))
    trigger_names = set(triggers) if isinstance(triggers, (dict, list)) else {triggers}
    assert "pull_request" in trigger_names, (
        f"linter.yml triggers on {sorted(map(str, trigger_names))}, not pull_request, so the pin "
        "check never runs when a pull request changes the pin"
    )
    # A paths filter on the trigger would keep the workflow from firing on a pin change outside
    # those paths, leaving every assertion below green while nothing ran.
    pull_request = triggers.get("pull_request") if isinstance(triggers, dict) else None
    if isinstance(pull_request, dict):
        for path_filter in ("paths", "paths-ignore"):
            assert path_filter not in pull_request, (
                f"linter.yml narrows the pull_request trigger with {path_filter}, so a change to "
                "the pin outside those paths would not run this check"
            )
    # Match a live pytest invocation with pytest as the command word, not the filename anywhere
    # in the script. A plain substring test was satisfied by a comment; an "anything before
    # pytest" test was satisfied by "echo python3 -m pytest", which prints the command and runs
    # nothing.
    invocation = re.compile(
        rf"^\s*(?:python[0-9.]*\s+-m\s+)?pytest\b[^\n]*{re.escape(pathlib.Path(__file__).name)}",
        re.MULTILINE,
    )
    owning = [
        (name, job, step)
        for name, job in workflow["jobs"].items()
        for step in job.get("steps", [])
        if invocation.search(step.get("run") or "")
    ]
    assert owning, "no CI job invokes this file, so nothing here runs on a pull request"
    name, job, step = owning[0]

    # A falsy condition disables the step or the whole job while leaving every string in place, so
    # check both. GitHub treats a bare "false", "${{ false }}" and any always-false expression the
    # same way, so restrict each to the small set of conditions that can actually be true.
    live_conditions = {"always()", "success()", "success() || failure()"}
    step_condition = str(step.get("if", "always()"))
    assert (
        step_condition in live_conditions
    ), f"the pin check step in {name} runs under {step_condition!r}, which may never be true"
    job_condition = str(job.get("if", "always()"))
    assert (
        job_condition in live_conditions
    ), f"job {name} runs under {job_condition!r}, so the pin check may never dispatch"

    # Compared as text, not executed. This used to run the step's own shell body under bash
    # against a stub, which meant whatever that body said got executed on every pull request:
    # appending an "echo ... >> /tmp/marker" line to the step left this test GREEN and the marker
    # written twice. That is the same defect this file already fixed for the docgen one-liner, and
    # the reasoning there applies here. Assert the shape of the command instead: it has to invoke
    # this file under pytest, with no flag that could deselect or neuter the run.
    script = step["run"]
    assert re.search(
        r"python3 -m pytest\s+\S*tests/py/dynamo/executorch/test_executorch_pin\.py",
        script,
    ), f"the pin check step in {name} does not run this file under pytest: {script!r}"
    for forbidden, why in (
        ("--collect-only", "collection alone never runs an assertion"),
        ("--co", "collection alone never runs an assertion"),
        ("|| true", "the exit status is discarded, so a failure cannot fail the job"),
        ("set +e", "the exit status is discarded, so a failure cannot fail the job"),
        ("continue-on-error", "a failure cannot fail the job"),
        ("--deselect", "a deselected test cannot fail"),
        ("-k ", "a keyword filter can silently select nothing"),
        ("exit 0", "the step reports success regardless of the result"),
    ):
        assert (
            forbidden not in script
        ), f"the pin check step in {name} contains {forbidden!r}, so {why}"

    assert not step.get(
        "continue-on-error"
    ), f"the pin check step in {name} is continue-on-error, so a failure cannot fail the job"
    assert not job.get(
        "continue-on-error"
    ), f"job {name} is continue-on-error, so a failed pin check cannot fail the workflow"

    # pytest and pyyaml must be installed by an earlier step of the SAME job: neither
    # requirements.txt nor dependency-groups.lint carries them, and without them the step exits 1
    # on "No module named pytest" before running any assertion.
    steps = job["steps"]
    # Comment-stripped: a commented-out "uv pip install ... pytest pyyaml" still matched the raw
    # text while installing nothing, so the step would die on a missing import at runtime.
    earlier = "\n".join(
        re.sub(r"(?:^|\s)#.*$", "", line)
        for step_before in steps[: steps.index(step)]
        for line in (step_before.get("run") or "").splitlines()
        if not line.lstrip().startswith("#")
    )
    for package in ("pytest", "pyyaml"):
        assert re.search(
            rf"uv pip install --system[^\n]*\b{package}\b", earlier
        ), f"job {name} does not install {package} before the pin check, so the step cannot run"


@pytest.mark.unit
def test_the_pairing_check_survives_the_gpu_lane_deselection() -> None:
    """The one test here that needs a real ExecuTorch must not be deselected with the rest.

    Every other test in this file is a source-consistency check, so the executorch tier deselects
    the whole module by name to avoid paying for them twice. ``-k`` matches the module name in the
    test id, so a bare ``not test_executorch_pin`` drops the pairing check too, and that check
    only means anything where ExecuTorch is installed, which is nowhere the lint job runs.

    Two routes reach this tier: the nightly manifest suite in ``tests/ci/suites.py``, and
    ``executorch-test-linux.yml``, which installs the pinned wheel and runs on pull requests once
    the runtime build succeeds. Both go through one of the two keyword expressions checked here.
    """
    # Run pytest's own collection under each expression rather than grepping for the name. A
    # string test passes on "and" in place of "or", which collects nothing at all, and on the
    # name surviving only in a comment. Both leave the pairing check unreachable.
    module = pathlib.Path(__file__).name
    for path, pattern in (
        ("tests/ci/suites.py", r'keyword=\(\s*(?:#[^\n]*\n\s*)*"([^"]+)"'),
        # Anchored on the executorch junitxml name, because the file passes -k in several
        # functions and the first match belongs to a different tier.
        (
            "tests/py/utils/ci_helpers.sh",
            r'executorch_tests_results[^\n]*?-k "([^"]+)"',
        ),
    ):
        text = (REPO_ROOT / path).read_text(encoding="utf-8")
        found = re.search(pattern, text)
        assert (
            found
        ), f"{path} no longer passes a single -k expression this test can read"
        keyword = found.group(1)
        assert "not test_executorch_pin" in keyword, (
            f"{path} no longer deselects this module, so the source-consistency checks here "
            "would run twice"
        )
        selected = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(pathlib.Path(__file__).parent),
                "--collect-only",
                "-q",
                "--noconftest",
                "-p",
                "no:cacheprovider",
                "-o",
                "addopts=",
                "-k",
                keyword,
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        ).stdout
        assert f"{module}::{PAIRING_TEST}" in selected, (
            f"{path} runs pytest with -k {keyword!r}, which does not select {PAIRING_TEST}, so "
            "the only check that needs a real ExecuTorch installed runs nowhere"
        )
        others = [
            line
            for line in selected.splitlines()
            if module in line and PAIRING_TEST not in line
        ]
        assert not others, (
            f"{path} selects {len(others)} other tests from this module, which the lane "
            f"deselects deliberately: {others[:2]}"
        )

    # Proving the -k expression selects the pairing test says nothing about whether either route
    # is actually wired to run it. Replacing the workflow's `trt_tier_executorch` call with `echo
    # skipped`, or pointing the executorch suite at a lane name no runner requests, both leave the
    # checks above green while the test runs nowhere. So assert each route reaches the tier.
    import yaml

    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/executorch-test-linux.yml").read_text(
            encoding="utf-8"
        )
    )
    scripts = [
        step.get("with", {}).get("script", "")
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
    ] + [job.get("with", {}).get("script", "") for job in workflow["jobs"].values()]
    # The call must not discard its own exit status. "trt_tier_executorch || true", a trailing
    # ";", "&" or a pipe each let the tier fail while the lane stays green, so reject any status
    # operator after the call on its line.
    tier_calls = [
        match
        for script in scripts
        for match in re.finditer(
            r"^\s*trt_tier_executorch\b([^\n]*)", script, re.MULTILINE
        )
    ]
    assert tier_calls, (
        "executorch-test-linux.yml no longer calls trt_tier_executorch, so the pairing check "
        "never runs on the GPU lane even though its -k expression would select it"
    )
    assert any(not re.search(r"[|;&]", call.group(1)) for call in tier_calls), (
        "every trt_tier_executorch call in executorch-test-linux.yml discards its exit status "
        "with a pipe, ';', '&' or '|| true', so a pairing failure cannot fail the lane"
    )

    # The manifest route: the executorch suite must exist and target a lane a runner requests.
    # A typo in its lane tuple silently drops it from every matrix, which the suite-name check
    # above cannot see.
    import importlib

    suites = importlib.import_module("tests.ci.suites")
    executorch_suite = next((s for s in suites.SUITES if s.name == "executorch"), None)
    assert executorch_suite is not None, (
        "tests/ci/suites.py no longer defines an 'executorch' suite, so the manifest route to the "
        "pairing check is gone"
    )
    assert "nightly" in executorch_suite.lanes, (
        f"the executorch suite runs on lanes {executorch_suite.lanes!r}, none of which is the "
        "nightly lane the GPU tier requests, so the pairing check runs nowhere"
    )


def test_the_range_install_runs_in_a_fresh_venv():
    # The end-user range install has to resolve the specifier from scratch. Run in the build
    # interpreter, which already holds the exact pin, and pip resolves nothing and proves nothing.
    # This locks the fresh-venv shape so it cannot silently regress to an in-place install.
    workflow = (REPO_ROOT / ".github/workflows/executorch-build-linux.yml").read_text(
        encoding="utf-8"
    )
    marker = "# pin-check: range-ok"
    assert marker in workflow, "the range-ok install marker is gone"
    after = workflow.split(marker, 1)[1].splitlines()
    # The install command is the first non-empty line under the marker.
    install = next(line for line in after if line.strip())
    assert (
        "range-check-venv" in install
    ), f"the range-ok install no longer runs in a fresh venv: {install.strip()!r}"


def test_the_pin_update_workflow_does_not_interpolate_untrusted_values_into_shell():
    # github.ref and inputs.track are attacker-influenceable text. Interpolated with ${{ }} into a
    # run: block they are shell source, so a crafted ref runs code in a job that holds a
    # write-scoped token. They must arrive through env and be read as "$REF" / "$TRACK" instead.
    import yaml

    path = REPO_ROOT / ".github/workflows/executorch-pin-update.yml"
    text = path.read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    untrusted = ("github.ref", "inputs.track", "steps.track.outputs.track")

    offenders = []
    for job in workflow.get("jobs", {}).values():
        for step in job.get("steps", []):
            run = step.get("run")
            if not run:
                continue
            for expr in untrusted:
                if "${{" in run and expr in run:
                    offenders.append((step.get("name", "?"), expr))
    assert not offenders, (
        "these steps interpolate an untrusted value into a run: script instead of reading it from "
        f"env: {offenders}"
    )

    # And the guard still reads the values, just safely: through env, as shell variables.
    assert "REF: ${{ github.ref }}" in text and "TRACK: ${{ inputs.track }}" in text, (
        "the trigger values are no longer passed through env, so the ref guard cannot read them "
        "safely"
    )
