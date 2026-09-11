"""Check exact build pins, compatible authoring ranges and installed-wheel provenance.

Discover requirements independently of the writer, and require known sites to remain present.
"""

import ast
import json
import os
import re
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO_ROOT = Path(__file__).resolve().parents[4]
VERSIONS = REPO_ROOT / "dev_dep_versions.yml"

# Discovery in mixed source/prose; packaging validates the matched requirement.
_CLAUSE = r"(?:===|==|>=|<=|~=|!=|<|>)\s*[0-9][0-9A-Za-z.!+*_-]*"
_MARKER_VALUE = r"""(?:[a-z_]+|"[^"\n]*"|'[^'\n]*')"""
_MARKER_ATOM = rf"(?:\([ \t]*)*{_MARKER_VALUE}[ \t]*(?:===|==|>=|<=|~=|!=|<|>|not[ \t]+in|in)[ \t]*{_MARKER_VALUE}(?:[ \t]*\))*"
REQUIREMENT = re.compile(
    r"(?<![0-9A-Za-z._-])executorch(?:\[[A-Za-z0-9_., -]+\])?\s*"
    rf"(?:@\s*[^\s\"'`]+|{_CLAUSE}(?:\s*,\s*{_CLAUSE})*)"
    rf"(?:\s*;\s*{_MARKER_ATOM}(?:\s+(?:and|or)\s+{_MARKER_ATOM})*)?",
    re.IGNORECASE,
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
    if canonicalize_name(parsed.name) != canonicalize_name(wanted.name):
        return f"which names {parsed.name}, not {wanted.name}"
    if parsed.url:
        return "direct URLs are not supported pin sites"
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

# Windows installs only the main wheel, without the Linux companion or authoring extra.
NO_NIGHTLY_MARKER = "pin-check: no-nightly"

# The files expected to pin ExecuTorch, mapped to how many sites each must carry, excluding
# dev_dep_versions.yml itself. A count per file rather than just the set of files, because a site
# that loses its version stops matching the search entirely rather than reporting a mismatch, and
# two of these files carry more than one site, so a set of paths let either quietly drop one. A
# minimum rather than an exact count, since the stacked runtime-wheel change removes one README
# site and an exact count could not hold on both branches.
_EXPECTED_REQUIREMENT_SITES = {
    ".github/workflows/build_linux.yml": 1,
    ".github/workflows/executorch-test-linux.yml": 2,
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

# Minimum source-commit sites, including the reference runner's shell default.
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
    result = subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0 or (
        arguments[0] == "grep" and result.returncode == 1
    ), result.stderr
    return result.stdout


def _tracked_files() -> list[str]:
    return [name for name in _git("ls-files", "-z").split("\0") if name]


def _is_source_test(name: str) -> bool:
    return (
        name.startswith("tests/")
        and Path(name).name.startswith("test_")
        and name.endswith(".py")
    )


def _versions() -> dict:
    values = yaml.safe_load(VERSIONS.read_text(encoding="utf-8"))
    assert isinstance(values, dict), "version source must be a YAML mapping"
    return values


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
    """Read the release line shared by final and nightly versions."""
    major, minor = version.split(".")[:2]
    return major, minor


def _expected(path: str, number: int, version: str) -> str:
    if not _wants_range(path, number):
        return f"executorch=={version}"

    major, minor = _release_line(version)
    return f"executorch>={version},<{major}.{int(minor) + 1}"


# Current TensorRT nightly channels. Retained cu126 artifacts do not mean new wheels publish there.
_PUBLISHED_NIGHTLY_CHANNELS = frozenset({"cu130", "cu132"})

# Tracked files with no suffix that still carry install commands. justfile writes the nightly
# ExecuTorch install for local builds, so the printed-install walk has to read it by name.
_EXTENSIONLESS_INSTALL_FILES = frozenset({"justfile"})

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
        # A URL scheme contains "//" and is not a comment. Splitting on it truncated any line
        # carrying an index URL, which hid a real pin from the search and reported the site as
        # missing rather than as wrong.
        for candidate in re.finditer(re.escape(marker), text):
            start = candidate.start()
            if marker == "//" and text[max(0, start - 1) : start] == ":":
                continue
            text = text[:start]
            break
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


def test_shared_workflows_export_the_row_cuda_channel() -> None:
    for filename, job_name in (
        ("build_linux.yml", "build"),
        ("linux-test.yml", "test"),
    ):
        workflow = yaml.safe_load(
            (REPO_ROOT / ".github/workflows" / filename).read_text()
        )
        job = workflow["jobs"][job_name]
        assert job["env"].get("CU_VERSION") == "${{ matrix.desired_cuda }}", filename
        setup = next(
            s
            for s in job["steps"]
            if s.get("uses", "").endswith("/setup-binary-builds")
        )
        assert setup["with"]["cuda-version"] == "${{ env.CU_VERSION }}", filename
        assert all("CU_VERSION" not in s.get("env", {}) for s in job["steps"]), filename

    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/executorch-test-linux.yml").read_text()
    )
    assert workflow["jobs"]["test"]["uses"] == "./.github/workflows/linux-test.yml"


@pytest.mark.parametrize(
    "caller,job",
    [
        ("_test-linux.yml", "build"),
        ("release-linux-x86_64.yml", "release-wheel-artifacts"),
        ("release-linux-aarch64.yml", "release-wheel-artifacts"),
    ],
)
def test_runtime_callers_use_the_shared_pinned_build(caller, job):
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows" / caller).read_text())
    build = workflow["jobs"][job]
    assert build["uses"] == "./.github/workflows/build_linux.yml"
    assert build["with"]["build-executorch-runtime"] in (
        True,
        "${{ !inputs.python-only && !inputs.use-rtx }}",
    )
    shared = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/build_linux.yml").read_text()
    )
    step = next(
        s
        for s in shared["jobs"]["build"]["steps"]
        if s.get("name") == "Build the ExecuTorch runtime wheel"
    )
    assert step["if"].startswith("${{ inputs.build-executorch-runtime")
    assert f'executorch=={_versions()["__executorch_version__"]}' in step["run"]
    assert "${CONDA_RUN} python -m pip install pyyaml" in step["run"]
    assert "nightly/${CU_VERSION}" in step["run"]
    assert "release-executorch-runtime-wheel-artifacts" not in workflow["jobs"]
    assert not (REPO_ROOT / ".github/workflows/executorch-build-linux.yml").exists()


def test_every_requirement_matches_the_pin() -> None:
    version = _versions()["__executorch_version__"]

    wrong = []
    found = 0
    seen: Counter[str] = Counter()
    for line in _git("grep", "-nIi", "executorch").splitlines():
        path, number, text = line.split(":", 2)
        if path == VERSIONS.name or _is_source_test(path) or path.startswith("docs/"):
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

    # The setup invocation must actually install the exact requirement.
    monkeypatch.syspath_prepend(str(REPO_ROOT / "tests"))
    from ci.runner import _executorch_requirement, _setup_commands

    argv = [arg for command, _cwd in _setup_commands("executorch") for arg in command]
    assert argv.count(_executorch_requirement()) == 1, (
        "the executorch setup step does not install the pinned ExecuTorch exactly once: "
        f"{argv}"
    )

    # Read metadata as source to avoid invoking the top-level build.
    setup_tree = ast.parse((REPO_ROOT / "setup.py").read_text(encoding="utf-8"))
    extras = next(
        node.value
        for node in ast.walk(setup_tree)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "EXTRAS_REQUIRE" for t in node.targets)
    )
    # Require the two documented extras, without constraining unrelated extras.
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

    # The doc build reads the pin through shell substitution, outside the literal scan.
    workflow = (REPO_ROOT / ".github/workflows/docgen.yml").read_text(encoding="utf-8")
    # A commented-out install is not an active pin site.
    embedded = re.search(
        r'^[ \t]*"executorch==\$\((python3 -c \'[^\']+\')\)"',
        workflow,
        re.MULTILINE,
    )
    assert embedded, (
        ".github/workflows/docgen.yml no longer pins ExecuTorch alongside the extra on a live "
        "line. Without the exact pin, the nightly index resolves "
        "through the range and takes whichever dev build is newest that day."
    )
    # Compare syntax trees without executing workflow-provided code.
    command = shlex.split(embedded.group(1))
    expected = 'import yaml; print(yaml.safe_load(open("dev_dep_versions.yml"))["__executorch_version__"])'
    assert command[:2] == ["python3", "-c"] and len(command) == 3
    assert ast.dump(ast.parse(command[2])) == ast.dump(ast.parse(expected)), command


_RUNTIME_SETUP_PY = "py/torch-tensorrt-executorch-runtime/setup.py"


def test_the_runtime_wheel_pins_executorch_to_the_public_pin(monkeypatch) -> None:
    """Evaluate the metadata without invoking a native build."""
    import importlib.metadata
    import runpy
    import types

    import setuptools

    pin = _versions()["__executorch_version__"]
    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(cuda="13.2")
    torch.__version__ = "2.15.0.dev20200103+cu132"
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv(
        "TORCH_TENSORRT_EXECUTORCH_RUNTIME_VERSION", "0.1.0.dev20200103+cu132"
    )
    installed = {
        "executorch": f"{pin}+cu132",
        "torch-tensorrt": torch.__version__,
        "tensorrt-cu13": "11.2.1",
        "nvidia-cuda-runtime": "13.2.0",
    }
    original_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: installed[name] if name in installed else original_version(name),
    )
    metadata = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: metadata.update(kwargs))
    runpy.run_path(str(REPO_ROOT / _RUNTIME_SETUP_PY))
    requirements = [
        r for r in metadata["install_requires"] if r.startswith("executorch")
    ]
    assert requirements == [f"executorch=={pin}"]


def _declared_cuda_versions(name: str) -> set[str]:
    """Read a CUDA version list the matrix filter declares, so a new row does not need editing here."""
    for node in ast.parse(
        (REPO_ROOT / ".github/scripts/filter-matrix.py").read_text(encoding="utf-8")
    ).body:
        target = getattr(node, "target", None) or next(
            iter(getattr(node, "targets", [])), None
        )
        if isinstance(target, ast.Name) and target.id == name:
            return {element.value for element in node.value.elts}
    raise AssertionError(f"filter-matrix.py declares no {name}")


@pytest.mark.unit
def test_matrix_keeps_every_cuda_13_row_the_pin_supports():
    """The Arm matrix must offer the same CUDA 13 rows as x86, since the pin supports both."""
    x86 = _declared_cuda_versions("x86_cuda_versions")
    arm = _declared_cuda_versions("arm_cuda_versions")
    assert arm == {cuda for cuda in x86 if not cuda.startswith("cu12")}


@pytest.mark.parametrize("channel", ["nightly", "test", "release", None])
@pytest.mark.parametrize("arch", ["cuda", "cuda-aarch64", "cuda-arm64"])
@pytest.mark.parametrize("use_rtx", ["true", "false"])
def test_matrix_keeps_cuda_12_only_for_release_channels(channel, arch, use_rtx):
    supported = _declared_cuda_versions(
        "x86_cuda_versions" if arch == "cuda" else "arm_cuda_versions"
    )
    rows = [
        {
            "python_version": "3.12",
            "desired_cuda": cuda,
            "gpu_arch_type": arch,
            **({"channel": channel} if channel is not None else {}),
        }
        for cuda in sorted(supported | {"cu126"})
    ]
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / ".github/scripts/filter-matrix.py"),
            "--matrix",
            json.dumps({"include": rows}),
            "--limit-pr-builds",
            "false",
            "--use-rtx",
            use_rtx,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    actual = {row["desired_cuda"] for row in json.loads(result.stdout)["include"]}
    expected = {cuda for cuda in supported if not cuda.startswith("cu12")}
    assert expected, "the matrix filter declares no CUDA 13 rows to keep"
    if channel in {"test", "release"} and arch == "cuda":
        expected.add("cu126")
    assert actual == expected


def test_jetpack_matrix_keeps_its_separate_cuda_contract():
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / ".github/scripts/filter-matrix.py"),
            "--matrix",
            json.dumps(
                {
                    "include": [
                        {
                            "python_version": "3.10",
                            "desired_cuda": "cu126",
                            "gpu_arch_type": "cuda-aarch64",
                            "channel": "nightly",
                        }
                    ]
                }
            ),
            "--jetpack",
            "true",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = json.loads(result.stdout)["include"]
    assert len(rows) == 1
    assert rows[0]["desired_cuda"] == "cu126"
    assert rows[0]["container_image"] == "nvcr.io/nvidia/l4t-jetpack:r36.4.0"


@pytest.mark.parametrize("channel", ["cu126", "cu130", "cu132", "cu134"])
def test_install_channel_guard_rejects_unsupported_nightly_recipes(
    tmp_path, monkeypatch, channel
):
    recipe = tmp_path / "README.md"
    recipe.write_text(
        '```bash\npython -m pip install --pre "torch-tensorrt[executorch]" '
        f"--extra-index-url https://download.pytorch.org/whl/nightly/{channel}\n```\n"
    )
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "README.md"], check=True)
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    if channel in {"cu130", "cu132"}:
        test_every_printed_install_instruction_names_the_nightly_channel()
    else:
        with pytest.raises(AssertionError, match=f"nightly/{channel}"):
            test_every_printed_install_instruction_names_the_nightly_channel()


@pytest.mark.unit
def test_the_runner_follows_the_row_s_cuda_version(monkeypatch) -> None:
    """The runner follows each row; cu130 is only the local default, not the cu132 PR row."""
    monkeypatch.syspath_prepend(str(REPO_ROOT / "tests"))
    from ci import runner

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
    assert channel_for("").endswith("/nightly/cu130")


@pytest.mark.unit
@pytest.mark.parametrize(
    "channel", ["cpu", "13.2", "cu126", "cu128", "cu134", "CU132", " cu132"]
)
def test_runner_rejects_unsupported_cuda_channels(monkeypatch, channel):
    from tests.ci import runner

    monkeypatch.setenv("CU_VERSION", channel)
    with pytest.raises(ValueError, match="CU_VERSION") as error:
        runner._setup_commands("executorch")
    assert repr(channel) in str(error.value)
    assert "cu130" in str(error.value) and "cu132" in str(error.value)


@pytest.mark.unit
def test_runner_channel_check_detects_removed_validator(monkeypatch):
    import inspect
    from tests.ci import runner

    source = inspect.getsource(runner._setup_commands)
    changed = source.replace('if cuda not in {"cu130", "cu132"}:', "if False:")
    assert changed != source
    namespace = vars(runner).copy()
    exec(compile(changed, runner.__file__, "exec"), namespace)
    monkeypatch.setattr(runner, "_setup_commands", namespace["_setup_commands"])
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        test_runner_rejects_unsupported_cuda_channels(monkeypatch, "cpu")


def _load_utils_channel_helpers(fake_cuda: str | None):
    """Execute the real helpers without importing their torch and TensorRT dependencies."""
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
    exec(compile(module, "<utils-extract>", "exec"), namespace)
    return (
        namespace["executorch_install_channel"],
        namespace["executorch_install_command"],
    )


def test_the_executorch_install_message_names_the_torch_channel() -> None:
    """Install advice follows the supported CUDA channel and does not request an upgrade."""
    channel, command = _load_utils_channel_helpers("13.2")
    assert channel() == "cu132"
    message = command()
    assert "download.pytorch.org/whl/nightly/cu132" in message, message
    # Permit prereleases without requesting replacement of an already suitable installation.
    assert "--pre" in message, message
    assert "--upgrade" not in message, message

    channel_130, command_130 = _load_utils_channel_helpers("13.0")
    assert channel_130() == "cu130"
    assert "nightly/cu130" in command_130()

    # Check each installation error independently, not just one helper call per file.
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


@pytest.mark.parametrize("cuda", ["13.0", "13.2", None])
@pytest.mark.parametrize("entrypoint", ["lazy", "save", "_save_as_executorch"])
def test_import_errors_preserve_context_and_install_guidance(
    monkeypatch, cuda, entrypoint
):
    import __future__
    import importlib.util
    import types

    _, command = _load_utils_channel_helpers(cuda)
    utils = types.ModuleType("torch_tensorrt._utils")
    utils.executorch_install_command = command
    package = types.ModuleType("torch_tensorrt")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a: (
            None if name == "executorch.exir" else original_find_spec(name, *a)
        ),
    )
    path = REPO_ROOT / "py/torch_tensorrt/executorch/__init__.py"
    spec = importlib.util.spec_from_file_location("torch_tensorrt.executorch", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    source = REPO_ROOT / "py/torch_tensorrt/_compile.py"
    functions = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"save", "_save_as_executorch", "_has_executorch_exir"}
    ]
    namespace = {
        "importlib": importlib,
        "executorch_install_command": command,
        "CudaGraphsTorchTensorRTModule": type("CudaGraphsStub", (), {}),
        "_parse_module_type": lambda value: None,
        "ENABLED_FEATURES": types.SimpleNamespace(torch_tensorrt_runtime=True),
    }
    exec(
        compile(
            ast.Module(functions, []),
            str(source),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    with pytest.raises(ImportError) as error:
        if entrypoint == "lazy":
            module.TensorRTBackend
        elif entrypoint == "save":
            namespace[entrypoint](object(), "unused.pte", output_format="executorch")
        else:
            namespace[entrypoint](object(), "unused.pte")
    context = {
        "lazy": "Cannot access torch_tensorrt.executorch.TensorRTBackend",
        "save": "Saving with output_format='executorch' requires executorch.exir",
        "_save_as_executorch": "Could not import the ExecuTorch export integration",
    }
    message = str(error.value)
    assert context[entrypoint] in message
    assert "This CUDA integration supports Linux" in message
    assert message.endswith("Setup: " + command())


def test_derived_requirements_roll_the_minor_over(tmp_path: Path) -> None:
    # The upper bound is a version, not a decimal: 1.9 has to become 1.10, not 1.1.
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
        # end-user install rehearsal in executorch-test-linux.yml -- so arriving here usually
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
@pytest.mark.parametrize("setup_rc", [0, 7])
def test_a_failed_setup_step_stops_the_suite(monkeypatch, tmp_path, setup_rc):
    """A failed setup must stop before import-skipped tests can report a false pass."""
    monkeypatch.syspath_prepend(str(REPO_ROOT / "tests"))
    from ci import runner

    monkeypatch.setenv("RUNNER_TEST_RESULTS_DIR", str(tmp_path))
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

    assert rc == setup_rc, f"run_suite returned {rc}, expected {setup_rc}"
    ran_pytest = any("pytest" in " ".join(argv) for argv in calls)
    assert ran_pytest is (setup_rc == 0), (
        "pytest ran even though a setup step failed"
        if ran_pytest
        else "pytest never ran even though every setup step succeeded"
    )


@pytest.mark.unit
def test_the_lockfile_records_the_same_executorch_range_as_setup_py():
    """Allow the separately refreshed development lock to lag, but never lead the pin."""
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
    """Find the shell command owning a match, respecting quotes and continuations."""
    scan = _blank_comments_preserving_length(block)
    starts = [
        m.start()
        for m in _INSTALL_INVOCATION.finditer(scan)
        if m.start() <= match_offset
    ]
    if not starts:
        return ""
    begin = starts[-1]
    quote = None
    escaped = False
    finish = len(scan)
    for index in range(begin, len(scan)):
        char = scan[index]
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
        elif quote:
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in "\n;|&":
            finish = index
            break
    return scan[begin:finish] if begin <= match_offset < finish else ""


@pytest.mark.unit
def test_every_printed_install_instruction_names_the_nightly_channel():
    """Supported shell install examples must name the pinned runtime's nightly channel.

    Inspect each command's arguments, including continued lines, in tracked source files.
    This is not a general shell or embedded-language parser.
    """
    tracked = _tracked_files()

    # Named/local extras, built wheels and direct ExecuTorch installs resolve the same pin.
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
        if _is_source_test(name) or name == "py/torch_tensorrt/_utils.py":
            # Generated advice is exercised by the install-helper tests.
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
            if not block:
                continue
            try:
                argv = shlex.split(block.replace("\\\n", ""), comments=True)
            except ValueError as error:
                missing.append(f"{name}:{line} cannot parse install command: {error}")
                continue
            # The explicit exemption is valid only for the Windows main-wheel install.
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
                if "executorch" in argv:
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
            indexes = [
                argv[i + 1]
                for i, arg in enumerate(argv[:-1])
                if arg in {"--extra-index-url", "--index-url"}
            ]
            index_text = " ".join(indexes)
            channel = re.search(
                r"download\.pytorch\.org/whl/nightly(?:/(cu\d+))?", index_text
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
                # An index variable must resolve to a preceding literal assignment.
                assignment = _resolve_shell_assignment(
                    text, variable_index.group(1), match.start()
                )
                if assignment is None:
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
            # Concrete CUDA channels must be supported; variable row exports are checked separately.
            suffix = channel.group(1)
            if suffix and suffix not in _PUBLISHED_NIGHTLY_CHANNELS:
                missing.append(
                    f"{name}:{line} installs from nightly/{suffix}, which is not a supported "
                    f"TensorRT nightly channel; expected one of {sorted(_PUBLISHED_NIGHTLY_CHANNELS)}"
                )
            # An unversioned named distribution needs permission to select a nightly.
            # Older released extras can declare a different ExecuTorch requirement.
            named_distribution = re.fullmatch(
                r"torch[-_]tensorrt\[[^]]*executorch[^]]*\]", match.group(0)
            )
            if named_distribution and not re.search(r"(?:^|\s)--pre(?:\s|$)", block):
                missing.append(
                    f"{name}:{line} installs {match.group(0)} without --pre, so pip resolves the "
                    "stable release rather than allowing the intended nightly prerelease"
                )

    assert (
        not missing
    ), f"these ExecuTorch install instructions do not satisfy the pinned nightly contract: {missing}"


@pytest.mark.unit
def test_the_no_nightly_marker_only_exempts_a_win32_install():
    """Only the Windows main-wheel install may omit this Linux integration's nightly index."""
    tracked = _tracked_files()

    misplaced = []
    for name in tracked:
        if not name or _is_source_test(name):
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
    # A sibling job's dependency installation cannot prepare this job.
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/linter.yml").read_text(encoding="utf-8")
    )
    # PyYAML's YAML 1.1 loader reads an unquoted "on" key as True.
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
    import tomllib

    groups = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())[
        "dependency-groups"
    ]
    installed = {canonicalize_name(Requirement(dep).name) for dep in groups["lint"]}
    assert {"pytest", "pyyaml", "packaging", "setuptools", "wheel"} <= installed
    install_code = (
        "import tomllib, subprocess; "
        "deps = tomllib.load(open('pyproject.toml', 'rb'))['dependency-groups']['lint']; "
        "subprocess.run(['uv', 'pip', 'install', '--system'] + deps, check=True)"
    )
    live_conditions = {"always()", "success()", "success() || failure()"}
    for module in ("test_executorch_pin.py", "test_update_executorch_pin.py"):
        owning = [
            (name, job, step)
            for name, job in workflow["jobs"].items()
            for step in job.get("steps", [])
            if module in step.get("run", "")
        ]
        assert (
            len(owning) == 1
        ), f"expected one CI invocation of {module}, got {len(owning)}"
        name, job, step = owning[0]
        for owner in (job, step):
            assert str(owner.get("if", "success()")) in live_conditions, (
                name,
                owner.get("if"),
            )
            assert not owner.get("continue-on-error"), name
        assert step.get("shell", "bash") in {"bash", "sh"}
        commands = [
            shlex.split(line, comments=True)
            for line in step["run"].replace("\\\n", "").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        assert len(commands) == 2 and commands[0] == [
            "cd",
            "$GITHUB_WORKSPACE",
        ], commands
        argv = commands[1]
        assert argv[:4] == [
            "python3",
            "-m",
            "pytest",
            f"tests/py/dynamo/executorch/{module}",
        ], argv
        options = argv[4:]
        assert "--noconftest" in options and "addopts=" in options, argv
        while options:
            option, *options = options
            if option in {"-q", "-v", "--no-header", "--noconftest"}:
                continue
            assert option in {"-p", "-o"} and options, argv
            value, *options = options
            assert value == {"-p": "no:cacheprovider", "-o": "addopts="}[option], argv

        earlier = job["steps"][: job["steps"].index(step)]
        dependencies_ready = False
        for previous in earlier:
            if str(previous.get("if", "success()")) not in live_conditions:
                continue
            for line in previous.get("run", "").splitlines():
                if not re.match(r"\s*python3\s+-c\s", line):
                    continue
                args = shlex.split(line, comments=True)
                if args[:2] == ["python3", "-c"] and len(args) == 3:
                    dependencies_ready |= ast.dump(ast.parse(args[2])) == ast.dump(
                        ast.parse(install_code)
                    )
        assert (
            dependencies_ready
        ), f"{name} does not install the lint group before {module}"


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
    from tests.ci.suites import by_name

    suite = by_name("executorch")
    captured = (
        subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; _trt_py() { printf "%s\\0" "$@"; }; _trt_xml() { echo ignored.xml; }; trt_tier_executorch',
                "capture",
                str(REPO_ROOT / "tests/py/utils/ci_helpers.sh"),
            ],
            cwd=REPO_ROOT,
            env={**os.environ, "TRT_REPO_ROOT": str(REPO_ROOT)},
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.rstrip("\0")
        .split("\0")
    )
    assert captured[:2] == ["-m", "pytest"] and "executorch/" in captured, captured
    assert captured.count("-k") == 1, captured
    keywords = [suite.keyword, captured[captured.index("-k") + 1]]
    from _pytest.mark.expression import Expression

    nodes = {
        f"test_executorch_pin.py::{PAIRING_TEST}": True,
        "test_executorch_pin.py::test_every_requirement_matches_the_pin": False,
        "test_update_executorch_pin.py::test_write_pins_updates_real_sites_and_is_idempotent": False,
        "test_api.py::test_export": True,
    }
    for keyword in keywords:
        expression = Expression.compile(keyword)
        actual = {
            node: expression.evaluate(
                lambda token, node=node: token.lower() in node.lower()
            )
            for node in nodes
        }
        assert actual == nodes, (keyword, actual)

    # Both invocation routes must reach the suite, not merely define the right filter.
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
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/executorch-test-linux.yml").read_text()
    )
    script = workflow["jobs"]["test"]["with"]["script"]
    before, after = script.split("# pin-check: range-ok", 1)
    creation = [
        shlex.split(line, comments=True)
        for line in before.splitlines()
        if re.match(r"\s*python\s+-m\s+venv\b", line)
    ]
    venv = "${RUNNER_TEMP}/range-check-venv"
    assert ["python", "-m", "venv", venv] in creation
    install = next(line for line in after.splitlines() if line.strip())
    argv = shlex.split(install, comments=True)
    assert argv[:5] == [venv + "/bin/python", "-m", "pip", "install", "--no-deps"], argv


def test_the_pin_update_workflow_does_not_interpolate_untrusted_values_into_shell():
    # github.ref and inputs.track are attacker-influenceable text. Interpolated with ${{ }} into a
    # run: block they are shell source, so a crafted ref runs code in a job that holds a
    # write-scoped token. They must arrive through env and be read as "$REF" / "$TRACK" instead.
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


@pytest.mark.parametrize(
    "module", ["test_executorch_pin.py", "test_update_executorch_pin.py"]
)
@pytest.mark.parametrize(
    "mutation",
    [
        "--help",
        "|| :",
        "| cat",
        "--collect-only",
        "-k never",
        "; true",
        "> /dev/null",
        "delete",
    ],
)
def test_review_ci_guard_rejects_disabled_checks(monkeypatch, mutation, module):
    path = REPO_ROOT / ".github/workflows/linter.yml"
    workflow = yaml.safe_load(path.read_text())
    steps = workflow["jobs"]["py-linting"]["steps"]
    step = next(s for s in steps if module in s.get("run", ""))
    if mutation == "delete":
        steps.remove(step)
    else:
        step["run"] = step["run"].rstrip() + " " + mutation + "\n"
    original = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda p, *a, **kw: (
            yaml.safe_dump(workflow) if p == path else original(p, *a, **kw)
        ),
    )
    with pytest.raises(AssertionError):
        test_the_pin_check_runs_in_ci()


@pytest.mark.parametrize(
    "body",
    [
        "python -m pip install --pre \\\n  executorch \\\n  --extra-index-url https://download.pytorch.org/whl/nightly/cu130",
        'python -m pip install --pre "torch-tensorrt[executorch]"\necho https://download.pytorch.org/whl/nightly/cu130',
        'python -m pip install --pre "torch-tensorrt[executorch]" # https://download.pytorch.org/whl/nightly/cu130',
        'python -m pip install --pre "torch-tensorrt[executorch]"; echo https://download.pytorch.org/whl/nightly/cu130',
    ],
)
def test_review_install_guard_rejects_unrelated_tokens(tmp_path, monkeypatch, body):
    (tmp_path / "README.md").write_text("```bash\n" + body + "\n```\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    with pytest.raises(AssertionError):
        test_every_printed_install_instruction_names_the_nightly_channel()


@pytest.mark.parametrize("workflow_name", ["build_linux.yml", "linux-test.yml"])
def test_review_cuda_export_is_required(monkeypatch, workflow_name):
    path = REPO_ROOT / ".github/workflows" / workflow_name
    original = Path.read_text
    text = path.read_text().replace(
        "CU_VERSION: ${{ matrix.desired_cuda }}", "REMOVED_CUDA: unused"
    )
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda p, *a, **kw: text if p == path else original(p, *a, **kw),
    )
    with pytest.raises(AssertionError):
        test_shared_workflows_export_the_row_cuda_channel()


def test_review_venv_name_in_comment_does_not_count(monkeypatch):
    path = REPO_ROOT / ".github/workflows/executorch-test-linux.yml"
    text = path.read_text().replace(
        '"${RUNNER_TEMP}/range-check-venv/bin/python" -m pip install --no-deps',
        "python -m pip install --no-deps # range-check-venv",
    )
    original = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda p, *a, **kw: text if p == path else original(p, *a, **kw),
    )
    with pytest.raises(AssertionError):
        test_the_range_install_runs_in_a_fresh_venv()


@pytest.mark.parametrize(
    "actual",
    [
        "ExecuTorch==1.5.0.dev1",
        "executorch[coreml] == 1.5.0.dev1",
        'executorch==1.5.0.dev1; python_version >= "3.10"',
        'executorch[coreml]==1.5.0.dev1; (sys_platform == "linux" or python_version >= "3.10")',
    ],
)
def test_review_requirement_scanner_preserves_valid_shapes(actual):
    matches = REQUIREMENT.findall(actual)
    assert matches == [actual]
    assert not _requirement_disagrees(
        matches[0], "executorch==1.5.0.dev1", "1.5.0.dev1"
    )


@pytest.mark.parametrize(
    "actual",
    [
        "my-executorch==1.5.0.dev1",
        "not_executorch==1.5.0.dev1",
        "not.executorch==1.5.0.dev1",
    ],
)
def test_review_requirement_scanner_ignores_other_distributions(actual):
    assert not REQUIREMENT.findall(actual)


@pytest.mark.parametrize(
    "line",
    [
        "__executorch_version__: '1.5.0.dev1'",
        "__executorch_version__: 1.5.0.dev1",
        '__executorch_version__ : "1.5.0.dev1" # pin',
    ],
)
def test_review_yaml_readers_agree(tmp_path, monkeypatch, line):
    source = tmp_path / "dev_dep_versions.yml"
    source.write_text(line + "\n")
    monkeypatch.setattr(sys.modules[__name__], "VERSIONS", source)
    assert _versions()["__executorch_version__"] == "1.5.0.dev1"
    assert _runner_requirement(tmp_path) == "executorch==1.5.0.dev1"


@pytest.mark.parametrize(
    "field,value",
    [
        ("tier", "l9"),
        ("lanes", ("nightl",)),
        ("variants", ("bad",)),
        ("platforms", ("darwin",)),
    ],
)
def test_review_suite_rejects_unknown_values(field, value):
    from tests.ci.suites import Suite

    args = dict(name="probe", tier="l2", lanes=("nightly",))
    args[field] = value
    with pytest.raises(ValueError, match=field):
        Suite(**args)


def test_review_runner_empty_channel_uses_local_default(monkeypatch):
    from tests.ci.runner import _setup_commands

    monkeypatch.setenv("CU_VERSION", "")
    argv = _setup_commands("executorch")[0][0]
    assert argv[argv.index("--extra-index-url") + 1].endswith("/cu130")


@pytest.mark.parametrize("cuda", [None, "", "12.6", "12.8", "13.4", "14.0"])
def test_unsupported_install_channels_give_guidance(cuda):
    channel, command = _load_utils_channel_helpers(cuda)
    assert channel() is None
    message = command()
    assert "Linux" in message and "13.0" in message and "13.2" in message
    assert "pip install" not in message and "https://" not in message


@pytest.mark.parametrize("platform", ["linux", "win32"])
@pytest.mark.parametrize("install_rc", [0, 7])
def test_final_wheel_install_controls_the_appended_script(platform, install_rc):
    script = (REPO_ROOT / ".github/scripts/install-torch-tensorrt.sh").read_text()
    stubs = r"""
uname() { echo x86_64; }
python() {
    if [[ "$*" == *"sys.platform"* ]]; then
        echo "$TEST_PLATFORM"
    elif [[ "$*" == *"sysconfig"* ]]; then
        echo /fake/site-packages
    elif [[ "$*" == *"torch_tensorrt"*".whl"* ]]; then
        echo final-wheel-install
        return "$TEST_INSTALL_RC"
    fi
}
"""
    result = subprocess.run(
        ["bash"],
        input=stubs + script + "\necho appended-caller-ran\n",
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "TEST_PLATFORM": platform,
            "TEST_INSTALL_RC": str(install_rc),
            "CHANNEL": "nightly",
            "CU_VERSION": "cu130",
            "RUNNER_ARTIFACT_DIR": "/fake/artifacts",
        },
    )
    assert "final-wheel-install" in result.stdout, result.stderr
    assert result.returncode == (1 if install_rc else 0), result.stderr
    assert ("appended-caller-ran" in result.stdout) is (install_rc == 0)


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_installer_check_detects_lost_exit_guard(monkeypatch, platform):
    path = REPO_ROOT / ".github/scripts/install-torch-tensorrt.sh"
    original = Path.read_text
    text = path.read_text().replace(" || exit 1", "")
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda p, *a, **kw: text if p == path else original(p, *a, **kw),
    )
    with pytest.raises(AssertionError):
        test_final_wheel_install_controls_the_appended_script(platform, 7)


@pytest.mark.parametrize("side_effect", [False, True])
def test_docgen_pin_reader_accepts_only_the_allowed_ast(monkeypatch, side_effect):
    path = REPO_ROOT / ".github/workflows/docgen.yml"
    original = Path.read_text
    text = path.read_text()
    embedded = re.search(r"python3 -c '([^']+)'", text)
    assert embedded
    old = embedded.group(1)
    code = (
        old.replace(";", ";  ").replace("print(", "print( ")
        if not side_effect
        else old + '; print("unexpected")'
    )
    text = text.replace(old, code)
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda p, *a, **kw: text if p == path else original(p, *a, **kw),
    )
    if side_effect:
        with pytest.raises(AssertionError):
            test_derived_requirements_match_the_pin(monkeypatch)
    else:
        test_derived_requirements_match_the_pin(monkeypatch)


@pytest.mark.parametrize("route", ["manifest", "shell"])
@pytest.mark.parametrize("removed", ["updater", "pairing"])
def test_gpu_filter_checks_detect_lost_selection(monkeypatch, tmp_path, route, removed):
    from dataclasses import replace
    from tests.ci import suites

    def mutate(keyword):
        if removed == "updater":
            return keyword.replace(" and not test_update_executorch_pin", "")
        return keyword.replace(f" or {PAIRING_TEST}", "")

    if route == "manifest":
        suite = suites.by_name("executorch")
        original_by_name = suites.by_name
        monkeypatch.setattr(
            suites,
            "by_name",
            lambda name: (
                replace(suite, keyword=mutate(suite.keyword))
                if name == "executorch"
                else original_by_name(name)
            ),
        )
    else:
        helper = REPO_ROOT / "tests/py/utils/ci_helpers.sh"
        changed = tmp_path / "ci_helpers.sh"
        changed.write_text(mutate(helper.read_text()))
        original_run = subprocess.run

        def run(argv, **kwargs):
            return original_run(
                [str(changed) if arg == str(helper) else arg for arg in argv], **kwargs
            )

        monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(AssertionError):
        test_the_pairing_check_survives_the_gpu_lane_deselection()


@pytest.mark.parametrize(
    "mutation", ["remove-install", "disable-install", "remove-pyyaml"]
)
def test_ci_guard_requires_dependencies_in_the_owning_job(monkeypatch, mutation):
    workflow_path = REPO_ROOT / ".github/workflows/linter.yml"
    metadata_path = REPO_ROOT / "pyproject.toml"
    workflow = yaml.safe_load(workflow_path.read_text())
    metadata = metadata_path.read_text()
    steps = workflow["jobs"]["py-linting"]["steps"]
    step = next(
        step for step in steps if "['dependency-groups']['lint']" in step.get("run", "")
    )
    if mutation == "remove-install":
        steps.remove(step)
    elif mutation == "disable-install":
        step["if"] = "false"
    else:
        metadata = metadata.replace('    "pyyaml>=6.0",', "")
    original = Path.read_text
    contents = {workflow_path: yaml.safe_dump(workflow), metadata_path: metadata}
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda p, *a, **kw: contents[p] if p in contents else original(p, *a, **kw),
    )
    with pytest.raises(AssertionError):
        test_the_pin_check_runs_in_ci()


@pytest.mark.parametrize("remove_pin", [False, True])
def test_requirement_discovery_ignores_fixtures_but_not_missing_sites(
    tmp_path, monkeypatch, remove_pin
):
    pin = "1.5.0.dev1"
    (tmp_path / "justfile").write_text(
        "pip install executorch" + ("" if remove_pin else "==" + pin) + "\n"
    )
    fixture = tmp_path / "tests/test_fixture.py"
    fixture.parent.mkdir()
    fixture.write_text('requirement = "executorch==0.0.1"\n')
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        sys.modules[__name__], "_versions", lambda: {"__executorch_version__": pin}
    )
    monkeypatch.setattr(
        sys.modules[__name__], "_EXPECTED_REQUIREMENT_SITES", {"justfile": 1}
    )
    if remove_pin:
        with pytest.raises(AssertionError):
            test_every_requirement_matches_the_pin()
    else:
        test_every_requirement_matches_the_pin()


@pytest.mark.parametrize("allow", [False, True])
def test_update_workflow_requires_manual_downgrade_authority(tmp_path, allow):
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/executorch-pin-update.yml").read_text()
    )
    triggers = workflow.get("on", workflow.get(True))
    declaration = triggers["workflow_dispatch"]["inputs"]["allow_downgrade"]
    assert declaration["type"] == "boolean" and declaration["default"] is False
    steps = workflow["jobs"]["update-pin"]["steps"]
    step = next(step for step in steps if step.get("id") == "update")
    assert (
        step["env"]["ALLOW_DOWNGRADE"]
        == "${{ github.event_name == 'workflow_dispatch' && inputs.allow_downgrade }}"
    )
    stubs = 'python() { printf "%s\\n" "$@"; }; git() { return 0; };\n'
    result = subprocess.run(
        ["bash"],
        input=stubs + step["run"],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "TRACK": "stable",
            "ALLOW_DOWNGRADE": str(allow).lower(),
            "GITHUB_OUTPUT": str(tmp_path / "output"),
        },
        check=True,
    )
    assert ("--allow-downgrade" in result.stdout.splitlines()) is allow
    assert result.stdout.splitlines()[1:3] == ["--track", "stable"]
    branch = next(
        step for step in steps if "create-pull-request@" in step.get("uses", "")
    )["with"]["branch"]
    assert (
        branch
        == "executorch-pin-update/${{ github.ref_name }}/${{ steps.track.outputs.track }}"
    )
    branches = {
        branch.replace("${{ github.ref_name }}", base).replace(
            "${{ steps.track.outputs.track }}", "stable"
        )
        for base in ("release/2.14", "release-2.14")
    }
    assert len(branches) == 2


def test_uv_cache_tracks_pin_metadata():
    import tomllib

    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    keys = config["tool"]["uv"].get("cache-keys", [])
    files = {key["file"] for key in keys if "file" in key}
    assert {"pyproject.toml", "setup.py", "setup.cfg", "dev_dep_versions.yml"} <= files


def test_uv_cache_guard_detects_missing_pin_key(monkeypatch):
    path = REPO_ROOT / "pyproject.toml"
    text = path.read_text().replace('{ file = "dev_dep_versions.yml" },', "")
    original = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda p, *a, **kw: text if p == path else original(p, *a, **kw),
    )
    with pytest.raises(AssertionError):
        test_uv_cache_tracks_pin_metadata()


@pytest.mark.parametrize(
    "field,value",
    [
        ("tier", "l9"),
        ("lanes", ("nightl",)),
        ("variants", ("bad",)),
        ("platforms", ("darwin",)),
    ],
)
def test_suite_validation_check_detects_removed_validator(monkeypatch, field, value):
    from tests.ci.suites import Suite

    monkeypatch.setattr(Suite, "__post_init__", lambda self: None)
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        test_review_suite_rejects_unknown_values(field, value)


@pytest.mark.parametrize(
    "event,ref,track,expected",
    [
        ("schedule", "refs/heads/main", "", "nightly"),
        ("schedule", "refs/heads/release/2.14", "", ""),
        ("workflow_dispatch", "refs/heads/main", "nightly", "nightly"),
        ("workflow_dispatch", "refs/heads/release/2.14", "stable", "stable"),
        ("workflow_dispatch", "refs/heads/release/2.14", "nightly", None),
    ],
)
def test_update_workflow_keeps_nightly_updates_on_main(
    tmp_path, event, ref, track, expected
):
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/executorch-pin-update.yml").read_text()
    )
    step = next(
        step
        for step in workflow["jobs"]["update-pin"]["steps"]
        if step.get("id") == "track"
    )
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash"],
        input=step["run"],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "EVENT": event,
            "REF": ref,
            "TRACK": track,
            "GITHUB_OUTPUT": str(output),
        },
    )
    if expected is None:
        assert result.returncode != 0 and not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert output.read_text() == f"track={expected}\n"
