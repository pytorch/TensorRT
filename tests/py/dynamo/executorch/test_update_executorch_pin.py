"""Exercise pin selection, wheel provenance and scoped rewrites without network access."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml
from packaging.version import Version

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPT = _REPO_ROOT / ".github/scripts/update_executorch_pin.py"
_SELF = "tests/py/dynamo/executorch/test_update_executorch_pin.py"
_spec = importlib.util.spec_from_file_location("update_executorch_pin", _SCRIPT)
updater = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(updater)
_COMMIT = "deadbeef" * 5


def _stage(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)


@pytest.fixture
def pin_repo(tmp_path, monkeypatch):
    """Only the real pin sites and bystanders, without repository history."""
    for name in (
        *updater._PIN_SITES,
        "dev_dep_versions.yml",
        str(_SCRIPT.relative_to(_REPO_ROOT)),
        _SELF,
    ):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_REPO_ROOT / name, target)
    _stage(tmp_path)
    monkeypatch.setattr(updater, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(updater, "_VERSIONS_FILE", tmp_path / "dev_dep_versions.yml")
    return tmp_path


@pytest.fixture
def wheel_download(monkeypatch):
    calls = []
    members = {"executorch/version.py": f'git_version: str = "{_COMMIT}"\n'}

    def download(cmd):
        calls.append(cmd)
        dest = Path(cmd[cmd.index("--dest") + 1])
        with zipfile.ZipFile(
            dest / "executorch-0.9.0-py3-none-any.whl", "w"
        ) as archive:
            for name, body in members.items():
                archive.writestr(name, body)
        return ""

    monkeypatch.setattr(updater, "_run", download)
    return members, calls


@pytest.mark.parametrize(
    "versions,track,expected",
    [
        (
            ["1.9.0.dev20200103+cu130", "1.10.0.dev20200101+cu132", "invalid"],
            "nightly",
            "1.10.0.dev20200101",
        ),
        (["0.9.9", "0.10.0", "0.11.0rc1", "0.11.0.dev1"], "stable", "0.10.0"),
        (
            ["1.5.0", "1.5.0rc1", "1.5.0.dev2+cu132", "1.5.0.dev1+cu130"],
            "nightly",
            "1.5.0.dev2",
        ),
    ],
)
@pytest.mark.unit
def test_pick_target_uses_version_order_and_public_versions(versions, track, expected):
    assert updater.pick_target(versions, track) == expected


@pytest.mark.parametrize(
    "versions,track", [(["1.0"], "nightly"), (["1.0.dev1", "bad"], "stable")]
)
@pytest.mark.unit
def test_pick_target_requires_a_matching_version(versions, track):
    with pytest.raises(SystemExit):
        updater.pick_target(versions, track)


@pytest.mark.unit
def test_available_versions_parses_pip_output(monkeypatch):
    monkeypatch.setattr(
        updater,
        "_run",
        lambda cmd: "executorch (1.0.dev2)\nAvailable versions: 1.0.dev2+cu130, 1.0.dev1+cu130\n",
    )
    assert updater.available_versions([]) == ["1.0.dev2+cu130", "1.0.dev1+cu130"]


@pytest.mark.unit
def test_run_surfaces_subprocess_diagnostics(capsys):
    with pytest.raises(subprocess.CalledProcessError):
        updater._run(
            [
                sys.executable,
                "-c",
                "import sys; print('pin-probe-error', file=sys.stderr); sys.exit(7)",
            ]
        )
    assert "pin-probe-error" in capsys.readouterr().err


@pytest.mark.parametrize(
    "line",
    [
        '__executorch_version__: "1.5.0.dev1"',
        "__executorch_version__: '1.5.0.dev1'",
        "__executorch_version__: 1.5.0.dev1",
        '__executorch_version__: "1.5.0.dev1" # keep this comment',
        '__executorch_version__ : "1.5.0.dev1"',
    ],
)
@pytest.mark.unit
def test_read_pin_accepts_yaml_scalar_formatting(tmp_path, monkeypatch, line):
    path = tmp_path / "versions.yml"
    path.write_text(line + "\n")
    monkeypatch.setattr(updater, "_VERSIONS_FILE", path)
    assert updater.read_pin("__executorch_version__") == "1.5.0.dev1"


@pytest.mark.unit
def test_wheel_provenance_is_read_without_executing_python(wheel_download):
    members, calls = wheel_download
    members["executorch/version.py"] += "raise RuntimeError('must not execute')\n"
    assert updater.wheel_git_version("0.9.0", []) == _COMMIT
    assert "--only-binary=:all:" in calls[0]
    assert "--no-deps" in calls[0]


@pytest.mark.parametrize(
    "body",
    [
        None,
        b"\xff",
        "git_version = None\n",
        f'git_version = "{_COMMIT}"\n' + "#" * 65537,
    ],
)
@pytest.mark.unit
def test_wheel_provenance_errors_are_contextual(wheel_download, body):
    members, _ = wheel_download
    if body is None:
        members.clear()
    else:
        members["executorch/version.py"] = body
    with pytest.raises(SystemExit, match="provenance"):
        updater.wheel_git_version("0.9.0", [])


@pytest.mark.parametrize("version,expected", [("1.9.0", "1.10"), ("1.5.0.dev1", "1.6")])
@pytest.mark.unit
def test_upper_bound(version, expected):
    assert updater._upper_bound(version) == expected


@pytest.mark.parametrize("channel", ["cu126", "cu128", "cu14", "cpu", "cu13"])
@pytest.mark.unit
def test_main_rejects_unsupported_channels(monkeypatch, channel):
    monkeypatch.setattr(
        updater, "available_versions", lambda args: pytest.fail("unexpected query")
    )
    with pytest.raises(SystemExit) as error:
        updater.main(["--channel", channel])
    assert error.value.code == 2


@pytest.mark.parametrize("channel", ["cu130", "cu132", "cu134"])
@pytest.mark.unit
def test_main_uses_selected_channel(monkeypatch, channel):
    calls = []
    monkeypatch.setattr(
        updater,
        "available_versions",
        lambda args: calls.append(args) or ["1.0.dev1+" + channel],
    )
    monkeypatch.setattr(updater, "read_pin", lambda field: "1.0.dev1")
    assert updater.main(["--channel", channel]) == 0
    assert calls == [
        ["--pre", "--index-url", f"https://download.pytorch.org/whl/nightly/{channel}"]
    ]


@pytest.mark.parametrize("allow", [False, True])
@pytest.mark.unit
def test_main_requires_explicit_downgrade_authority(monkeypatch, allow):
    monkeypatch.setattr(updater, "read_pin", lambda field: "1.5.0.dev1")
    monkeypatch.setattr(updater, "available_versions", lambda args: ["1.4.1"])
    downloads, writes = [], []
    monkeypatch.setattr(
        updater,
        "wheel_git_version",
        lambda version, args: downloads.append(version) or _COMMIT,
    )
    monkeypatch.setattr(
        updater,
        "write_pins",
        lambda version, commit: writes.append((version, commit)) or True,
    )
    result = updater.main(
        ["--track", "stable"] + (["--allow-downgrade"] if allow else [])
    )
    assert (result == 0) is allow
    assert downloads == (["1.4.1"] if allow else [])
    assert writes == ([("1.4.1", _COMMIT)] if allow else [])


def _contents(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }


@pytest.mark.unit
def test_write_pins_updates_real_sites_and_is_idempotent(pin_repo):
    old = updater.read_pin("__executorch_version__")
    new = f"{Version(old).major + 1}.0.0.dev1"
    assert updater.write_pins(new, _COMMIT)
    assert updater.read_pin("__executorch_version__") == new
    assert updater.read_pin("__executorch_commit__") == _COMMIT
    workflow = (pin_repo / ".github/workflows/executorch-test-linux.yml").read_text()
    assert f"executorch=={new}" in workflow
    assert f"executorch>={new},<{Version(new).major}.1" in workflow
    before = _contents(pin_repo)
    assert not updater.write_pins(new, _COMMIT)
    assert _contents(pin_repo) == before


@pytest.mark.parametrize("shape", ["missing", "untracked", "undecodable", "stale"])
@pytest.mark.unit
def test_write_pins_preflights_every_input(pin_repo, shape):
    path = pin_repo / "justfile"
    if shape == "missing":
        path.unlink()
    elif shape == "untracked":
        subprocess.run(
            ["git", "-C", str(pin_repo), "rm", "--cached", "justfile"], check=True
        )
    elif shape == "undecodable":
        path.write_bytes(b"\xff")
    else:
        path.write_text(
            path.read_text().replace(
                updater.read_pin("__executorch_version__"), "0.0.0"
            )
        )
    before = _contents(pin_repo)
    with pytest.raises(SystemExit, match="justfile"):
        updater.write_pins("9.0.0.dev1", _COMMIT)
    assert _contents(pin_repo) == before


@pytest.mark.parametrize(
    "template",
    [
        "ExecuTorch=={old}",
        "executorch[coreml] == {old}",
        "executorch < {upper}, >= {old}",
        "executorch >= {old}, < {upper}",
        'executorch=={old}; python_version >= "3.10"',
        'ExecuTorch[coreml] < {upper}, >= {old}; (sys_platform == "linux" or python_version >= "3.10")',
    ],
)
@pytest.mark.unit
def test_write_pins_handles_equivalent_requirements(pin_repo, template):
    old = updater.read_pin("__executorch_version__")
    upper = updater._upper_bound(old)
    path = pin_repo / "MODULE.bazel"
    requirement = template.format(old=old, upper=upper)
    path.write_text(path.read_text() + "\n# " + requirement + "\n")
    assert updater.write_pins("9.0.0.dev1", _COMMIT)
    expected = template.format(old="9.0.0.dev1", upper="9.1")
    assert expected in path.read_text()


@pytest.mark.unit
def test_write_pins_normalizes_equivalent_upper_bound(pin_repo):
    old = updater.read_pin("__executorch_version__")
    upper = updater._upper_bound(old)
    path = pin_repo / "MODULE.bazel"
    path.write_text(path.read_text() + f"\n# executorch >= {old}, < {upper}.0\n")
    assert updater.write_pins("9.0.0.dev1", _COMMIT)
    assert path.read_text().endswith("# executorch >= 9.0.0.dev1, < 9.1\n")


@pytest.mark.parametrize(
    "template",
    [
        "executorch=={old}+cu130",
        "executorch @ https://example.invalid/et.whl",
        "executorch~={old}",
        "executorch=={old},!={old}",
    ],
)
@pytest.mark.unit
def test_write_pins_rejects_unsupported_requirements_before_writing(pin_repo, template):
    path = pin_repo / "MODULE.bazel"
    path.write_text(
        path.read_text()
        + "\n# "
        + template.format(old=updater.read_pin("__executorch_version__"))
        + "\n"
    )
    before = _contents(pin_repo)
    with pytest.raises(SystemExit, match="MODULE.bazel"):
        updater.write_pins("9.0.0.dev1", _COMMIT)
    assert _contents(pin_repo) == before


@pytest.mark.unit
def test_write_pins_preserves_bystanders_and_version_prefixes(pin_repo):
    assert updater.write_pins("1.7.1", _COMMIT)
    skylib = 'bazel_dep(name = "bazel_skylib", version = "1.7.1")'
    assert skylib in (pin_repo / "MODULE.bazel").read_text()
    new = "1.7.1"
    module = pin_repo / "MODULE.bazel"
    bystanders = (
        f"# my-executorch=={new}\n# not_executorch=={new}\n# torch-tensorrt=={new}\n"
    )
    module.write_text(module.read_text() + bystanders)
    outside = pin_repo / "unrelated.txt"
    outside.write_text(f"executorch=={new}\n")
    before = {
        path: (pin_repo / path).read_bytes()
        for path in (str(_SCRIPT.relative_to(_REPO_ROOT)), _SELF, "unrelated.txt")
    }
    target = new + ".post1"
    assert updater.write_pins(target, "a" * 40)
    assert bystanders in module.read_text()
    assert skylib in module.read_text()
    assert target + ".post1" not in module.read_text()
    assert all(
        (pin_repo / path).read_bytes() == content for path, content in before.items()
    )


@pytest.mark.unit
def test_write_pins_preserves_yaml_formatting_and_other_fields(pin_repo):
    path = pin_repo / "dev_dep_versions.yml"
    text = path.read_text()
    old = updater.read_pin("__executorch_version__")
    text = text.replace(
        f'__executorch_version__: "{old}"',
        f"__executorch_version__ : '{old}' # preserve comment",
    )
    path.write_text(text)
    before = yaml.safe_load(text)
    assert updater.write_pins("9.0.0.dev1", _COMMIT)
    expected = dict(
        before, __executorch_version__="9.0.0.dev1", __executorch_commit__=_COMMIT
    )
    assert yaml.safe_load(path.read_text()) == expected
    assert (
        "__executorch_version__ : '9.0.0.dev1' # preserve comment" in path.read_text()
    )


@pytest.mark.unit
def test_write_pins_updates_the_development_constraint(pin_repo):
    import tomllib

    path = pin_repo / "pyproject.toml"
    assert updater.write_pins("9.0.0.dev1", _COMMIT)
    constraints = tomllib.loads(path.read_text())["tool"]["uv"][
        "constraint-dependencies"
    ]
    assert "executorch==9.0.0.dev1" in constraints


@pytest.mark.unit
def test_development_constraint_update_detects_removed_site(pin_repo, monkeypatch):
    monkeypatch.setattr(
        updater,
        "_PIN_SITES",
        tuple(name for name in updater._PIN_SITES if name != "pyproject.toml"),
    )
    with pytest.raises(AssertionError):
        test_write_pins_updates_the_development_constraint(pin_repo)


def test_write_pins_requires_a_separate_lock_refresh(tmp_path, monkeypatch):
    """A history-free pin bump must pass source guards and fail only the stale lock."""
    import xml.etree.ElementTree as ET

    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\0")
    for name in filter(None, tracked):
        source = _REPO_ROOT / name
        if not source.is_file():
            continue
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    _stage(tmp_path)
    monkeypatch.setattr(updater, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(updater, "_VERSIONS_FILE", tmp_path / "dev_dep_versions.yml")
    major = Version(updater.read_pin("__executorch_version__")).major
    lock = tmp_path / "uv.lock"
    locked = lock.read_bytes()
    assert updater.write_pins(f"{major + 1}.0.0.dev1", _COMMIT)
    assert lock.read_bytes() == locked
    report = tmp_path / "pin-guards.xml"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/py/dynamo/executorch/test_executorch_pin.py",
            "-q",
            "--no-header",
            f"--junitxml={report}",
            "-p",
            "no:cacheprovider",
            "--noconftest",
            "-o",
            "addopts=",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    cases = ET.parse(report).findall(".//testcase")
    assert not [case for case in cases if case.find("error") is not None], result.stdout
    assert [
        case.attrib["name"] for case in cases if case.find("failure") is not None
    ] == ["test_the_lockfile_executorch_matches_the_pin"], (
        result.stdout + result.stderr
    )
