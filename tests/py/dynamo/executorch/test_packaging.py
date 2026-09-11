"""Exercise companion setup with controlled dependencies and native outputs."""

import ast
import importlib.metadata
import runpy
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import setuptools
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).parents[4]
COMPANION = REPO_ROOT / "py/torch-tensorrt-executorch-runtime"
PACKAGE = "torch_tensorrt_executorch_runtime"
LIBRARY = "libexecutorch_backend_tensorrt.so"
FLAG_VALUES = [
    (None, False),
    ("", False),
    ("0", False),
    ("false", False),
    ("FALSE", False),
    ("off", False),
    ("other", False),
    (" true ", False),
    ("1", True),
    ("true", True),
    ("TrUe", True),
    ("yes", True),
    ("YES", True),
    ("on", True),
    ("ON", True),
]


@pytest.fixture
def packaging_build(tmp_path, monkeypatch):
    project = tmp_path / "checkout/py/torch-tensorrt-executorch-runtime"
    shutil.copytree(
        COMPANION,
        project,
        ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__"),
    )
    shutil.copyfile(
        REPO_ROOT / "dev_dep_versions.yml", project.parents[1] / "dev_dep_versions.yml"
    )
    pinned = yaml.safe_load((project.parents[1] / "dev_dep_versions.yml").read_text())[
        "__executorch_version__"
    ]
    versions = {
        "executorch": pinned,
        "torch-tensorrt": "2.15.0.dev20200103+cu130",
        "tensorrt-cu13": "11.2.1",
        "nvidia-cuda-runtime": "13.0.0",
    }
    dependency_root = tmp_path / "dependencies"
    cmake = dependency_root / "executorch/share/cmake"
    cmake.mkdir(parents=True)
    (cmake / "executorch-config.cmake").touch()
    torch = types.ModuleType("torch")
    torch.__file__ = str(dependency_root / "torch/__init__.py")
    torch.__version__ = "2.15.0.dev20200103+cu130"
    torch.version = types.SimpleNamespace(cuda="13.0")
    monkeypatch.setitem(sys.modules, "torch", torch)
    original_version = importlib.metadata.version
    original_distribution = importlib.metadata.distribution
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: versions[name] if name in versions else original_version(name),
    )
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: (
            types.SimpleNamespace(
                version=versions[name], locate_file=lambda path: dependency_root / path
            )
            if name == "executorch"
            else original_distribution(name)
        ),
    )
    for name in (
        "TORCH_TENSORRT_ALLOW_UNPINNED_EXECUTORCH",
        "TORCH_TENSORRT_EXECUTORCH_DEBUG",
        "BAZEL_ARGS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(
        "TORCH_TENSORRT_EXECUTORCH_RUNTIME_VERSION", "0.2.0.dev20200103+cu130"
    )
    monkeypatch.chdir(project)
    monkeypatch.setattr(sys, "argv", [str(project / "setup.py"), "build_py"])
    state = types.SimpleNamespace(
        project=project,
        versions=versions,
        calls=[],
        builds=[],
        payload=b"controlled native output",
    )
    bazel_bin = tmp_path / "bazel-bin"
    original_which = shutil.which
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: (
            "/controlled/bazel"
            if name in {"bazel", "bazelisk"}
            else original_which(name)
        ),
    )
    original_run = subprocess.run
    original_output = subprocess.check_output

    def produce(command, **kwargs):
        if command[0] != "/controlled/bazel":
            return original_run(command, **kwargs)
        state.calls.append(command)
        output = (
            bazel_bin
            / "py/torch-tensorrt-executorch-runtime/native/delegate_native/lib"
            / LIBRARY
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(state.payload)
        return subprocess.CompletedProcess(command, 0)

    def bazel_info(command, **kwargs):
        if command[0] != "/controlled/bazel":
            return original_output(command, **kwargs)
        state.calls.append(command)
        return str(bazel_bin)

    monkeypatch.setattr(subprocess, "run", produce)
    monkeypatch.setattr(subprocess, "check_output", bazel_info)
    original_setup = setuptools.setup

    def setup(**kwargs):
        # Exercise Linux-only command behavior without changing setuptools' host wheel tags.
        kwargs["cmdclass"]["build_py"].run.__globals__["sys"] = types.SimpleNamespace(
            platform="linux", executable=sys.executable
        )
        distribution = original_setup(**kwargs)
        state.builds.append(distribution.get_command_obj("build_py"))
        return distribution

    monkeypatch.setattr(setuptools, "setup", setup)
    return state


@pytest.mark.parametrize("value,enabled", FLAG_VALUES)
@pytest.mark.parametrize("flag", ["unpinned", "debug"])
def test_setup_boolean_flags(packaging_build, monkeypatch, flag, value, enabled):
    state = packaging_build
    name = (
        "TORCH_TENSORRT_ALLOW_UNPINNED_EXECUTORCH"
        if flag == "unpinned"
        else "TORCH_TENSORRT_EXECUTORCH_DEBUG"
    )
    if value is not None:
        monkeypatch.setenv(name, value)
    if flag == "unpinned":
        state.versions["executorch"] = "1.4.0"
    if flag == "unpinned" and not enabled:
        with pytest.raises(RuntimeError, match="pins"):
            runpy.run_path(str(state.project / "setup.py"), run_name="__main__")
        assert state.calls == []
    else:
        runpy.run_path(str(state.project / "setup.py"), run_name="__main__")
        mode = "dbg" if flag == "debug" and enabled else "opt"
        assert len(state.calls) == 2
        assert all(f"--compilation_mode={mode}" in call for call in state.calls)
        build = state.builds[-1]
        output = Path(build.build_lib) / PACKAGE / "lib" / LIBRARY
        assert output.read_bytes() == state.payload
        assert (
            output.parent
            / "cmake/torchtrt_executorch/torchtrt_executorch-config-version.cmake"
        ).is_file()


@pytest.mark.parametrize("flag", ["unpinned", "debug"])
def test_setup_flags_reject_string_presence_control(packaging_build, monkeypatch, flag):
    path = packaging_build.project / "setup.py"
    tree = ast.parse(path.read_text())
    name = (
        "TORCH_TENSORRT_ALLOW_UNPINNED_EXECUTORCH"
        if flag == "unpinned"
        else "TORCH_TENSORRT_EXECUTORCH_DEBUG"
    )
    checks = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Call)
        and any(
            isinstance(child, ast.Constant) and child.value == name
            for child in ast.walk(node.left)
        )
    ]
    assert len(checks) == 1
    check = checks[0]
    check.left = ast.parse(f"os.getenv({name!r})", mode="eval").body
    check.ops = [ast.Is() if flag == "unpinned" else ast.IsNot()]
    check.comparators = [ast.Constant(value=None)]
    path.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    failure = pytest.fail.Exception if flag == "unpinned" else AssertionError
    with pytest.raises(failure):
        test_setup_boolean_flags(packaging_build, monkeypatch, flag, "false", False)
