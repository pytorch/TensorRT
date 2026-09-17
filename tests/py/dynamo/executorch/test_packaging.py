# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise companion setup with controlled dependencies and native outputs."""

import ast
import importlib.metadata
import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
import types
import zipfile
from pathlib import Path

import pytest
import setuptools
import yaml
from setuptools import build_meta

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
        ignore=shutil.ignore_patterns(
            "build", "dist", "*.egg-info", "__pycache__", LIBRARY
        ),
    )
    shutil.copyfile(
        REPO_ROOT / "dev_dep_versions.yml", project.parents[1] / "dev_dep_versions.yml"
    )
    pinned = yaml.safe_load((project.parents[1] / "dev_dep_versions.yml").read_text())[
        "__executorch_version__"
    ]
    versions = {
        # With the label a real CUDA wheel carries. A bare version here is what a processor-only
        # build looks like, and the build refuses that, correctly.
        "executorch": f"{pinned}+cu130",
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
        # SystemExit, not RuntimeError: the build converts every failure to the one class
        # setuptools does not swallow during an editable install, so this refusal reaches the user
        # instead of becoming a warning pip hides.
        with pytest.raises(SystemExit, match="pins"):
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


def _run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    print(shlex.join(map(str, command)))
    print(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture
def native_payload(packaging_build, tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for the packaging load probe")
    source = tmp_path / "probe.c"
    source.write_text("int packaging_probe(void) { return 42; }\n")
    library = tmp_path / LIBRARY
    _run([compiler, "-shared", "-fPIC", str(source), "-o", str(library)])
    packaging_build.payload = library.read_bytes()


def _install_and_probe(state, wheel, environment, tmp_path, mode):
    _run([sys.executable, "-m", "venv", "--without-pip", str(environment)])
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "--python",
            str(python),
            "install",
            "--no-deps",
            "--no-index",
            "--disable-pip-version-check",
            str(wheel),
        ]
    )
    probe = """
import ctypes
import hashlib
import json
from pathlib import Path
import torch_tensorrt_executorch_runtime as runtime
root = Path(runtime.__file__).absolute().parent
library = Path(runtime._delegate_path())
handle = ctypes.CDLL(str(library))
assert handle.packaging_probe() == 42
cmake = root / "lib/cmake/torchtrt_executorch"
config = cmake / "torchtrt_executorch-config.cmake"
version = cmake / "torchtrt_executorch-config-version.cmake"
assert config.is_file()
assert 'set(PACKAGE_VERSION "0.2.0")' in version.read_text()
assert 'set(TORCHTRT_EXECUTORCH_FULL_VERSION "0.2.0.dev20200103+cu130")' in version.read_text()
print(json.dumps({"module": str(root), "library": str(library),
                  "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                  "config": str(config), "version": str(version)}))
"""
    output = _run(
        [str(python), "-I", "-B", "-c", probe],
        cwd=tmp_path,
        env={**os.environ, "TORCH_TENSORRT_SKIP_DELEGATE_REGISTRATION": "1"},
    )
    data = json.loads(output)
    root = Path(data["module"])
    if mode == "default":
        assert root == state.project / PACKAGE
    elif mode == "strict":
        assert root.is_relative_to(state.project / "build")
    else:
        assert root.is_relative_to(environment)
    assert Path(data["library"]).read_bytes() == state.payload


@pytest.mark.parametrize("mode", ["default", "strict"])
def test_editable_outputs_survive_backend_cleanup(
    packaging_build, native_payload, monkeypatch, tmp_path, mode
):
    state = packaging_build
    temporary = tmp_path / "backend-temporary"
    temporary.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    settings = {"editable_mode": "strict"} if mode == "strict" else None
    wheel = wheels / build_meta.build_editable(str(wheels), settings)
    build = state.builds[-1]
    assert build.editable_mode
    assert len(state.calls) == 2
    assert not Path(build.build_lib).exists()
    assert list(temporary.iterdir()) == []
    print(f"Removed temporary build_lib: {build.build_lib}")
    _install_and_probe(state, wheel, tmp_path / "installed", tmp_path, mode)
    generated = {
        f"lib/{LIBRARY}",
        "lib/cmake/torchtrt_executorch/torchtrt_executorch-config.cmake",
        "lib/cmake/torchtrt_executorch/torchtrt_executorch-config-version.cmake",
    }
    mapping = build.get_output_mapping()
    outputs = build.get_outputs()
    for filename in generated:
        destination = str(Path(build.build_lib) / PACKAGE / filename)
        assert destination in outputs
        assert (
            Path(mapping[destination]).resolve() == state.project / PACKAGE / filename
        )
        assert Path(mapping[destination]).is_file()


def _remove_editable_fix(path):
    tree = ast.parse(path.read_text())
    build = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BazelBuild"
    )
    removed = {"_generated_output_mapping", "get_outputs", "get_output_mapping"}
    assert removed.issubset({getattr(node, "name", None) for node in build.body})
    build.body = [
        node for node in build.body if getattr(node, "name", None) not in removed
    ]
    branches = [
        node
        for node in ast.walk(build)
        if isinstance(node, ast.IfExp)
        and isinstance(node.test, ast.Attribute)
        and node.test.attr == "editable_mode"
    ]
    assert len(branches) == 1
    branches[0].test = ast.Constant(value=False)
    path.write_text(ast.unparse(ast.fix_missing_locations(tree)))


@pytest.mark.parametrize("mode", ["default", "strict"])
def test_editable_install_rejects_removed_fix(
    packaging_build, native_payload, monkeypatch, tmp_path, mode
):
    _remove_editable_fix(packaging_build.project / "setup.py")
    with pytest.raises(AssertionError, match="delegate library is missing"):
        test_editable_outputs_survive_backend_cleanup(
            packaging_build, native_payload, monkeypatch, tmp_path, mode
        )


@pytest.mark.parametrize("editable", [False, True])
def test_generated_outputs_are_reported_before_build(
    packaging_build, monkeypatch, editable
):
    state = packaging_build
    monkeypatch.setattr(sys, "argv", [str(state.project / "setup.py"), "--name"])
    runpy.run_path(str(state.project / "setup.py"), run_name="__main__")
    command = state.builds[-1]
    command.ensure_finalized()
    command.editable_mode = editable
    expected = {
        str(Path(command.build_lib) / PACKAGE / filename)
        for filename in (
            f"lib/{LIBRARY}",
            "lib/cmake/torchtrt_executorch/torchtrt_executorch-config.cmake",
            "lib/cmake/torchtrt_executorch/torchtrt_executorch-config-version.cmake",
        )
    }
    outputs = command.get_outputs(include_bytecode=False)
    assert expected.issubset(outputs)
    mapping = command.get_output_mapping()
    assert expected.issubset(mapping) is editable
    command.run()
    for filename in expected:
        output = Path(mapping[filename]) if editable else Path(filename)
        assert output.is_file()


@pytest.mark.parametrize("editable", [False, True])
def test_generated_cleanup_stays_in_the_output_location(
    packaging_build, monkeypatch, editable
):
    state = packaging_build
    monkeypatch.setattr(sys, "argv", [str(state.project / "setup.py"), "--name"])
    runpy.run_path(str(state.project / "setup.py"), run_name="__main__")
    command = state.builds[-1]
    command.ensure_finalized()
    command.editable_mode = editable
    root = state.project / PACKAGE if editable else Path(command.build_lib) / PACKAGE
    (root / "lib").mkdir(parents=True, exist_ok=True)
    stale = root / "lib/libexecutorch.so"
    stale.write_bytes(b"obsolete private runtime")
    # A stale shared object from the old layout, which must go, and the forwarder the released main
    # wheel imports by name, which must not. They used to be swept together, so the published wheel
    # shipped without the forwarder and that import failed.
    legacy = [root / "old_delegate.so"]
    shipped = root / "runtime.py"
    for path in [*legacy, shipped]:
        path.write_bytes(b"root output")
    bystanders = [
        state.project / "bystander.so",
        root / "lib/notes.txt",
        root.parent / "other/libexecutorch.so",
    ]
    for path in bystanders:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"keep")
    source = state.project / PACKAGE / "__init__.py"
    original = source.read_bytes()
    command.run()
    assert not stale.exists()
    assert (root / "lib" / LIBRARY).read_bytes() == state.payload
    assert source.read_bytes() == original
    assert all(path.read_bytes() == b"keep" for path in bystanders)
    assert all(path.exists() is editable for path in legacy)
    assert (
        shipped.exists()
    ), "the forwarder the main wheel imports was removed from the wheel"


def test_editable_cleanup_rejects_source_deletion_control(packaging_build, monkeypatch):
    path = packaging_build.project / "setup.py"
    tree = ast.parse(path.read_text())
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and isinstance(node.test.operand, ast.Attribute)
        and node.test.operand.attr == "editable_mode"
    ]
    assert len(guards) == 1
    guards[0].test = ast.Constant(value=True)
    path.write_text(ast.unparse(tree))
    with pytest.raises(AssertionError):
        test_generated_cleanup_stays_in_the_output_location(
            packaging_build, monkeypatch, True
        )


def test_ordinary_wheel_payload_is_unchanged(
    packaging_build, native_payload, monkeypatch, tmp_path
):
    state = packaging_build
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    built = wheels / build_meta.build_wheel(str(wheels))
    with zipfile.ZipFile(built) as archive:
        expected = {name: archive.read(name) for name in archive.namelist()}
    files = {name for name in expected if name.startswith(f"{PACKAGE}/")}
    assert files == {
        f"{PACKAGE}/__init__.py",
        # The forwarder the released main wheel imports by name. This set omitting it is what let a
        # published wheel ship without it, since this is the only check that reads the built archive.
        f"{PACKAGE}/runtime.py",
        f"{PACKAGE}/lib/{LIBRARY}",
        f"{PACKAGE}/lib/cmake/torchtrt_executorch/torchtrt_executorch-config.cmake",
        f"{PACKAGE}/lib/cmake/torchtrt_executorch/torchtrt_executorch-config-version.cmake",
    }
    assert expected[f"{PACKAGE}/lib/{LIBRARY}"] == state.payload
    _install_and_probe(state, built, tmp_path / "installed", tmp_path, "wheel")
    # Use a separate source/build tree so no output can be reused in the comparison.
    control_project = tmp_path / "control-checkout/py/torch-tensorrt-executorch-runtime"
    shutil.copytree(
        state.project,
        control_project,
        ignore=shutil.ignore_patterns("build", "*.egg-info"),
    )
    shutil.copyfile(
        state.project.parents[1] / "dev_dep_versions.yml",
        control_project.parents[1] / "dev_dep_versions.yml",
    )
    _remove_editable_fix(control_project / "setup.py")
    monkeypatch.chdir(control_project)
    control = tmp_path / "control-wheels"
    control.mkdir()
    old = control / build_meta.build_wheel(str(control))
    with zipfile.ZipFile(old) as archive:
        actual = {name: archive.read(name) for name in archive.namelist()}
    assert actual == expected
    print(
        f"Ordinary wheel has identical bytes for {len(expected)} members with editable fix removed"
    )


@pytest.mark.unit
def test_the_executorch_requirement_keeps_its_cuda_label() -> None:
    """A requirement without the local label is satisfied by a build the delegate cannot use.

    The delegate links one specific ExecuTorch build, so pinning the version while dropping the
    part that names the CUDA variant leaves a pin that a CPU build, or another CUDA build of the
    same date, resolves against happily.
    """
    source = (COMPANION / "setup.py").read_text(encoding="utf-8")
    assert 'f"executorch=={executorch_version}"' in source, source[-400:]
    assert (
        'f"executorch=={public_version(executorch_version)}"' not in source
    ), "the local label is being stripped again"


@pytest.mark.unit
def test_a_missing_pin_file_fails_rather_than_disabling_the_check() -> None:
    """Returning an empty version switched the pin check off instead of failing it."""
    source = (COMPANION / "setup.py").read_text(encoding="utf-8")
    reader = source.split("def pinned_executorch_version")[1].split("\ndef ")[0]
    assert "raise RuntimeError" in reader, reader
    assert 'return ""' not in reader, reader


@pytest.mark.parametrize(
    "installed,accepted",
    [("+cu130", True), ("+cu134", True), ("+cpu", False), ("", False)],
)
def test_the_build_refuses_an_executorch_that_is_not_a_cuda_build(
    packaging_build, monkeypatch, installed, accepted
):
    """Matching the version is not matching the build.

    The delegate links the CUDA runtime out of the installed wheel, so a processor-only build of the
    pinned date cannot supply it. Comparing only the public parts of the two versions accepted that,
    and the wheel the build then published required a label the build had never checked.
    """
    state = packaging_build
    pinned = updater_pin = yaml.safe_load(
        (state.project.parents[1] / "dev_dep_versions.yml").read_text()
    )["__executorch_version__"]
    state.versions["executorch"] = f"{pinned}{installed}"
    if accepted:
        runpy.run_path(str(state.project / "setup.py"), run_name="__main__")
    else:
        with pytest.raises(SystemExit, match="CUDA build"):
            runpy.run_path(str(state.project / "setup.py"), run_name="__main__")


@pytest.mark.unit
def test_the_wheel_is_tagged_for_any_python_and_one_platform() -> None:
    """Neither of these two overrides had a test, so dropping either changed the built wheel.

    The payload is one shared library loaded through ctypes, with no Python ABI, so it is identical
    across CPython versions and only the platform matters. Losing the tag override would build one
    identical copy per interpreter. Losing the platform marking would tag a compiled object as pure
    Python and let it install on the wrong architecture.
    """
    source = (COMPANION / "setup.py").read_text(encoding="utf-8")
    namespace: dict[str, object] = {}
    tree = ast.parse(source)
    wanted = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name in ("WheelTag", "PlatformDistribution")
    ]
    assert len(wanted) == 2, [n.name for n in wanted]
    # Run the two class bodies against stub bases, so the behaviour is exercised rather than read.
    namespace["bdist_wheel"] = type(
        "StubBdist",
        (),
        {"get_tag": lambda self: ("cp312", "cp312", "manylinux_2_28_x86_64")},
    )
    namespace["Distribution"] = type(
        "StubDistribution", (), {"has_ext_modules": lambda self: False}
    )
    exec(
        compile(ast.Module(body=wanted, type_ignores=[]), "<setup>", "exec"), namespace
    )
    assert namespace["WheelTag"]().get_tag() == ("py3", "none", "manylinux_2_28_x86_64")
    assert namespace["PlatformDistribution"]().has_ext_modules() is True
