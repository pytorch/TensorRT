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
import stat
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
        requires=[],
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
        # Keep what the file actually declares. Reading the source for the words cannot see a value
        # rewritten between the read and the call, which is how the pin lost its build label
        # unnoticed.
        state.requires = list(kwargs.get("install_requires") or [])
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
            / "cmake/executorch_backend_tensorrt/executorch_backend_tensorrt-config-version.cmake"
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
cmake = root / "lib/cmake/executorch_backend_tensorrt"
config = cmake / "executorch_backend_tensorrt-config.cmake"
version = cmake / "executorch_backend_tensorrt-config-version.cmake"
assert config.is_file()
assert 'set(PACKAGE_VERSION "0.2.0")' in version.read_text()
assert 'set(EXECUTORCH_BACKEND_TENSORRT_FULL_VERSION "0.2.0.dev20200103+cu130")' in version.read_text()
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
        "lib/cmake/executorch_backend_tensorrt/executorch_backend_tensorrt-config.cmake",
        "lib/cmake/executorch_backend_tensorrt/executorch_backend_tensorrt-config-version.cmake",
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
            "lib/cmake/executorch_backend_tensorrt/executorch_backend_tensorrt-config.cmake",
            "lib/cmake/executorch_backend_tensorrt/executorch_backend_tensorrt-config-version.cmake",
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
        f"{PACKAGE}/lib/cmake/executorch_backend_tensorrt/executorch_backend_tensorrt-config.cmake",
        f"{PACKAGE}/lib/cmake/executorch_backend_tensorrt/executorch_backend_tensorrt-config-version.cmake",
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
    packaging_build, installed, accepted
):
    """Matching the version is not matching the build.

    The delegate links the CUDA runtime out of the installed wheel, so a processor-only build of the
    pinned date cannot supply it. Comparing only the public parts of the two versions accepted that,
    and the wheel the build then published required a label the build had never checked.
    """
    state = packaging_build
    pinned = yaml.safe_load(
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


def _built_wheel_tag(built: Path) -> tuple[str, dict[str, str]]:
    """The tag in the filename, and the fields the archive's own WHEEL file records."""
    _, _, python_tag, abi_tag, platform_tag = built.name.removesuffix(".whl").split("-")
    with zipfile.ZipFile(built) as archive:
        name = next(
            member
            for member in archive.namelist()
            if member.endswith(".dist-info/WHEEL")
        )
        fields = dict(
            line.split(": ", 1)
            for line in archive.read(name).decode("utf-8").splitlines()
            if ": " in line
        )
    return f"{python_tag}-{abi_tag}-{platform_tag}", fields


def test_a_built_wheel_carries_the_tag_those_two_overrides_ask_for(
    packaging_build, tmp_path
):
    """Exercising the two classes says nothing about whether the build uses them.

    Both are wired in by one argument each to the setup call, and dropping either one left every
    check green while the built wheel took setuptools' own tag: one copy per interpreter, or a
    compiled object marked as pure Python. So this builds a wheel and reads the tag back off it.
    """
    wheels = tmp_path / "tagged-wheels"
    wheels.mkdir()
    built = wheels / build_meta.build_wheel(str(wheels))
    tag, fields = _built_wheel_tag(built)
    python_tag, abi_tag, platform_tag = tag.split("-")
    assert (python_tag, abi_tag) == ("py3", "none"), built.name
    # Whatever platform built it, but a platform: "any" is what a pure-Python wheel says.
    assert platform_tag != "any", built.name
    assert fields.get("Tag") == tag, fields
    # The payload is a compiled object, so it belongs in the platform-specific location.
    assert fields.get("Root-Is-Purelib") == "false", fields


@pytest.mark.parametrize("dropped", ["distclass", "cmdclass"])
def test_the_wheel_tag_read_back_notices_either_override_being_dropped(
    packaging_build, tmp_path, dropped
):
    """The read-back is only worth having if losing the wiring fails it."""
    setup_py = packaging_build.project / "setup.py"
    source = setup_py.read_text(encoding="utf-8")
    if dropped == "distclass":
        wiring, without = "    distclass=PlatformDistribution,\n", ""
    else:
        wiring, without = (
            '    cmdclass={"build_py": BazelBuild, "bdist_wheel": WheelTag},\n',
            '    cmdclass={"build_py": BazelBuild},\n',
        )
    assert wiring in source, source[-600:]
    setup_py.write_text(source.replace(wiring, without), encoding="utf-8")
    with pytest.raises(AssertionError):
        test_a_built_wheel_carries_the_tag_those_two_overrides_ask_for(
            packaging_build, tmp_path
        )


@pytest.mark.unit
def test_the_three_pinned_runtimes_keep_their_build_labels() -> None:
    """A version without its label admits a processor build and any other build of the same date.

    ExecuTorch is the one the delegate is compiled and linked against, so a requirement another
    build satisfies is not a pin at all. PyTorch and Torch-TensorRT are not linked here; they share
    a process with a delegate that links one CUDA runtime and one TensorRT, and the label is the
    only part of a version that names the row those came from.
    """
    source = (COMPANION / "setup.py").read_text(encoding="utf-8")
    requires = source.split("install_requires=[", 1)[1].split("]", 1)[0]
    for name, expression in (
        ("torch", 'f"torch=={torch.__version__}"'),
        ("executorch", 'f"executorch=={executorch_version}"'),
        (
            "torch-tensorrt",
            "f\"torch-tensorrt=={installed_version('torch-tensorrt')}\"",
        ),
    ):
        assert (
            expression in requires
        ), f"{name} is not pinned with its label: {requires}"
    # And the two that legitimately have no label keep the public form.
    assert "public_version(tensorrt_version)" in requires, requires
    assert "public_version(cuda_runtime_version)" in requires, requires


@pytest.mark.unit
def test_the_declared_pins_keep_their_build_labels(packaging_build, monkeypatch):
    """Reading the file for the words cannot see a value rewritten before it is used.

    Dropping the label from the ExecuTorch pin, while leaving every word the old checks looked for,
    changed the requirement from one build to any build of that date and went unnoticed. So the
    requirement the file actually declares is read back here.
    """
    state = packaging_build
    monkeypatch.setattr(sys, "argv", [str(state.project / "setup.py"), "--name"])
    runpy.run_path(str(state.project / "setup.py"), run_name="__main__")
    for name in ("torch", "executorch", "torch-tensorrt"):
        requirement = [r for r in state.requires if r.startswith(f"{name}==")]
        assert requirement, state.requires
        assert "+cu" in requirement[0], requirement[0]


@pytest.mark.unit
def test_a_missing_pin_file_stops_the_build(tmp_path):
    """The refusal was checked by looking for words, and its branch never ran.

    Rewriting it to fall back to the installed version, and replacing its body with something that
    would be obvious, both left the suite green. So the branch is executed here, with no pin file
    present.
    """
    source = (COMPANION / "setup.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    reader = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "pinned_executorch_version"
    )
    namespace: dict[str, object] = {"REPO_ROOT": tmp_path, "yaml": yaml}
    exec(
        compile(ast.Module(body=[reader], type_ignores=[]), "<setup>", "exec"),
        namespace,
    )
    # No pin file in tmp_path, which is the case the refusal exists for.
    with pytest.raises(RuntimeError, match="is missing"):
        namespace["pinned_executorch_version"]()
    # And with one present it returns the pinned version rather than guessing.
    (tmp_path / "dev_dep_versions.yml").write_text(
        '__executorch_version__: "1.6.0.dev20260915+cu134"\n', encoding="utf-8"
    )
    assert namespace["pinned_executorch_version"]() == "1.6.0.dev20260915+cu134"


@pytest.mark.unit
def test_a_second_build_in_the_same_tree_succeeds(tmp_path):
    """Building twice without cleaning failed on the first build's own output.

    The build system leaves its output read only and copy2 carries the mode across, so the second
    build could not overwrite what the first one left. The sweep before the copy also skipped the
    destination, which is the one file that needed removing.

    The whole build needs bazel, torch and Linux, so it cannot run here. What runs instead is the
    three statements the build itself uses, lifted out of its own parsed source and executed twice
    over a read only fixture, so deleting one of them or narrowing its glob turns this red rather
    than passing against a copy of the idea.
    """
    source = (COMPANION / "setup.py").read_text(encoding="utf-8")
    build = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "_build"
    )
    wanted = (
        'for stale in output.parent.glob("*.so*")',
        "shutil.copy2(built, output)",
        "output.chmod(",
    )
    # Whole statements out of the parsed function, so a commented-out line is not there to find,
    # and the destination sweep cannot come back with the exemption that caused the failure.
    routine = [
        text
        for text in (ast.get_source_segment(source, node) or "" for node in build.body)
        if text.startswith(wanted)
    ]
    assert len(routine) == len(wanted), routine
    assert "if stale != output:" not in routine[0], routine[0]

    built = tmp_path / "built.so"
    built.write_bytes(b"\x7fELF")
    built.chmod(0o555)
    output = tmp_path / "out" / "built.so"
    output.parent.mkdir()
    # A shared object the previous build left behind under a versioned name. The sweep exists to
    # take these out of the wheel, and the copy alone would leave it there.
    (output.parent / "libstale.so.1").write_bytes(b"\x7fELF")
    namespace = {"shutil": shutil, "stat": stat, "built": built, "output": output}
    for _ in range(2):
        for text in routine:
            exec(text, namespace)
    assert output.stat().st_mode & stat.S_IWUSR, oct(output.stat().st_mode)
    assert sorted(p.name for p in output.parent.iterdir()) == ["built.so"]
