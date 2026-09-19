# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only command-boundary tests for the shared companion workflows."""

import ast
import importlib
import json
import os
import shutil
import subprocess
import sys
import types
import warnings
from pathlib import Path

import pytest
import yaml
from wheel.wheelfile import WheelFile

from packaging.requirements import Requirement

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[4]


def _workflow(name):
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())


def _run(script, root, env):
    result = subprocess.run(
        [shutil.which("bash"), "-c", script],
        cwd=root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **env},
        text=True,
        capture_output=True,
        timeout=30,
    )
    (root / "stdout.log").write_text(result.stdout)
    (root / "stderr.log").write_text(result.stderr)
    return result


def _assert_test_wheel_dependency(command):
    requirements = [Requirement(arg) for arg in command if arg.startswith("wheel")]
    assert len(requirements) == 1, "ExecuTorch tests require wheel>=0.40"
    assert "0.40.0" in requirements[0].specifier
    assert "0.37.1" not in requirements[0].specifier


@pytest.mark.parametrize("cuda", ["cu130", "cu132"])
def test_manifest_suite_provisions_wheel(monkeypatch, cuda):
    monkeypatch.syspath_prepend(str(ROOT))
    from tests.ci import runner

    monkeypatch.setenv("CU_VERSION", cuda)
    commands = runner._setup_commands("executorch")
    command, _ = next((argv, cwd) for argv, cwd in commands if "install" in argv)
    _assert_test_wheel_dependency(command)


@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
@pytest.mark.parametrize("cuda", ["cu130", "cu132"])
@pytest.mark.parametrize("release", [True, False])
def test_shared_build_provisions_tensorrt_metadata(tmp_path, arch, cuda, release):
    """Start with no TensorRT metadata; run the actual build step up to native build."""
    # Use the main wheel's real architecture-specific dependency selector.
    source = ast.parse((ROOT / "setup.py").read_text())
    selector = (
        "get_sbsa_requirements" if arch == "aarch64" else "get_x86_64_requirements"
    )
    function = next(
        n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == selector
    )
    scope = {
        "IS_DLFW_CI": False,
        "USE_TRT_RTX": False,
        "torch": types.SimpleNamespace(
            version=types.SimpleNamespace(cuda={"cu130": "13.0", "cu132": "13.2"}[cuda])
        ),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "<requirements>", "exec"),
        scope,
    )
    requirements = scope[selector]([])
    expected = [r for r in requirements if Requirement(r).name.startswith("tensorrt")]
    assert expected

    site = tmp_path / "site"
    site.mkdir()
    dist = tmp_path / "dist"
    dist.mkdir()
    version = (ROOT / "version.txt").read_text().strip().removesuffix("a0")
    name = f"torch_tensorrt-{version}"
    with WheelFile(dist / f"{name}-py3-none-any.whl", "w") as wheel:
        wheel.writestr("torch_tensorrt/lib/libtorchtrt.so", b"native fixture")
        wheel.writestr(
            "torch_tensorrt/__init__.py", "raise AssertionError('compiler imported')\n"
        )
        wheel.writestr(
            f"{name}.dist-info/WHEEL", "Wheel-Version: 1.0\nTag: py3-none-any\n"
        )
        wheel.writestr(
            f"{name}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: torch-tensorrt\nVersion: {version}\n"
            + "".join(f"Requires-Dist: {r}\n" for r in requirements),
        )
    # Only the package-manager boundary is fake. Metadata lookups and selection run normally.
    # The installed version comes from the requirement the real selector produced, so a
    # TensorRT upgrade does not need editing here.
    lower_bound = next(
        specifier
        for specifier in Requirement(expected[0]).specifier
        if specifier.operator == ">="
    )
    installed_version = lower_bound.version.removesuffix(".0")
    (site / "pip.py").write_text(
        "import importlib.metadata as m, json, os, sys, zipfile\n"
        "from pathlib import Path\n"
        "from packaging.requirements import Requirement\n"
        "site = Path(__file__).parent\n"
        "args = sys.argv[1:]\n"
        f"pinned = {installed_version!r}\n"
        "with open(os.environ['EVENTS'], 'a') as f: f.write(json.dumps(args) + '\\n')\n"
        "if args[0] == 'install':\n"
        "    for arg in args[1:]:\n"
        "        if arg.endswith('.whl'):\n"
        "            with zipfile.ZipFile(arg) as w: w.extractall(site)\n"
        "        elif arg.startswith('tensorrt'):\n"
        "            r = Requirement(arg)\n"
        "            assert r.specifier.contains(pinned), (arg, pinned)\n"
        "            for name in (r.name, 'tensorrt-cu13', 'tensorrt-cu13-bindings', 'tensorrt-cu13-libs'):\n"
        "                info = site / (name.replace('-', '_') + '-' + pinned + '.dist-info')\n"
        "                info.mkdir(exist_ok=True)\n"
        "                (info / 'METADATA').write_text(f'Name: {name}\\nVersion: {pinned}\\n')\n"
        "elif args[0] == 'wheel':\n"
        "    installed = {d.metadata['Name']: d.version for d in m.distributions(path=[site])}\n"
        "    if 'tensorrt-cu13' not in installed: raise m.PackageNotFoundError('tensorrt-cu13')\n"
        "    assert installed['tensorrt-cu13'] == pinned\n"
        "    assert installed['tensorrt-cu13-libs'] == pinned\n"
        "    Path(os.environ['BUILT_VERSION']).write_text(os.environ['TORCH_TENSORRT_EXECUTORCH_RUNTIME_VERSION'])\n"
        "else: raise AssertionError(args)\n"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (bin_dir / "python").chmod(0o755)
    (tmp_path / "build-env").write_text("export CONDA_RUN=''\n")
    shutil.copy2(ROOT / "version.txt", tmp_path / "version.txt")
    runtime = tmp_path / "py/torch-tensorrt-executorch-runtime"
    runtime.mkdir(parents=True)
    (runtime / "version.txt").write_text("7.4.1\n")
    scripts = tmp_path / ".github/scripts"
    scripts.mkdir(parents=True)
    step = next(
        s
        for s in _workflow("build_linux.yml")["jobs"]["build"]["steps"]
        if s.get("id") == "executorch-runtime"
    )
    script = step["run"].replace("${{ inputs.is-release-wheel }}", str(release).lower())
    result = _run(
        script,
        tmp_path,
        {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "PYTHONPATH": str(site),
            "BUILD_ENV_FILE": str(tmp_path / "build-env"),
            "BUILD_VERSION": version if release else f"{version}.dev20260911",
            "CU_VERSION": cuda,
            "ARCH": arch,
            "EVENTS": str(tmp_path / "events"),
            "BUILT_VERSION": str(tmp_path / "built-version"),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    events = [
        json.loads(line) for line in (tmp_path / "events").read_text().splitlines()
    ]
    selected = [
        arg
        for event in events
        if event[0] == "install"
        for arg in event[1:]
        if arg.startswith("tensorrt")
    ]
    assert [Requirement(r) for r in selected] == [Requirement(r) for r in expected]
    assert events[0][:2] == ["install", "--no-deps"]
    wheel_requirements = [
        Requirement(arg)
        for event in events
        if event[0] == "install"
        for arg in event[1:]
        if arg.startswith("wheel")
    ]
    assert len(wheel_requirements) == 1
    assert (
        "0.37.1" not in wheel_requirements[0].specifier
    ), "wheel tags needs wheel>=0.40"
    assert "0.40.0" in wheel_requirements[0].specifier
    assert events[-1] == [
        "wheel",
        "--no-build-isolation",
        "--no-deps",
        "--wheel-dir",
        "dist",
        "py/torch-tensorrt-executorch-runtime",
    ]
    assert (tmp_path / "built-version").read_text() == (
        "7.4.1" if release else f"7.4.1.dev20260911+{cuda}"
    )


@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
def test_missing_wheel_tool_minimum_is_detected(tmp_path, monkeypatch, arch):
    workflow = _workflow("build_linux.yml")
    step = next(
        s
        for s in workflow["jobs"]["build"]["steps"]
        if s.get("id") == "executorch-runtime"
    )
    assert step["run"].count('"wheel>=0.40"') == 1
    step["run"] = step["run"].replace('"wheel>=0.40"', "wheel")
    monkeypatch.setitem(globals(), "_workflow", lambda _: workflow)
    with pytest.raises(AssertionError, match="wheel tags needs"):
        test_shared_build_provisions_tensorrt_metadata(tmp_path, arch, "cu132", True)


@pytest.mark.parametrize("arch", ["x86_64", "aarch64"])
def test_missing_tensorrt_provisioning_is_detected(tmp_path, monkeypatch, arch):
    workflow = _workflow("build_linux.yml")
    step = next(
        s
        for s in workflow["jobs"]["build"]["steps"]
        if s.get("id") == "executorch-runtime"
    )
    command = (
        'subprocess.check_call([sys.executable, "-m", "pip", "install", *tensorrt])'
    )
    assert step["run"].count(command) == 1
    step["run"] = step["run"].replace(command, "pass")
    monkeypatch.setitem(
        test_shared_build_provisions_tensorrt_metadata.__globals__,
        "_workflow",
        lambda _: workflow,
    )
    with pytest.raises(
        AssertionError, match="No package metadata was found for tensorrt-cu13"
    ):
        test_shared_build_provisions_tensorrt_metadata(tmp_path, arch, "cu132", True)


_DEVICE_EXPORT = 'python examples/torchtrt_executorch_example/export_device_resident.py \\\n  --model_path="${RUNNER_TEMP}/torchtrt-device-resident.pte"\n'
_DEVICE_RUN = 'python examples/executorch_reference_runner/load_model_device_resident.py \\\n  --model_path="${RUNNER_TEMP}/torchtrt-device-resident.pte" --num_runs=2\n'


def _assert_device_commands(tmp_path, workflow, failure=""):
    job = workflow["jobs"]["test"]
    assert job.get("if", "success()") in ("success()", "${{ success() }}")
    assert job["uses"] == "./.github/workflows/linux-test.yml"
    script = job["with"]["script"]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    runner = tmp_path / "runner"
    (runner / "bin").mkdir(parents=True)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "torch_tensorrt_executorch_runtime-fixture.whl").touch()
    helpers = tmp_path / "tests/py/utils/ci_helpers.sh"
    helpers.parent.mkdir(parents=True)
    helpers.write_text("trt_tier_executorch() { :; }\n")
    dispatcher = bin_dir / "dispatch"
    dispatcher.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\nfrom pathlib import Path\n"
        "tool, args = Path(sys.argv[0]).name, sys.argv[1:]\n"
        "with open(os.environ['EVENTS'], 'a') as f: f.write(json.dumps([tool, *args]) + '\\n')\n"
        "if tool == 'python':\n"
        "    if args[:2] == ['-m', 'pip']: pass\n"
        "    elif args[:2] == ['-m', 'venv']:\n"
        "        p = Path(args[2]) / 'bin/python'; p.parent.mkdir(parents=True); p.symlink_to(os.environ['DISPATCH'])\n"
        "    elif args[:4] == ['-u', '-X', 'faulthandler', '-c']: pass\n"
        # The step that builds the module consumer asks python where the two packages put their
        # CMake files. Answer with a directory that exists, so the configure step it feeds is
        # given something plausible rather than an empty string.
        "    elif args[0] == '-c': print(os.environ['RUNNER_TEMP'])\n"
        "    elif args[0].startswith('examples/'):\n"
        "        model = Path(next(a.split('=', 1)[1] for a in args if a.startswith('--model_path=')))\n"
        "        if Path(args[0]).name.startswith('export_'): model.write_text('exported')\n"
        "        else: assert model.read_text() == 'exported'\n"
        "        if os.environ['FAILURE'] and os.environ['FAILURE'] == Path(args[0]).name: sys.exit(17)\n"
        "    else: raise AssertionError(args)\n"
        # cmake configures and builds the module consumer. Neither call needs to do anything here:
        # what this test checks is that the step runs in the right order and propagates a failure.
        "elif tool == 'cmake':\n"
        "    if os.environ['FAILURE'] == 'cmake': sys.exit(17)\n"
        # The build call has to leave the binary the next command runs, or the step fails on a
        # missing file rather than on whatever this case is actually about.
        "    if args[0] == '--build':\n"
        "        exe = Path(args[1]) / 'executorch_module_consumer'\n"
        "        exe.parent.mkdir(parents=True, exist_ok=True)\n"
        "        exe.symlink_to(os.environ['DISPATCH'])\n"
        "elif tool == 'nproc': print('2')\n"
        "elif tool == 'executorch_module_consumer':\n"
        "    if os.environ['FAILURE'] == 'module_consumer': sys.exit(17)\n"
        "elif tool == 'bazel':\n"
        "    if args[0] == 'info': print(os.environ['RUNNER_TEMP'])\n"
        "    elif args[0] == 'query': print(os.environ['RUNNER_TEMP'] + '/executorch/CMakeLists.txt:1:1')\n"
        "    else: assert args[0] in ('build', 'test')\n"
        "elif tool == 'curl': pass\n"
        "elif tool == 'find': print(os.environ['RUNNER_TEMP'] + '/libs')\n"
        "elif tool == 'verify-executorch-reference-runner.sh':\n"
        "    assert all(Path(a).read_text() == 'exported' for a in args)\n"
        "else: raise AssertionError((tool, args))\n"
    )
    dispatcher.chmod(0o755)
    for tool in ("python", "bazel", "curl", "find", "cmake", "nproc"):
        (bin_dir / tool).symlink_to(dispatcher)
    (runner / "bin/bazel").symlink_to(dispatcher)
    for tool in ("mkdir", "chmod", "sort", "head", "dirname"):
        (bin_dir / tool).symlink_to(shutil.which(tool))
    reference = tmp_path / ".github/scripts/verify-executorch-reference-runner.sh"
    reference.parent.mkdir(parents=True)
    reference.symlink_to(dispatcher)
    result = _run(
        script.replace("/opt/torch-tensorrt-builds", str(artifacts)),
        tmp_path,
        {
            "PATH": str(bin_dir),
            "RUNNER_TEMP": str(runner),
            "CU_VERSION": "cu132",
            "EVENTS": str(tmp_path / "events"),
            "DISPATCH": str(dispatcher),
            "FAILURE": failure,
        },
    )
    events = [
        json.loads(line) for line in (tmp_path / "events").read_text().splitlines()
    ]
    setup = next(
        event for event in events if event[:4] == ["python", "-m", "pip", "install"]
    )
    _assert_test_wheel_dependency(setup)
    examples = [
        event[1:]
        for event in events
        if event[0] == "python" and event[1].startswith("examples/")
    ]
    assert (
        examples
        == [
            [
                "examples/torchtrt_executorch_example/export_static_shape.py",
                f"--model_path={runner}/torchtrt-python.pte",
            ],
            [
                "examples/torchtrt_executorch_example/export_kv_cache_decode.py",
                f"--model_path={runner}/torchtrt-kv-cache-decode.pte",
            ],
            [
                "examples/torchtrt_executorch_example/export_coalesced.py",
                f"--model_path={runner}/torchtrt-coalesced.pte",
            ],
            [
                "examples/torchtrt_executorch_example/export_device_resident.py",
                f"--model_path={runner}/torchtrt-device-resident.pte",
            ],
            [
                "examples/executorch_reference_runner/load_model.py",
                f"--model_path={runner}/torchtrt-python.pte",
                "--num_runs=1",
            ],
            [
                "examples/executorch_reference_runner/load_model_device_resident.py",
                f"--model_path={runner}/torchtrt-device-resident.pte",
                "--num_runs=2",
            ],
        ][: 4 if failure == "export_device_resident.py" else 6]
    ), (
        result.stdout + result.stderr
    )
    reference_calls = [e[1:] for e in events if e[0] == reference.name]
    assert reference_calls == (
        []
        if failure == "export_device_resident.py"
        else [
            [
                str(runner / f"torchtrt-{name}.pte")
                for name in ("python", "kv-cache-decode", "coalesced")
            ]
        ]
    )
    assert result.returncode == (17 if failure else 0), result.stdout + result.stderr


@pytest.mark.parametrize(
    "failure", ["", "export_device_resident.py", "load_model_device_resident.py"]
)
def test_device_commands_execute_and_propagate_failure(tmp_path, failure):
    _assert_device_commands(tmp_path, _workflow("executorch-test-linux.yml"), failure)


@pytest.mark.parametrize(
    "command", [_DEVICE_EXPORT, _DEVICE_RUN], ids=["export", "run"]
)
@pytest.mark.parametrize("mutation", ["commented", "disabled", "removed"])
def test_device_command_removal_is_detected(tmp_path, command, mutation):
    workflow = _workflow("executorch-test-linux.yml")
    script = workflow["jobs"]["test"]["with"]["script"]
    assert script.count(command) == 1
    replacement = {
        "commented": "".join("# " + line for line in command.splitlines(keepends=True)),
        "disabled": "if false; then\n" + command + "fi\n",
        "removed": "",
    }[mutation]
    workflow["jobs"]["test"]["with"]["script"] = script.replace(command, replacement)
    with pytest.raises(AssertionError):
        _assert_device_commands(tmp_path, workflow)


@pytest.mark.parametrize("entrypoint", ["manifest", "workflow"])
def test_missing_test_wheel_dependency_is_detected(tmp_path, monkeypatch, entrypoint):
    if entrypoint == "manifest":
        monkeypatch.syspath_prepend(str(ROOT))
        from tests.ci import runner

        setup = runner._setup_commands
        monkeypatch.setattr(
            runner,
            "_setup_commands",
            lambda step: [
                ([arg for arg in argv if not arg.startswith("wheel")], cwd)
                for argv, cwd in setup(step)
            ],
        )
        with pytest.raises(AssertionError, match="ExecuTorch tests require wheel"):
            test_manifest_suite_provisions_wheel(monkeypatch, "cu132")
    else:
        workflow = _workflow("executorch-test-linux.yml")
        script = workflow["jobs"]["test"]["with"]["script"]
        assert script.count('"wheel>=0.40"') == 1
        workflow["jobs"]["test"]["with"]["script"] = script.replace('"wheel>=0.40"', "")
        with pytest.raises(AssertionError, match="ExecuTorch tests require wheel"):
            _assert_device_commands(tmp_path, workflow)


def test_disabled_device_job_is_detected(tmp_path):
    workflow = _workflow("executorch-test-linux.yml")
    workflow["jobs"]["test"]["if"] = "${{ false }}"
    with pytest.raises(AssertionError):
        _assert_device_commands(tmp_path, workflow)


@pytest.mark.unit
def test_the_delegate_lane_narrows_the_matrix_to_cuda_13_rows() -> None:
    """The lane passes --executorch-runtime, and the filter must act on it.

    Nothing else asserted this, so deleting the flag from the workflow, or the branch it gates
    in the filter, left every guard green while the lane silently tested rows whose channel
    carries no ExecuTorch.
    """
    workflow = (ROOT / ".github/workflows/executorch-test-linux.yml").read_text(
        encoding="utf-8"
    )
    # Not a text search: the flag has to reach the filter. Leaving it only inside a comment, which is
    # what commenting the whole invocation out does, passed a search of the file.
    document = yaml.safe_load(workflow)
    invocations = [
        step["run"]
        for job in document["jobs"].values()
        for step in job.get("steps", [])
        if isinstance(step, dict) and "filter-matrix.py" in (step.get("run") or "")
    ]
    assert invocations, "no step runs the matrix filter"
    live = [
        line
        for run in invocations
        for line in run.splitlines()
        if "--executorch-runtime" in line and not line.lstrip().startswith("#")
    ]
    assert live, f"the flag reaches no live command line: {invocations}"

    script = ROOT / ".github/scripts/filter-matrix.py"
    rows = [
        {"desired_cuda": cuda, "python_version": "3.10", "gpu_arch_type": "cuda"}
        for cuda in ("cu126", "cu130", "cu134")
    ]
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--executorch-runtime",
            "--use-rtx",
            "false",
            "--limit-pr-builds",
            "false",
            "--matrix",
            json.dumps({"include": rows}),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    kept = {row["desired_cuda"] for row in json.loads(result.stdout)["include"]}
    assert kept == {"cu130", "cu134"}, kept


@pytest.mark.unit
def test_the_removed_entry_points_still_exist_as_shims() -> None:
    """activate() and get_runtime() were public, so removing them outright breaks callers."""
    package = ROOT / "py/torch-tensorrt-executorch-runtime"
    source = (package / "torch_tensorrt_executorch_runtime/__init__.py").read_text(
        encoding="utf-8"
    )
    assert "def activate(" in source, source
    assert "def get_runtime(" in source, source
    for name in ("activate", "get_runtime", "register"):
        assert f'"{name}"' in source.split("__all__")[-1], name


@pytest.mark.parametrize(
    "raised",
    [
        RuntimeError("no bazel here"),
        OSError("toolchain gone"),
        ValueError("bad config"),
    ],
)
@pytest.mark.unit
def test_a_failed_native_build_cannot_report_success(raised) -> None:
    """During an editable install setuptools routes a customized build_py through its own
    _safely_run, which catches Exception and downgrades it to a warning pip hides, so a failed
    delegate build would leave pip printing that it installed successfully. SystemExit is not an
    Exception, so converting to it is what escapes.

    Driven rather than read: the wrapper is applied to a build that raises, and the result is passed
    through a stand-in for setuptools' catch. A version that re-raised RuntimeError unchanged, which
    is what a missing bazel raises, passed a source check while still being swallowed here.
    """
    source = (ROOT / "py/torch-tensorrt-executorch-runtime/setup.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    build_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BazelBuild"
    )
    run = next(
        node
        for node in build_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )

    class Fake:
        def _build(self):
            raise raised

    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=[run], type_ignores=[]), "<run>", "exec"), namespace)
    # setuptools' own shape: Exception becomes a warning, anything else propagates.
    try:
        namespace["run"](Fake())
    except Exception as error:  # noqa: BLE001
        pytest.fail(f"a {type(raised).__name__} would be swallowed as {error!r}")
    except SystemExit as exit_error:
        assert "ExecuTorch delegate build failed" in str(exit_error), exit_error
    else:
        pytest.fail("the wrapper let a failing build return normally")


@pytest.mark.unit
def test_the_kept_entry_points_actually_warn_and_forward(monkeypatch) -> None:
    """activate() and get_runtime() were public, so they stay as deprecated shims.

    Checking the source for "def activate(" proves only that the text is present: the bodies can be
    emptied, or their warnings removed, and the check still passes. So import the package and call
    them. Registration is skipped through the package's own escape hatch, because the native library
    is not built here, and each shim's own call to register is replaced so the forwarding is visible.
    """
    monkeypatch.setenv("TORCH_TENSORRT_SKIP_DELEGATE_REGISTRATION", "1")
    monkeypatch.syspath_prepend(str(ROOT / "py/torch-tensorrt-executorch-runtime"))
    for name in [
        n for n in sys.modules if n.startswith("torch_tensorrt_executorch_runtime")
    ]:
        monkeypatch.delitem(sys.modules, name)
    delegate = importlib.import_module("torch_tensorrt_executorch_runtime")

    registered = []
    monkeypatch.setattr(delegate, "register", lambda: registered.append("register"))
    for shim in ("activate", "get_runtime"):
        registered.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # The proof that forwarding happened is the registration below, not an exception. Where
            # ExecuTorch is absent the forwarding import raises, and where it is installed the call
            # succeeds, so demanding the raise would make this pass only in an environment without
            # ExecuTorch and fail in one with it. Either outcome is accepted; a shim that registers,
            # warns and forwards nowhere is still caught, because it would not register.
            try:
                getattr(delegate, shim)()
            except ImportError:
                pass
        assert registered == ["register"], f"{shim} did not register: {registered}"
        assert any(
            issubclass(w.category, DeprecationWarning) for w in caught
        ), f"{shim} raised no DeprecationWarning: {[w.category for w in caught]}"


@pytest.mark.unit
@pytest.mark.parametrize("green_returns_the_wrong_numbers", [False, True])
def test_the_coalesced_program_is_run_on_a_caller_stream(
    tmp_path, green_returns_the_wrong_numbers
) -> None:
    """Nothing exercised the caller stream, which is the path the delegate is built around.

    The delegate takes the stream from the caller and a green context confines it to a slice of the
    machine. Both were reachable only by hand: every automated run used the default stream, so a
    delegate that stopped honouring the caller's stream would have kept passing.

    Searching the script for the flag is not enough. Wrapping the green-context run in `if false`
    leaves every string in place, so a text check stays green while CI stops exercising the caller
    stream. So the script's own coalesced section is executed here, against a stub runner, and the
    stub has to be handed the flag. The wrong-numbers case proves the output is still checked and
    not only the exit status, since that check is the script's own function running for real.
    """
    lines = (
        (ROOT / ".github/scripts/verify-executorch-reference-runner.sh")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    # The script's own output check and its own coalesced section, taken whole rather than
    # paraphrased. The section is the last block in the file.
    check = lines.index("assert_runner_output() {")
    harness = "\n".join(
        lines[check : lines.index("}", check) + 1]
        + lines[lines.index('if [[ -n "${coalesced_model_path}" ]]; then') :]
    )
    calls = tmp_path / "calls"
    stub = tmp_path / "runner"
    stub.write_text(
        "#!/bin/sh\n"
        'echo "$@" >> "$CALLS"\n'
        'echo "planned buffer[0] = 16384 bytes on device_type 1"\n'
        'echo "output[0] shape=[64,64]"\n'
        'if [ -n "$WRONG_GREEN" ] && echo "$@" | grep -q green_context; then\n'
        '  echo "first 3 values: 9.0000 9.0000 9.0000"\n'
        "else\n"
        '  echo "first 3 values: 0.5000 0.5000 0.5000"\n'
        "fi\n"
    )
    stub.chmod(0o755)
    model = tmp_path / "coalesced.pte"
    model.write_bytes(b"pte")
    (tmp_path / "coalesced.pte.expected").write_text("[64,64]\n0.5000\n")
    work = tmp_path / "work"
    work.mkdir()
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", harness],
        env={
            **os.environ,
            "CALLS": str(calls),
            "WRONG_GREEN": "1" if green_returns_the_wrong_numbers else "",
            "runner_path": str(stub),
            "verify_root": str(work),
            "coalesced_model_path": str(model),
        },
        capture_output=True,
        text=True,
    )
    invocations = (
        calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []
    )
    green = [call for call in invocations if "--green_context_sms=" in call]
    assert green, f"the runner was never given a green context: {invocations}"
    assert (
        work / "coalesced_green_context.log"
    ).exists(), f"the green-context run produced no log: {list(work.iterdir())}"
    if green_returns_the_wrong_numbers:
        assert result.returncode != 0, result.stdout + result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
