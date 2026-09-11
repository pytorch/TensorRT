"""CPU-only command-boundary tests for the shared companion workflows."""

import ast
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from wheel.wheelfile import WheelFile

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
    (site / "pip.py").write_text(
        "import importlib.metadata as m, json, os, sys, zipfile\n"
        "from pathlib import Path\n"
        "from packaging.requirements import Requirement\n"
        "site = Path(__file__).parent\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['EVENTS'], 'a') as f: f.write(json.dumps(args) + '\\n')\n"
        "if args[0] == 'install':\n"
        "    for arg in args[1:]:\n"
        "        if arg.endswith('.whl'):\n"
        "            with zipfile.ZipFile(arg) as w: w.extractall(site)\n"
        "        elif arg.startswith('tensorrt'):\n"
        "            r = Requirement(arg)\n"
        "            assert '11.2.1.2' in r.specifier\n"
        "            for name in (r.name, 'tensorrt-cu13', 'tensorrt-cu13-bindings', 'tensorrt-cu13-libs'):\n"
        "                info = site / (name.replace('-', '_') + '-11.2.1.2.dist-info')\n"
        "                info.mkdir(exist_ok=True)\n"
        "                (info / 'METADATA').write_text(f'Name: {name}\\nVersion: 11.2.1.2\\n')\n"
        "elif args[0] == 'wheel':\n"
        "    installed = {d.metadata['Name']: d.version for d in m.distributions(path=[site])}\n"
        "    if 'tensorrt-cu13' not in installed: raise m.PackageNotFoundError('tensorrt-cu13')\n"
        "    assert installed['tensorrt-cu13'] == '11.2.1.2'\n"
        "    assert installed['tensorrt-cu13-libs'] == '11.2.1.2'\n"
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
    shutil.copy2(ROOT / ".github/scripts/filter-executorch-cuda-arches.py", scripts)
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
