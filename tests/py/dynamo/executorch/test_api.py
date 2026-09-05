import ast
import importlib
import importlib.metadata
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import types
import zipfile
from pathlib import Path

import pytest
import torch
from torch._library.fake_class_registry import FakeScriptObject
from torch._subclasses.fake_tensor import FakeTensor
from torch.export.graph_signature import InputKind
from torch_tensorrt.dynamo._exporter import _resolve_lifted_custom_obj, lift

# CMake command names are case-insensitive, so IF(FALSE) and If(FALSE) open the same block a
# case-sensitive pattern misses. Every block command counts, not just if(): wrapping the guard in
# while(FALSE) or in an uncalled function() hides it just as completely. Measured with real cmake
# builds: all four spellings produced a delegate needing libcudart.so.12, which is what the guard
# exists to stop, with the wiring test green.
_CMAKE_BLOCK_OPEN = r"(?:if|while|foreach|function|macro|block)\s*\("
_CMAKE_BLOCK_CLOSE = r"end(?:if|while|foreach|function|macro|block)\s*\("


@pytest.mark.unit
def test_lazy_import_error_when_executorch_missing(monkeypatch):
    import torch_tensorrt

    original_module = sys.modules.pop("torch_tensorrt.executorch", None)
    original_attribute = getattr(torch_tensorrt, "executorch", None)
    if hasattr(torch_tensorrt, "executorch"):
        delattr(torch_tensorrt, "executorch")
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, package=None):
        if name == "executorch.exir":
            return None
        return original_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    module = importlib.import_module("torch_tensorrt.executorch")

    with pytest.raises(ImportError, match=r"torch_tensorrt\[executorch\]"):
        _ = module.TensorRTBackend

    sys.modules.pop("torch_tensorrt.executorch", None)
    if original_module is not None:
        sys.modules["torch_tensorrt.executorch"] = original_module
    if original_attribute is not None:
        torch_tensorrt.executorch = original_attribute
    elif hasattr(torch_tensorrt, "executorch"):
        delattr(torch_tensorrt, "executorch")


@pytest.mark.unit
def test_save_executorch_error_when_executorch_missing(monkeypatch, tmp_path):
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, package=None):
        if name == "executorch.exir":
            return None
        return original_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    from torch_tensorrt._compile import save

    with pytest.raises(ImportError, match=r"torch_tensorrt\[executorch\]"):
        save(
            torch.nn.Linear(1, 1),
            str(tmp_path / "model.pte"),
            output_format="executorch",
        )


@pytest.mark.unit
def test_load_does_not_accept_an_executorch_format():
    """Loading a .pte is ExecuTorch's API, not this one.

    ``load`` used to take ``format="executorch"`` and forward to a wrapper shipped in the delegate
    wheel. That wrapper duplicated ExecuTorch's own ``Runtime``/``Program``, down to the line that
    holds the file buffer alive, so it was removed along with the parameter. A consumer imports the
    delegate package for its registration side effect and then uses ``executorch.runtime`` directly.
    Asserted rather than assumed, because silently accepting and ignoring the keyword would leave
    callers thinking the old path still worked.
    """
    from torch_tensorrt import _compile

    with pytest.raises(TypeError, match="format"):
        _compile.load("model.pte", format="executorch")

    # None was the documented default, so a wrapper forwarding an optional format it received has
    # to keep working. Rejecting it would break callers that never asked for ExecuTorch at all.
    #
    # What the error IS depends on the build, so this asserts only what matters: that it is not the
    # guard's TypeError. A build with the TorchScript frontend raises ValueError from load()'s own
    # "not a valid TorchScript module or ExportedProgram" path, while a build without it reaches the
    # zipfile open and raises FileNotFoundError. Pinning either one makes this test pass on one build
    # and fail on the other.
    try:
        _compile.load("does-not-exist.pt2", format=None)
    except TypeError as error:  # pragma: no cover - the guard must not fire here
        raise AssertionError(
            f"load() rejected format=None, the documented default: {error}"
        ) from error
    except Exception:
        pass  # Reached the real loader, which is the point: the guard let None through.
    else:
        raise AssertionError("load() of a missing file unexpectedly succeeded")


@pytest.mark.unit
def test_public_api_symbols_present():
    module = importlib.import_module("torch_tensorrt.executorch")
    assert "get_edge_compile_config" in module.__all__
    assert "TensorRTPartitioner" in module.__all__
    assert "TensorRTBackend" in module.__all__
    assert "export" in module.__all__
    assert "Program" not in module.__all__
    assert "load" not in module.__all__
    assert "to_executorch" not in module.__all__


_REPO_ROOT = Path(__file__).resolve().parents[4]
_SETUP_PY = _REPO_ROOT / "setup.py"
_RUNTIME_SETUP_PY = _REPO_ROOT / "py/torch-tensorrt-executorch-runtime/setup.py"

# The RUNPATH the build declares, TORCH_TENSORRT_DELEGATE_RUNPATH in native/CMakeLists.txt joined
# with ':'. Production hands this to the guard as its fourth argument, which selects the
# exact-whole-RUNPATH branch.
_GUARD_GOOD_RUNPATH = (
    "$ORIGIN:$ORIGIN/../../executorch/lib:$ORIGIN/../../tensorrt_libs"
    ":$ORIGIN/../../nvidia/cu13/lib:$ORIGIN/../../nvidia/cuda_runtime/lib"
)
# Cases kept on the 3-argument invocation so the fallback branch (the elif in the guard that
# spot-checks for '$ORIGIN/../../executorch/lib' when no expected RUNPATH is passed) and the
# absolute-entry check below it stay covered. Everything else runs with the fourth argument the
# way production does, exercising the exact-whole-RUNPATH branch.
_THREE_ARG_CASES = {
    "wrong_depth_runpath",
    "runpath_missing_executorch",
    "absolute_runpath",
    "runpath_fallback_reaches_executorch",
}


@pytest.mark.unit
def test_the_cmake_package_defines_a_linkable_target():
    """Configure the package with real CMake and require the imported target to exist.

    The sibling test searches the config for strings, and a string search cannot tell a working
    package from a broken one: inserting ``return()`` after ``cmake_minimum_required`` makes the
    config define nothing at all, and every string assertion still passes because the text it reads
    is still there. find_package does not even error under REQUIRED in that case, so the failure
    surfaces later as an unlinkable target in a consumer's build.

    This runs cmake against a prefix laid out the way the wheel installs, with a stub library, and
    asks for IMPORTED_LOCATION back. That is the property consumers depend on.
    """
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake is not installed, so the package cannot be configured here")

    package_dir = _REPO_ROOT / "py/torch-tensorrt-executorch-runtime"
    config = package_dir / "cmake/torchtrt_executorch-config.cmake"
    library = "libexecutorch_backend_tensorrt.so"

    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        # Exactly the wheel's layout: the config under lib/cmake/<name>/ beside the library in lib/.
        cmake_dir = root / "prefix/lib/cmake/torchtrt_executorch"
        cmake_dir.mkdir(parents=True)
        shutil.copy2(config, cmake_dir / config.name)
        (cmake_dir / "torchtrt_executorch-config-version.cmake").write_text(
            'set(PACKAGE_VERSION "2.15.0")\nset(PACKAGE_VERSION_COMPATIBLE TRUE)\n',
            encoding="utf-8",
        )
        # A stub is enough: configuring never opens the file, and building a real one here would
        # need the whole ExecuTorch toolchain.
        (root / "prefix/lib" / library).write_bytes(b"\x7fELF stub")

        app = root / "app"
        app.mkdir()
        (app / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.28)\n"
            "project(probe LANGUAGES NONE)\n"
            "find_package(torchtrt_executorch REQUIRED)\n"
            "if(NOT TARGET torchtrt::executorch_backend)\n"
            '  message(FATAL_ERROR "NO_TARGET")\n'
            "endif()\n"
            "get_target_property(location torchtrt::executorch_backend IMPORTED_LOCATION)\n"
            'message(STATUS "LOCATION=${location}")\n'
            "get_target_property(options torchtrt::executorch_backend INTERFACE_LINK_OPTIONS)\n"
            'message(STATUS "OPTIONS=${options}")\n',
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                cmake,
                "-S",
                str(app),
                "-B",
                str(app / "build"),
                f"-DCMAKE_PREFIX_PATH={root / 'prefix'}",
                "-DCMAKE_SYSTEM_NAME=Linux",
            ],
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        assert (
            result.returncode == 0
        ), f"find_package(torchtrt_executorch) failed:\n{output}"
        assert "NO_TARGET" not in output, (
            "find_package succeeded but defined no torchtrt::executorch_backend target, so a "
            f"consumer has nothing to link:\n{output}"
        )
        # The target has to point AT the library, not at the directory holding it. A walk that
        # stops at the first lib/ it meets yields the directory and links nothing.
        assert (
            f"LOCATION={root / 'prefix/lib' / library}" in output
        ), f"IMPORTED_LOCATION is not the delegate library:\n{output}"
        assert (
            "no-as-needed" in output
        ), f"the target carries no --no-as-needed, so the backend can silently not register:\n{output}"


@pytest.mark.unit
def test_the_delegate_follows_the_main_wheels_cuda_versions():
    """The slim wheel must build every CUDA version the main wheel builds, per architecture.

    Both read the same matrix filter, so this holds by construction rather than by a list kept in
    step by hand. The test exists because the two could drift: an ExecuTorch-only filter, or a
    hardcoded CUDA version in the delegate's own build, would silently drop a row. CUDA 12.6 is the
    live case, published for x86_64 only while the Arm rows stay on CUDA 13.
    """
    matrix = {
        "include": [
            {
                "python_version": "3.12",
                "desired_cuda": cuda,
                "gpu_arch_version": cuda[2:],
                "gpu_arch_type": arch_type,
                "validation_runner": "runner",
                "container_image": "image",
                "package_type": "wheel",
                "build_name": f"b_{cuda}_{arch}",
                "channel": "nightly",
                "upload_to_base_dir": "d",
                "stable_version": "s",
                "use-rtx": "false",
                "os": os_name,
                "arch": arch,
            }
            for cuda in ("cu126", "cu130", "cu132")
            for os_name, arch, arch_type in (
                ("linux", "x86_64", "cuda"),
                ("linux-aarch64", "aarch64", "cuda-aarch64"),
            )
        ]
    }
    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / ".github/scripts/filter-matrix.py"),
            "--matrix",
            json.dumps(matrix),
            "--use-rtx",
            "false",
            "--limit-pr-builds",
            "false",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = json.loads(result.stdout.strip().splitlines()[-1])["include"]
    by_arch = {}
    for row in rows:
        by_arch.setdefault(row["os"], set()).add(row["desired_cuda"])

    assert "cu126" in by_arch.get("linux", set()), (
        "the x86_64 rows dropped CUDA 12.6, which the main wheel publishes, so the delegate would "
        f"not be built for it. x86_64 rows carry {sorted(by_arch.get('linux', ()))}"
    )
    for os_name in ("linux", "linux-aarch64"):
        assert {"cu130", "cu132"} <= by_arch.get(
            os_name, set()
        ), f"{os_name} lost a CUDA 13 row: {sorted(by_arch.get(os_name, ()))}"

    # Deliberately silent on which CUDA versions the Arm rows carry. The delegate follows the
    # main wheel rather than a list of its own, so when Arm gains CUDA 12.6 upstream this test
    # keeps passing instead of needing an edit in that same change.


def test_the_export_is_skipped_only_when_there_is_no_gpu():
    """The export needs a GPU, but skipping it must not quietly drop the check everywhere.

    The aarch64 builders are CPU-only Graviton instances, so the export and the reference runner
    cannot run there. They are skipped on a real capability test rather than on the architecture, so
    the x86_64 rows, which do have a GPU, still run them. A skip keyed on anything else would remove
    the only place the delegate is executed in CI.
    """
    import yaml

    script = yaml.safe_load(
        (_REPO_ROOT / ".github/workflows/executorch-build-linux.yml").read_text(
            encoding="utf-8"
        )
    )["jobs"]["build"]["with"]["script"]

    assert (
        "export_static_shape.py" in script
    ), "the build no longer runs the export at all"
    assert (
        "verify-executorch-reference-runner.sh" in script
    ), "the build no longer runs the reference runner"
    # Keyed on the device being usable, not on the architecture or the runner label: those would
    # skip on a GPU runner too if the label ever changed.
    assert "torch.cuda.is_available()" in script, (
        "the export is not guarded by a GPU capability test, so it either fails on the CPU-only "
        "builders or is skipped by something that does not measure what it depends on"
    )
    for wrong in ("uname -m", "aarch64", "arm64"):
        assert (
            f"if {wrong}" not in script and f'"{wrong}" =' not in script
        ), f"the export skip is keyed on {wrong!r} rather than on whether a GPU is present"


def test_the_build_script_never_imports_torch_tensorrt():
    """The delegate builders have no GPU, and importing torch_tensorrt needs one.

    The lowering passes call torch.cuda.get_device_capability() at import time while deciding
    whether they are running on Tegra, so any import of the package in the build script fails with
    "Found no NVIDIA driver on your system". The x86_64 builder happens to have a GPU, so this only shows up
    on the aarch64 lane, and it cost a full build cycle twice: once when it was first written and
    again when a revert restored the original line and nothing noticed.
    """
    import yaml

    script = yaml.safe_load(
        (_REPO_ROOT / ".github/workflows/executorch-build-linux.yml").read_text(
            encoding="utf-8"
        )
    )["jobs"]["build"]["with"]["script"]

    for forbidden in (
        "import torch_tensorrt\n",
        "from torch_tensorrt",
        "import torch_tensorrt ",
    ):
        assert forbidden not in script, (
            f"the build script contains {forbidden.strip()!r}, which pulls in "
            "torch.cuda.get_device_capability() at import time and fails on a builder with no GPU"
        )
    # torch_tensorrt_executorch_runtime is a different distribution and is fine: it dlopens the
    # delegate and never imports the compiler package
    bare = [
        line
        for line in script.splitlines()
        if "torch_tensorrt" in line
        and "torch_tensorrt_executorch_runtime" not in line
        and "torch-tensorrt" not in line
        and line.strip().startswith(("python", "import", "from"))
    ]
    assert not bare, f"the build script still imports torch_tensorrt: {bare}"


def test_the_delegate_is_built_for_every_architecture_the_main_wheel_ships():
    """The delegate must cover the same architectures as the torch-tensorrt wheel it pairs with.

    That wheel publishes manylinux_2_28 for x86_64 and aarch64 on every CUDA channel, so an aarch64
    user who installs the extra would otherwise find no delegate at all. Three things have to hold
    and each one failed in a real run before it was asserted here.
    """
    import yaml

    workflows = _REPO_ROOT / ".github/workflows"
    build = yaml.safe_load(
        (workflows / "executorch-build-linux.yml").read_text(encoding="utf-8")
    )

    # 1. The build has to accept an architecture and pass it on. The inner workflow defaults to
    # x86_64, so a caller that omits it silently builds the wrong row rather than failing.
    inputs = build[True]["workflow_call"]["inputs"]
    assert "architecture" in inputs, "the delegate build takes no architecture input"
    assert (
        build["jobs"]["build"]["with"].get("architecture")
        == "${{ inputs.architecture }}"
    ), "the architecture is accepted but not forwarded, so the inner workflow uses its default"

    callers = {}
    for path in sorted(workflows.glob("ci-*.yml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        jobs = data.get("jobs") or {}
        job = jobs.get("executorch-runtime-build")
        if job is None:
            continue
        callers[path.name] = (data, jobs, job)

    assert len(callers) >= 2, (
        f"only {len(callers)} workflow builds the delegate, so it cannot cover both architectures "
        "the main wheel ships"
    )

    expected_os = {"x86_64": "linux", "aarch64": "linux-aarch64"}
    for name, (data, jobs, job) in callers.items():
        arch = (job.get("with") or {}).get("architecture", "x86_64")
        matrix_os = data["jobs"]["generate-matrix"]["with"]["os"]
        assert matrix_os == expected_os[arch], (
            f"{name} builds the delegate for {arch} from a {matrix_os} matrix, so the rows it "
            "iterates do not match the architecture it compiles for"
        )

        # 2. Ordered after whichever job uploads the wheel it downloads. Measured on a real run:
        # the delegate lane started at 19:25 and failed on a missing artifact at 19:32, while the
        # wheel it wanted finished uploading at 19:44.
        needs = job.get("needs") or []
        needs = [needs] if isinstance(needs, str) else needs
        producers = [
            n for n in needs if n not in {"decide", "generate-matrix", "filter-matrix"}
        ]
        assert producers, (
            f"{name} builds the delegate without waiting for any wheel-producing job, so it races "
            "the upload and fails on a missing artifact"
        )

    # 3. The aarch64 lane stays out of its workflow's gate. A gate fails the whole workflow on any
    # failure among its needs, and ci-sbsa.yml gates the main wheel for every architecture-related
    # pull request, so a delegate failure there blocks work that has nothing to do with the
    # delegate. This is not a rule about gates in general: ci-linux-x86_64.yml has gated the
    # delegate since before this change, and that workflow exists to gate the delegate's own lanes.
    sbsa = callers.get("ci-sbsa.yml")
    if sbsa is not None:
        _, sbsa_jobs, _ = sbsa
        gate_needs = sbsa_jobs.get("gate", {}).get("needs") or []
        gate_needs = [gate_needs] if isinstance(gate_needs, str) else gate_needs
        assert "executorch-runtime-build" not in gate_needs, (
            "ci-sbsa.yml gates the main wheel's aarch64 channels, so a delegate failure in its "
            "gate blocks pull requests unrelated to the delegate"
        )


def test_the_guard_is_given_the_platform_it_must_compare_against():
    """The symbol version ceiling needs the manylinux tag, and an unpassed argument is silent.

    The guard skips the ceiling entirely when the tag is empty, so a build that forgets to pass it
    loses the check without failing. The tag also has to differ per architecture: the aarch64 row
    builds in manylinux_2_39 rather than 2_28, because TensorRT needs a newer glibc there, and
    comparing that row against 2_28 rejects it for requiring what its own platform guarantees.
    """
    cmake = (
        _REPO_ROOT / "py/torch-tensorrt-executorch-runtime/native/CMakeLists.txt"
    ).read_text(encoding="utf-8")
    guard = (
        _REPO_ROOT
        / "py/torch-tensorrt-executorch-runtime/native/check_imports_executorch_runtime.sh"
    ).read_text(encoding="utf-8")

    assert "TORCH_TENSORRT_MANYLINUX_TAG" in cmake, (
        "the build never computes a manylinux tag, so the guard receives nothing and skips the "
        "symbol version ceiling without saying so"
    )
    invocation = cmake[cmake.index("check_imports_executorch_runtime.sh") :]
    invocation = invocation[: invocation.index("VERBATIM")]
    assert (
        "${TORCH_TENSORRT_MANYLINUX_TAG}" in invocation
    ), "the tag is computed but not passed to the guard, so the ceiling is skipped"
    assert (
        'glibc_floor="${5:-}"' in guard
    ), "the guard does not read a fifth argument, so the tag the build passes is ignored"
    # Per architecture, not one constant. Asserted on the set() calls rather than on the text,
    # because the tag names also appear in the comment explaining why they differ, so a substring
    # search stayed green when both branches were collapsed to the same value.
    assigned = set(
        re.findall(r"set\(TORCH_TENSORRT_MANYLINUX_TAG \"([^\"]+)\"\)", cmake)
    )
    assert assigned == {"manylinux_2_28", "manylinux_2_39"}, (
        "the build assigns "
        f"{sorted(assigned)} as its manylinux tag, but the two architectures ship under different "
        "platforms and using one for both rejects the aarch64 row for requiring exactly what its "
        "own builder image provides"
    )
    for tag in ("_2_28", "_2_39"):
        assert tag in guard, f"the guard has no floor entry for manylinux{tag}"


def test_the_wheel_ships_a_cmake_package_for_cpp_consumers():
    """The delegate has to be linkable from C++, not just importable from Python.

    ExecuTorch ships each of its backends as a prebuilt shared library plus a CMake package, so a
    C++ app links ``executorch::backend_cuda`` and the backend registers itself. This wheel is an
    out-of-tree backend and needs the same two pieces, or its shared library is reachable only by
    building this repository from source.

    Three things are asserted because each fails differently. Without the config file there is no
    target to link. With the library outside ``lib/`` the config cannot find it, and the Python
    loader and the C++ consumer would disagree about where it lives. Without ``--no-as-needed`` the
    link succeeds and the backend silently never registers, which is the worst of the three because
    it fails at run time with an unregistered backend rather than at build time.
    """
    package_dir = _REPO_ROOT / "py/torch-tensorrt-executorch-runtime"
    config = package_dir / "cmake/torchtrt_executorch-config.cmake"
    assert (
        config.is_file()
    ), "no CMake package config, so a C++ app cannot link the delegate out of the wheel"
    config_text = config.read_text(encoding="utf-8")

    assert (
        "SHARED IMPORTED" in config_text
    ), "the config does not define an imported shared library target"
    assert "/lib/libexecutorch_backend_tensorrt.so" in config_text, (
        "the config does not look for the delegate under lib/, where the wheel installs it and "
        "where ExecuTorch keeps its own backends"
    )
    # Asserted against the LINKER: line, not the whole file, because the comment above it also says
    # --no-as-needed. A substring test over the file would pass on a config that explains the flag
    # and then does not pass it.
    link_options = [
        line
        for line in config_text.splitlines()
        if "LINKER:" in line and "no-as-needed" in line
    ]
    assert link_options, (
        "the config does not force the delegate onto the link line; nothing references a symbol "
        "it defines, so the linker would drop it and the backend would never register"
    )
    assert "push-state" in link_options[0] and "pop-state" in link_options[0], (
        "--no-as-needed is not bracketed with push-state/pop-state, so it leaks into the rest of "
        "the consumer's link line"
    )

    # setup.py has to actually ship both files, and put the library where the config looks.
    setup_text = (package_dir / "setup.py").read_text(encoding="utf-8")
    for fragment in (
        "lib/cmake/torchtrt_executorch/*.cmake",
        "lib/{DELEGATE_LIBRARY}",
    ):
        assert (
            fragment in setup_text
        ), f"setup.py does not package {fragment}, so the wheel would omit it"
    # The directory the build WRITES to has to be the one package_data names. Those are two
    # separate statements in setup.py, and when they disagree the build writes the CMake package
    # somewhere the wheel never collects, so the wheel ships no package at all and every C++
    # consumer fails at find_package. Asserting the path components rather than a joined string,
    # since setup.py builds it with pathlib.
    assert re.search(
        r'cmake_dir\s*=\s*package_dir\s*/\s*"lib"\s*/\s*"cmake"\s*/\s*"torchtrt_executorch"',
        setup_text,
    ), (
        "setup.py writes the CMake package somewhere other than lib/cmake/torchtrt_executorch, so "
        "it no longer agrees with the path package_data collects and the wheel would ship no "
        "CMake package"
    )
    assert "torchtrt_executorch-config-version.cmake" in setup_text, (
        "setup.py writes no version file, so find_package(torchtrt_executorch 2.15) would match "
        "any version at all"
    )
    # The package has to REFUSE a prefix with no library, rather than export a target pointing
    # nowhere. Without this gate find_package reports success and IMPORTED_LOCATION comes out empty,
    # which was measured: "FOUND_ANYWAY, IMPORTED_LOCATION=[]". The consumer then fails at link time
    # with a message that names neither this package nor the missing file.
    assert "find_package_handle_standard_args(" in config_text, (
        "the config never calls find_package_handle_standard_args, so a prefix with no delegate "
        "still reports success"
    )
    assert "REQUIRED_VARS TORCHTRT_EXECUTORCH_BACKEND_LIBRARY" in config_text, (
        "the library is not listed in REQUIRED_VARS, so find_package succeeds when the delegate is "
        "absent and exports a target with an empty IMPORTED_LOCATION"
    )
    assert "if(NOT torchtrt_executorch_FOUND)" in config_text, (
        "the config does not return early when the library is missing, so it goes on to define an "
        "imported target from an empty path"
    )
    # The version file has to consult the upper end of a range and the major, not just ask whether
    # the installed version is at least the requested one. A lone VERSION_LESS accepts two requests
    # it must refuse: 2.14...<2.15 is handed 2.15, and a request for 1.0 is satisfied by 2.15, so a
    # consumer written against a different major links this delegate anyway. Both were measured
    # against write_basic_package_version_file(COMPATIBILITY SameMajorVersion), which refuses them.
    for required, why in (
        ("PACKAGE_FIND_VERSION_RANGE", "the upper end of a version range is ignored"),
        ("PACKAGE_FIND_VERSION_MAX", "a range's maximum is never compared"),
        ("PACKAGE_FIND_VERSION_MAJOR", "a request from another major is accepted"),
    ):
        assert required in setup_text, (
            f"the generated version file does not mention {required}, so {why} and find_package "
            "matches versions it was told not to"
        )


@pytest.mark.unit
def test_the_runtime_package_ships_no_runtime_api():
    """The delegate wheel registers a backend and nothing else.

    It used to carry a ``runtime.py`` wrapping ExecuTorch's ``Runtime``/``Program``, which duplicated
    what ExecuTorch already exports and put a second inference API in a wheel whose only job is
    registration. Both files are asserted so a reintroduction anywhere is caught: neither the old
    location under torch_tensorrt nor one inside the delegate package.
    """
    assert not (_REPO_ROOT / "py/torch_tensorrt/executorch/runtime.py").exists()
    assert not (
        _REPO_ROOT
        / "py/torch-tensorrt-executorch-runtime"
        / "torch_tensorrt_executorch_runtime/runtime.py"
    ).exists()

    # And the package exports only the registration surface.
    delegate_init = (
        _REPO_ROOT
        / "py/torch-tensorrt-executorch-runtime"
        / "torch_tensorrt_executorch_runtime/__init__.py"
    ).read_text(encoding="utf-8")
    exported = ast.literal_eval(
        re.search(r"^__all__\s*=\s*(\[[^\]]*\])", delegate_init, re.MULTILINE).group(1)
    )
    assert set(exported) == {
        "BACKEND_NAME",
        "DelegateCompatibilityError",
        "register",
    }, (
        "the delegate package exports something beyond its registration surface: "
        f"{sorted(exported)}"
    )


@pytest.mark.unit
def test_runtime_extension_has_dependency_wheel_rpaths():
    """The search path that actually ships is the patchelf literal, so assert on that one.

    The list is declared once and consumed twice: as INSTALL_RPATH for the linker, and as the
    value handed to ``patchelf --set-rpath``. patchelf runs ``--remove-rpath`` first, so only its
    copy reaches the artifact -- which is why this asserts that both consumers really do read the
    one declaration, rather than that two literals happen to agree today.

    Set equality rather than membership, so an entry silently added to the shipped path fails here
    too and has to be justified.
    """
    cmake = (
        _REPO_ROOT / "py/torch-tensorrt-executorch-runtime/native/CMakeLists.txt"
    ).read_text(encoding="utf-8")
    assert "BUILD_WITH_INSTALL_RPATH ON" in cmake
    assert "-Wl,-Bsymbolic" not in cmake

    declared = re.search(
        r'set\(\s*TORCH_TENSORRT_DELEGATE_RUNPATH\s+((?:"[^"]+"\s*)+)\)', cmake
    )
    assert (
        declared
    ), "the RUNPATH list is no longer a single declaration this test can read"
    entries = set(re.findall(r'"([^"]+)"', declared.group(1)))
    # libexecutorch.so belongs to the executorch distribution, not this one, so the delegate has
    # to reach out of its own package to find it. Two levels: this artifact installs under lib/ in
    # the package directory, so site-packages is two levels above $ORIGIN.
    # Exactly the directories the delegate's own DT_NEEDED entries resolve through. No torch/lib:
    # this wheel links no torch. Both CUDA layouts, because the two majors package their runtime
    # differently: cu13 wheels install nvidia/cu13/lib and cu12 wheels install
    # nvidia/cuda_runtime/lib, and torch-tensorrt publishes both channels.
    assert entries == {
        "$ORIGIN",
        "$ORIGIN/../../executorch/lib",
        "$ORIGIN/../../tensorrt_libs",
        "$ORIGIN/../../nvidia/cu13/lib",
        "$ORIGIN/../../nvidia/cuda_runtime/lib",
    }
    # The single-level form is the bug this guards against: it was right when the artifact
    # installed flat in the package directory, and it is wrong now that it installs under lib/.
    # A stale one-level entry resolves to site-packages/torch_tensorrt_executorch_runtime
    # instead of site-packages, so the delegate cannot find libexecutorch.so and a C++ consumer
    # fails to link it with undefined references to cudart and nvinfer.
    assert "$ORIGIN/../executorch/lib" not in cmake

    # The linker's copy has to say the same thing. It does not reach the artifact, since patchelf
    # removes it, but a build without patchelf and every in-tree consumer read it, and two lists
    # that are supposed to be the same path are a bug once they disagree.
    # Both consumers have to read the declaration rather than restate it, or the deduplication
    # is cosmetic and the copies can drift apart again.
    assert (
        'INSTALL_RPATH "${TORCH_TENSORRT_DELEGATE_RUNPATH}"' in cmake
    ), "INSTALL_RPATH no longer reads the shared RUNPATH declaration"
    assert (
        '"${TORCH_TENSORRT_DELEGATE_RUNPATH_COLONS}"' in cmake
    ), "patchelf --set-rpath no longer reads the shared RUNPATH declaration"
    assert re.search(
        r'string\(JOIN ":" TORCH_TENSORRT_DELEGATE_RUNPATH_COLONS\s+\$\{TORCH_TENSORRT_DELEGATE_RUNPATH\}\)',
        cmake,
    ), "the colon-joined form is not derived from the same list"

    # DT_RUNPATH, not the older DT_RPATH: --force-rpath would flip the tag, and the pinned
    # ExecuTorch passes --enable-new-dtags precisely to avoid it. Comments are stripped first,
    # because the CMakeLists explains this in prose and the prose names the flag.
    code = "\n".join(
        line for line in cmake.splitlines() if not line.lstrip().startswith("#")
    )
    assert "--force-rpath" not in code


@pytest.mark.unit
def test_runtime_extension_consumes_the_prebuilt_executorch_runtime():
    """The wheel must link ExecuTorch's shipped runtime, not rebuild one of its own.

    Rebuilding it would give the delegate a second copy of the backend registry and of the
    caller-stream thread-local, so registration would land somewhere the user's ExecuTorch
    never reads. Both spellings are asserted because the build is only correct if it takes the
    runtime from the package and never adds ExecuTorch's own source tree.
    """
    cmake = (
        _REPO_ROOT / "py/torch-tensorrt-executorch-runtime/native/CMakeLists.txt"
    ).read_text(encoding="utf-8")

    assert "find_package(executorch REQUIRED)" in cmake

    # Scoped to the link call: both names also appear in the required-target guard above it, so
    # searching the whole file would pass even if the delegate linked neither. Matched by
    # balancing parens rather than with a non-greedy regex, which would stop at the first `)`
    # and silently truncate the block if a generator expression were added to the call.
    opening = re.search(r"target_link_libraries\(executorch_backend_tensorrt\b", cmake)
    assert opening, "the delegate no longer links anything"
    depth, end = 1, None
    for index in range(opening.end(), len(cmake)):
        if cmake[index] == "(":
            depth += 1
        elif cmake[index] == ")":
            depth -= 1
            if depth == 0:
                end = index
                break
    assert end is not None, "unbalanced target_link_libraries call"
    linked = cmake[opening.end() : end]
    for required in ("executorch::runtime", "executorch::extension_cuda"):
        assert required in linked, f"the delegate does not link {required}"

    code = [line for line in cmake.splitlines() if not line.lstrip().startswith("#")]
    for forbidden in ("add_subdirectory", "EXECUTORCH_BUILD_"):
        offenders = [line for line in code if forbidden in line]
        assert not offenders, (
            f"{forbidden} builds ExecuTorch from source, which defeats the point of "
            f"linking its prebuilt runtime: {offenders}"
        )


@pytest.mark.unit
def test_the_delegate_cannot_outgrow_the_runtime_it_loads_beside():
    """The post-build guard must compare C++ symbol versions against the pinned runtime.

    This wheel bundles no libstdc++, so both the delegate and ExecuTorch resolve against
    whatever the host provides. A delegate built with a newer toolchain can require a GLIBCXX or
    CXXABI version the host lacks while the ExecuTorch beside it loads fine, and that failure
    appears on the user's machine rather than in the build, because the build container's own
    toolchain libraries sit on LD_LIBRARY_PATH. Comparing against libexecutorch.so rather than a
    hardcoded floor keeps the check honest when the pin moves.
    """
    cmake = (
        _REPO_ROOT / "py/torch-tensorrt-executorch-runtime/native/CMakeLists.txt"
    ).read_text(encoding="utf-8")
    guard = (
        _REPO_ROOT
        / "py/torch-tensorrt-executorch-runtime/native/check_imports_executorch_runtime.sh"
    ).read_text(encoding="utf-8")

    # The runtime has to be handed to the guard, or it has nothing to compare against.
    assert "$<TARGET_FILE:executorch::runtime>" in cmake
    for family in ("GLIBCXX", "CXXABI"):
        assert family in guard, f"the guard does not look at {family} versions"
    # Ordered numerically field by field, or 3.4.9 would outrank 3.4.21. The sort appears at two
    # sites, highest() and the comparison itself, and mutating one alone leaves the other's copy to
    # satisfy a single-occurrence check. Require both, so the behavioural case below is not the only
    # thing standing between a text sort in highest() and a wrongly rejected delegate.
    assert (
        guard.count("sort -t. -k1,1n -k2,2n") >= 2
    ), "the numeric version sort is not applied at both highest() and the comparison"


@pytest.mark.unit
def test_the_delegate_ships_no_absolute_runpath():
    """ExecuTorch's imported targets add the build machine's own path, and it must not ship.

    The entry arrives as a raw ``INTERFACE_LINK_OPTIONS`` ``-rpath``, so
    ``BUILD_WITH_INSTALL_RPATH`` does not suppress it and ``cmake --install`` does not rewrite
    it. It also sorts ahead of the relative entries, so on any host whose site-packages path
    matches the builder's the loader never consults ``$ORIGIN/../../executorch/lib`` -- which is
    what let an earlier wrong RUNPATH depth go unnoticed. Stripping it is what makes the
    relative entries load-bearing, and therefore testable.
    """
    cmake = (
        _REPO_ROOT / "py/torch-tensorrt-executorch-runtime/native/CMakeLists.txt"
    ).read_text(encoding="utf-8")
    guard = (
        _REPO_ROOT
        / "py/torch-tensorrt-executorch-runtime/native/check_imports_executorch_runtime.sh"
    ).read_text(encoding="utf-8")

    # Comments name --set-rpath in prose (the CMakeLists explains why it avoids --force-rpath), so
    # deleting the whole patchelf command would leave a raw-text "--set-rpath" in cmake satisfied by
    # that comment. Match the live command instead: --remove-rpath then --set-rpath reading the
    # shared colon-joined declaration, all in code with comments stripped.
    code = "\n".join(
        line for line in cmake.splitlines() if not line.lstrip().startswith("#")
    )
    assert "--remove-rpath" in code
    assert re.search(
        r'--set-rpath\s*\n?\s*"\$\{TORCH_TENSORRT_DELEGATE_RUNPATH_COLONS\}"', code
    ), "patchelf --set-rpath no longer reads the shared RUNPATH declaration"
    declared = re.search(
        r'set\(\s*TORCH_TENSORRT_DELEGATE_RUNPATH\s+((?:"[^"]+"\s*)+)\)', cmake
    )
    assert (
        declared
    ), "the RUNPATH list is no longer a single declaration this test can read"
    for entry in re.findall(r'"([^"]+)"', declared.group(1)):
        assert entry.startswith("$ORIGIN"), f"{entry} is not relative to the artifact"
    # And the guard has to assert the strip happened, or a regression ships silently.
    assert "not relative to the artifact" in guard


@pytest.mark.unit
def test_the_in_tree_target_survives_as_needed():
    """A registration-only library is dropped by --as-needed unless the link says otherwise.

    The delegate registers from a static initializer, so a consumer references no symbol from it.
    Measured on a real link: with a plain target the DT_NEEDED entry disappears under
    ``-Wl,--as-needed`` and the initializer never runs, silently. ExecuTorch wraps its own
    registration-only component libraries in scoped retention for exactly this reason, so the alias
    this file advertises has to carry it too or the advertised parity is false.

    Source-text only, deliberately: a matching text pattern still passes when the retention is
    dead code (``if(FALSE)``, ``if(WIN32)``, a reordered option list). The behaviour itself is
    covered by test_a_consumer_of_the_alias_keeps_the_delegate_linked, which links a consumer.
    """
    cmake = (
        _REPO_ROOT / "py/torch-tensorrt-executorch-runtime/native/CMakeLists.txt"
    ).read_text(encoding="utf-8")
    code = "\n".join(
        line for line in cmake.splitlines() if not line.lstrip().startswith("#")
    )
    # The whole option in one pattern, in order: a list whose --pop-state precedes the library,
    # or whose library is not the generator expression, retains nothing.
    assert re.search(
        r'"LINKER:--push-state,--no-as-needed,'
        r'\$<TARGET_FILE:executorch_backend_tensorrt>,--pop-state"',
        code,
    ), "the retention option is missing, reordered, or no longer names the delegate"
    assert re.search(
        r"target_link_options\(\s*executorch_backend_tensorrt\s+INTERFACE", code
    )
    # Guarded on Linux, and on nothing narrower: if(FALSE) and if(WIN32) both disable it while
    # leaving the option text above intact.
    guard_line = re.search(
        r"if\((.*?)\)\s*\n\s*target_link_options\(\s*executorch_backend_tensorrt\s+INTERFACE",
        code,
    )
    assert (
        guard_line
    ), "the retention is not inside a platform condition this test can read"
    assert (
        guard_line.group(1) == 'CMAKE_SYSTEM_NAME STREQUAL "Linux"'
    ), f"retention is conditioned on {guard_line.group(1)!r}, so it does not apply on Linux builds"


@pytest.mark.unit
def test_a_consumer_of_the_alias_keeps_the_delegate_linked(tmp_path):
    """Link a real consumer and check the DT_NEEDED survives.

    The text assertions above pin the option's shape, but a shape is not a behaviour: the option
    can be present and still retain nothing. This runs the production CMake file itself against a
    stand-in target of the same name, links a consumer that references no symbol from it under
    ``-Wl,--as-needed``, and requires both that the dependency is retained and that the static
    initializer runs. Including the real file rather than copying a regex match out of it is what
    makes an outer ``if(FALSE)`` around the retention block visible: a copy is still a copy of a
    line that production may no longer execute. Skipped where the toolchain is absent.
    """
    cmake_bin = shutil.which("cmake")
    if cmake_bin is None or shutil.which("readelf") is None or sys.platform != "linux":
        pytest.skip("needs cmake, readelf, and a Linux linker")

    production = (
        _REPO_ROOT / "py/torch-tensorrt-executorch-runtime/native/CMakeLists.txt"
    )
    # Everything from the retention comment to the install() call: the block under test, lifted
    # whole so any condition wrapping it comes along. Anchored on the comment rather than the
    # if(), so an outer guard cannot be left behind.
    block = re.search(
        r"\n(# Registration happens in a static initializer.*?)\ninstall\(TARGETS",
        production.read_text(encoding="utf-8"),
        re.DOTALL,
    )
    assert (
        block
    ), "the retention block is no longer identifiable in the production CMake file"
    retention = block.group(1).replace("executorch_backend_tensorrt", "delegate")

    # Lifting the block proves the flags work; it cannot see a condition wrapped around them
    # upstream. So also require the production block to be unconditional: flipping
    # CMAKE_SYSTEM_NAME around it left this test green while the flags reached no build.
    prologue = production.read_text(encoding="utf-8")[: block.start(1)]
    open_conditions: list[str] = []
    for line in prologue.splitlines():
        stripped = line.strip()
        if re.match(_CMAKE_BLOCK_OPEN, stripped, re.IGNORECASE):
            open_conditions.append(stripped)
        elif re.match(_CMAKE_BLOCK_CLOSE, stripped, re.IGNORECASE) and open_conditions:
            open_conditions.pop()
    assert not open_conditions, (
        "the retention block sits inside a conditional, so the flags it sets may not reach the "
        f"build this test proves them against: {open_conditions}"
    )

    (tmp_path / "reg.cpp").write_text(
        '#include <cstdio>\nnamespace { struct R { R() { printf("registered\\n"); } } r; }\n'
    )
    (tmp_path / "main.cpp").write_text("int main() { return 0; }\n")
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.24)\n"
        "project(retention CXX)\n"
        "add_library(delegate SHARED reg.cpp)\n"
        f"{retention}\n"
        "add_library(ns::delegate ALIAS delegate)\n"
        "add_executable(app main.cpp)\n"
        "target_link_options(app PRIVATE -Wl,--as-needed)\n"
        "target_link_libraries(app PRIVATE ns::delegate)\n"
        'set_target_properties(app PROPERTIES BUILD_RPATH "$ORIGIN")\n'
    )
    build = tmp_path / "build"
    # Fail rather than skip: the fixture is generated from the production block, so a
    # configure or build error is usually that block being malformed, which is the thing under
    # test. Skipping on it would turn the interesting failure into a silent pass.
    for stage in (
        [cmake_bin, "-S", str(tmp_path), "-B", str(build)],
        [cmake_bin, "--build", str(build)],
    ):
        done = subprocess.run(stage, capture_output=True, text=True)
        assert done.returncode == 0, (
            f"the fixture project failed at {' '.join(stage[1:3])}:\n"
            f"{done.stdout}\n{done.stderr}"
        )

    needed = subprocess.run(
        ["readelf", "-dW", str(build / "app")], capture_output=True, text=True
    ).stdout
    assert (
        "libdelegate.so" in needed
    ), "the linker dropped the registration-only dependency despite the retention option"
    ran = subprocess.run([str(build / "app")], capture_output=True, text=True)
    assert "registered" in ran.stdout, "the static initializer never ran"


def _assert_the_checker_is_reachable(prologue: str) -> None:
    """Fail if anything in ``prologue`` can stop the wheel checker from running.

    Checked with ``bash -n`` and anchored scans, never by executing. An earlier version sliced the
    raw YAML and ran it with ``bash -c``, which downloaded bazelisk, put it on PATH and pip
    installed ExecuTorch, once per parameter case.
    """
    parsed = subprocess.run(
        ["bash", "-n", "-c", prologue], capture_output=True, text=True
    )
    # An unterminated compound command leaves the prologue an incomplete script, which is what a
    # condition wrapped around the checker produces.
    assert "unexpected end of file" not in parsed.stderr, (
        "the wheel checker runs under an unterminated shell condition, so the rules this test "
        f"proves may never execute in CI: {parsed.stderr.strip()[:200]}"
    )
    # Syntax is not reachability: these parse cleanly and still skip the checker.
    for pattern, why in (
        (r"^[ \t]*if\b[^\n]*\bfi[ \t]*$", "an inline conditional"),
        (r"^[ \t]*(?:false|true)[ \t]*(?:&&|\|\|)", "a short-circuit that skips it"),
    ):
        offender = re.search(pattern, prologue, re.MULTILINE)
        assert not offender, (
            f"the wheel checker sits after {why}, so it may never run: "
            f"{offender.group(0).strip()[:80]!r}"
        )
    # An unconditional exit skips the checker whatever its indentation, so a column-0 scan misses
    # an indented `exit 0`. But the real prologue legitimately exits from inside a case arm for an
    # unsupported platform, so a scan that flags any indented exit is a false positive. Track block
    # depth instead: exit, exec or return is unconditional only at depth 0, outside every
    # if/case/for/while/until block.
    #
    # Split each line into commands on the shell separators too, or `: && exit 0`, `foo || exit 1`
    # and `{ exit 0; }` slip past a scan that only reads the first word: the exit is unconditional
    # but does not start the line. Heredoc bodies are skipped rather than scanned, or a body line
    # beginning with `if` desynchronises the depth counter and hides a later top-level exit. `exec`
    # is only a bypass when it replaces the shell with another program: a bare `exec 3>&1` or
    # `exec >log` is a redirection that returns, so it does not count.
    depth = 0
    block_opener = re.compile(r"^\s*(?:if|case|for|while|until|select)\b")
    block_closer = re.compile(r"^\s*(?:fi|esac|done)\b")
    heredoc_delimiter = None
    for raw_line in prologue.splitlines():
        stripped = raw_line.strip()
        if heredoc_delimiter is not None:
            if stripped == heredoc_delimiter:
                heredoc_delimiter = None
            continue
        if not stripped or stripped.startswith("#"):
            continue
        opening_heredoc = re.search(
            r"<<-?\s*[\"']?([A-Za-z_][A-Za-z0-9_]*)[\"']?", raw_line
        )
        if opening_heredoc:
            heredoc_delimiter = opening_heredoc.group(1)
        if block_closer.match(raw_line):
            depth = max(0, depth - 1)
            continue
        # An exit inside an inline block on this same line is conditional, not a bypass:
        # `if ...; then exit 1; fi` and `case x in ...) exit 0 ;; esac` guard the exit behind
        # `then` or `do` or a case pattern. Only the part of the line before any such keyword runs
        # unconditionally, so scan that prefix: it still sees `: && exit 0`, `foo || exit 1` and
        # `{ exit 0; }`, but not an exit the same line makes conditional.
        unconditional_prefix = re.split(r"\b(?:then|do|in)\b", raw_line, maxsplit=1)[0]
        if depth == 0:
            for command in re.split(r"&&|\|\||;|\{|\}", unconditional_prefix):
                command = command.strip()
                if re.match(r"(?:exit|return)\b", command) or re.match(
                    r"exec\s+[^0-9<>&]", command
                ):
                    raise AssertionError(
                        "the wheel checker sits after an unconditional exit, so it may never "
                        f"run: {stripped[:80]!r}"
                    )
        if block_opener.match(raw_line) and not re.search(
            r"\b(?:fi|esac|done)\b", stripped
        ):
            depth += 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "case,expect_pass",
    [
        ("good", True),
        ("no_register_backend", False),
        ("defines_register_backend", False),
        ("no_runpath", False),
        ("wrong_depth_runpath", False),
        ("dt_rpath", False),
        ("runpath_missing_executorch", False),
        ("absolute_runpath", False),
        ("floor_above_runtime", False),
        ("no_cxxabi", False),
        ("glibc_above_runtime", False),
        ("named_node_missing_from_runtime", False),
        ("lower_compatible_nodes", True),
        # GLIBCXX_3.4.22 is below the manylinux_2_28 ceiling of 3.4.25, so it is legitimately
        # accepted now: the guard compares against the platform, not against the sibling wheel.
        ("one_std_thread_above_the_runtime", True),
        ("above_the_runtime", False),
        # GCC_4.8.0 is below the platform ceiling, so no longer a rejection.
        ("gcc_above_the_runtime", True),
        ("cxxabi_above_the_runtime", False),
        # A family the runtime declares nothing from is fine: the host provides libstdc++, not the
        # sibling wheel. GLIBCXX_3.4.21 is far below the platform ceiling.
        ("family_absent_from_runtime", True),
        ("no_needed_executorch", False),
        ("no_needed_extension_cuda", False),
        ("no_needed_libstdcxx", False),
        # Both majors accepted: torch-tensorrt publishes cu12 and cu13 channels, and the
        # RUNPATH carries the layout directory each one uses.
        ("cuda_12_runtime", True),
        ("cuda_13_runtime", True),
        ("runtime_only_imports_register_backend", False),
        ("runtime_exports_a_near_miss", False),
        ("elfutils_bare_rpath", False),
        ("elfutils_undef_dialect", True),
        ("readelf_v_broken", False),
        ("readelf_broken", False),
        # The --dyn-syms check: an unversioned std C++ UND is rejected, a versioned one accepted,
        # and a readelf that cannot list dynamic symbols fails closed.
        ("unversioned_cxx_undef", False),
        ("versioned_cxx_undef", True),
        ("readelf_dynsyms_broken", False),
        # Exercises highest()'s numeric sort: accepted numerically, rejected under a text sort.
        ("runtime_numbered_nodes_out_of_text_order", True),
        # Exercises the 4-argument exact-whole-RUNPATH branch production always selects.
        ("runpath_missing_a_sibling", False),
        # Keeps the 3-argument fallback branch covered.
        ("runpath_fallback_reaches_executorch", True),
        # The pybindings extension seeds the symbol-version ceiling; a missing one must hard-fail
        # rather than silently narrow the guard.
        ("no_pybindings_extension", False),
    ],
)
def test_the_guard_actually_rejects_a_bad_artifact(tmp_path, case, expect_pass):
    """Run the guard, rather than reading it.

    Every other assertion in this file checks that the guard's *source* contains certain words.
    None of them notice if the guard is never invoked, or returns 0 unconditionally: replacing
    ``COMMAND sh`` with ``COMMAND true``, or inserting ``exit 0`` after ``set -u``, leaves them
    all green. The guard takes readelf as its first argument precisely so it can be driven, so
    drive it with a stub and require the right exit status for each artifact shape.
    """
    guard = (
        _REPO_ROOT
        / "py/torch-tensorrt-executorch-runtime/native/check_imports_executorch_runtime.sh"
    )

    # Default to the whole RUNPATH the build declares, so the exact-match branch passes and the
    # version and symbol checks are what decide each case. The RUNPATH-shape cases below override
    # it and run on the 3-argument fallback.
    runpaths = {
        "no_runpath": None,
        # Reached the required check as a basic regex, where the dots are wildcards.
        "wrong_depth_runpath": "$ORIGIN:$ORIGIN/xy/executorch/lib",
        "runpath_missing_executorch": "$ORIGIN:$ORIGIN/../torch/lib",
        "absolute_runpath": "$ORIGIN:$ORIGIN/../../executorch/lib:/build/site-packages/lib",
        # A RUNPATH the loader could resolve on the build host but that omits two of the four
        # sibling directories. Only the exact-whole-set branch, which production always selects,
        # rejects it; the fallback that spot-checks executorch/lib alone lets it through.
        "runpath_missing_a_sibling": "$ORIGIN:$ORIGIN/../../executorch/lib",
        # The 3-argument fallback: a bare relative RUNPATH that reaches executorch/lib is accepted
        # when no build string is given, which keeps that branch covered.
        "runpath_fallback_reaches_executorch": "$ORIGIN:$ORIGIN/../../executorch/lib",
        "dt_rpath": _GUARD_GOOD_RUNPATH,
        "elfutils_bare_rpath": _GUARD_GOOD_RUNPATH,
    }
    default_runpath = _GUARD_GOOD_RUNPATH
    tag = "RPATH" if case == "dt_rpath" else "RUNPATH"
    # No DT_NEEDED on the runtime means the delegate resolves register_backend from nowhere. This
    # case drops only this line and keeps the extension_cuda line below, so the libexecutorch.so
    # branch is the one that rejects it and the case pins that branch.
    dyn = (
        ""
        if case == "no_needed_executorch"
        else " 0x0000000000000001 (NEEDED) Shared library: [libexecutorch.so]\n"
    )
    # extension_cuda is linked PRIVATE and shared, so a well-formed delegate carries this DT_NEEDED.
    # The no_needed_extension_cuda case drops it to exercise the static-link rejection. The
    # no_needed_executorch case keeps it, so only the libexecutorch.so line is missing and the
    # branch that case is named for is the one that fires, not this one two checks below.
    if case != "no_needed_extension_cuda":
        dyn += (
            " 0x0000000000000001 (NEEDED) Shared library: "
            "[libexecutorch_extension_cuda.so]\n"
        )
    # The delegate calls out-of-line libstdc++ functions, so a well-formed one records a
    # DT_NEEDED on the C++ runtime. The no_needed_libstdcxx case drops it to exercise the
    # under-linked rejection the guard adds for the symbol lld silently discards.
    if case != "no_needed_libstdcxx":
        dyn += " 0x0000000000000001 (NEEDED) Shared library: [libstdc++.so.6]\n"
    # Both CUDA majors resolve, each through its own layout directory in the RUNPATH.
    if case == "cuda_12_runtime":
        dyn += " 0x0000000000000001 (NEEDED) Shared library: [libcudart.so.12]\n"
    if case == "cuda_13_runtime":
        dyn += " 0x0000000000000001 (NEEDED) Shared library: [libcudart.so.13]\n"
    rp = runpaths.get(case, default_runpath)
    if rp:
        if case == "elfutils_bare_rpath":
            # eu-readelf prints the tag bare, where binutils parenthesises it. The guard claims to
            # reject DT_RPATH in either dialect, so exercise the one binutils never emits.
            dyn += f"  RPATH                Library rpath: [{rp}]\n"
        else:
            dyn += f" 0x000000000000001d ({tag})   Library runpath: [{rp}]\n"
    # The mangled name, because that is what the guard greps .dynsym for.
    mangled = "_ZN10executorch7runtime16register_backendERKNS0_7BackendE"
    # eu-readelf spells an undefined symbol UNDEF where binutils spells it UND. Both must be
    # accepted, so one case uses the elfutils spelling.
    undefined = "UNDEF" if case == "elfutils_undef_dialect" else "UND"
    syms = "" if case == "no_register_backend" else f"  1: {undefined} {mangled}\n"
    if case == "defines_register_backend":
        syms = f"  1: 000123 FUNC GLOBAL DEFAULT 12 {mangled}\n"
    # What the runtime's own symbol table says. A defined export carries a section index; the
    # runtime_only_imports case carries UND instead, which is a runtime that imports the symbol
    # rather than providing it, and must be rejected.
    runtime_syms = f"  1: 000123    82 FUNC GLOBAL DEFAULT 8 {mangled}\n"
    if case == "runtime_only_imports_register_backend":
        runtime_syms = f"  1: 000000     0 FUNC GLOBAL DEFAULT UND {mangled}\n"
    if case == "runtime_exports_a_near_miss":
        runtime_syms = (
            "  1: 000456    82 FUNC GLOBAL DEFAULT 8 "
            "_ZN10executorch7runtime16register_backendERKNS0_9BackendV2E\n"
        )
    # The delegate's real floor is CXXABI_1.3.9 and no GLIBCXX; the runtime tops out at 3.4.21.
    target_v = "CXXABI_1.3.9"
    if case == "floor_above_runtime":
        target_v = "GLIBCXX_3.4.30 CXXABI_1.3.9"
    if case == "no_cxxabi":
        target_v = ""
    # GLIBC is a family of its own: a delegate needing a newer glibc than the runtime fails on
    # the same hosts, and it was not compared at all before review.
    if case == "glibc_above_runtime":
        target_v = "CXXABI_1.3.9 GLIBC_2.38"
    # CXXABI_TM_1 carries no dotted version, so a pattern demanding digits drops it silently.
    if case == "named_node_missing_from_runtime":
        target_v = "CXXABI_1.3.9 CXXABI_TM_1"
    # Must be ACCEPTED. Symbol versioning is backward compatible: a runtime declaring GLIBC_2.34
    # satisfies a delegate needing GLIBC_2.4, and one declaring GLIBCXX_3.4.21 satisfies
    # GLIBCXX_3.4.11. An exact-set check rejected exactly this and broke the build against the
    # runtime the delegate is pinned to.
    # Must be REJECTED, and this is the case a manylinux baseline wrongly accepted: one step above
    # the runtime's own maximum, which is all a std::thread costs. The wheel carries a bare
    # linux_x86_64 tag, so nothing promises a host provides 3.4.22 just because it is old.
    if case == "one_std_thread_above_the_runtime":
        target_v = "CXXABI_1.3.9 GLIBCXX_3.4.22"
    # Must be REJECTED. Further above still: GLIBCXX_3.4.26 is GCC 9's std::filesystem.
    if case == "above_the_runtime":
        target_v = "CXXABI_1.3.9 GLIBCXX_3.4.26"
    # Must be REJECTED. GCC is its own family and had no case at all: dropping it from the loop
    # left the whole file green, while dropping GLIBC turned a case red. It is also the family
    # that discriminates most sharply in practice, spanning GCC_3.0 to GCC_4.0.0 inside the pinned
    # ExecuTorch wheel.
    if case == "gcc_above_the_runtime":
        target_v = "CXXABI_1.3.9 GCC_4.8.0"
    # Must be REJECTED. CXXABI had the same gap GCC did: dropping it from the loop left every case
    # green, because every other case declares a CXXABI the runtime satisfies.
    if case == "cxxabi_above_the_runtime":
        target_v = "CXXABI_1.3.15"
    # Must be REJECTED, with a message that does not call the absence a version number. A family
    # the runtime declares nothing from means it uses none of that library, so nothing beside the
    # delegate guarantees a host provides what the delegate asks for.
    if case == "family_absent_from_runtime":
        target_v = "CXXABI_1.3.9 GLIBCXX_3.4.21"
    if case == "lower_compatible_nodes":
        target_v = "CXXABI_1.3 CXXABI_1.3.9 GLIBC_2.4 GLIBC_2.17 GLIBCXX_3.4.11 GCC_3.0"
    # Must be ACCEPTED, and it is the one case that exercises highest()'s own numeric sort rather
    # than the comparison's. The delegate needs GLIBCXX_3.4.21; the runtime declares 3.4.9 and
    # 3.4.21. Numerically the runtime's ceiling is 3.4.21 and the delegate is within it, but a
    # highest() that sorted as text would pick 3.4.9 as the runtime maximum and reject the delegate
    # against the very runtime it is pinned to. The comparison site's own sort cannot cause this:
    # it only ever compares the delegate's single required node against the ceiling highest() found.
    if case == "runtime_numbered_nodes_out_of_text_order":
        target_v = "CXXABI_1.3.9 GLIBCXX_3.4.21"
    # The runtime declares a spread, not just its maximum, the way a real library does.
    runtime_v = "GLIBCXX_3.4 GLIBCXX_3.4.21 CXXABI_1.3 CXXABI_1.3.9 GLIBC_2.2.5 GLIBC_2.34 GCC_3.0"
    if case == "family_absent_from_runtime":
        runtime_v = "CXXABI_1.3 CXXABI_1.3.9 GLIBC_2.2.5 GLIBC_2.34 GCC_3.0"
    # Two numbered GLIBCXX nodes whose text order inverts their numeric order: text sort ranks
    # 3.4.9 above 3.4.21, numeric sort ranks 3.4.21 above 3.4.9.
    if case == "runtime_numbered_nodes_out_of_text_order":
        runtime_v = (
            "GLIBCXX_3.4.9 GLIBCXX_3.4.21 CXXABI_1.3 CXXABI_1.3.9 GLIBC_2.34 GCC_3.0"
        )
    # What readelf --dyn-syms lists for the target. The guard rejects an UNVERSIONED undefined std
    # C++ symbol (a newer-toolchain helper the nonshared archive failed to supply) and accepts one
    # that carries an @GLIBCXX version. Default to a well-formed versioned symbol so the ordinary
    # cases pass this check; the two named cases below drive the reject and accept branches.
    dyn_syms = (
        "    1: 0000000000000000 0 FUNC GLOBAL DEFAULT UND "
        "_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_appendEPKcm@GLIBCXX_3.4.21\n"
    )
    if case == "unversioned_cxx_undef":
        dyn_syms = (
            "    1: 0000000000000000 0 FUNC GLOBAL DEFAULT UND "
            "_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE15_M_replace_coldEPcmPKcmm\n"
        )
    if case == "versioned_cxx_undef":
        dyn_syms = (
            "    1: 0000000000000000 0 FUNC GLOBAL DEFAULT UND "
            "_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_appendEPKcm@GLIBCXX_3.4.21\n"
        )
    stub = tmp_path / "readelf"
    stub.write_text(
        "#!/bin/sh\n" + ("exit 3\n" if case == "readelf_broken" else "")
        # -V broken for the target only, not the runtime: the case is named for the target-side
        # read, so breaking both lets the runtime-side read decide it and the named branch never
        # runs. The target is the second argument to -V; the runtime ends in libexecutorch.so.
        + (
            '[ "$1" = "-V" ] && case "$2" in *libexecutorch.so) ;; *) exit 3 ;; esac\n'
            if case == "readelf_v_broken"
            else ""
        )
        + (
            '[ "$1" = "--dyn-syms" ] && exit 3\n'
            if case == "readelf_dynsyms_broken"
            else ""
        )
        + 'case "$1" in\n'
        f"  -d) printf %s '{dyn}' ;;\n"
        f"  --dyn-syms) printf %s '{dyn_syms}' ;;\n"
        '  -Ws) case "$2" in\n'
        f"        *libexecutorch.so) printf %s '{runtime_syms}' ;;\n"
        f"        *) printf %s '{syms}' ;;\n"
        "      esac ;;\n"
        '  -V) case "$2" in\n'
        f"        *libexecutorch.so) echo '{runtime_v}' ;;\n"
        f"        *) echo '{target_v}' ;;\n"
        "      esac ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    # The runtime lives in a per-case lib/ subdirectory so the guard's ../extension/pybindings path
    # resolves inside this case's tmp_path rather than the parent tmp_path shared across the whole
    # parametrization. Sharing it let one case's pybindings stub leak into no_pybindings_extension.
    libdir = tmp_path / "lib"
    libdir.mkdir()
    target = libdir / "libexecutorch_backend_tensorrt.so"
    target.write_bytes(b"\x7fELF")
    runtime = libdir / "libexecutorch.so"
    runtime.write_bytes(b"\x7fELF")

    # The guard seeds its symbol-version ceiling from the pybindings extension, which sits at
    # runtime_dir/../extension/pybindings/_C.*.so, so a well-formed layout has it. The
    # no_pybindings_extension case leaves it out to exercise the hard-fail that keeps a missing
    # extension from silently narrowing the ceiling.
    if case != "no_pybindings_extension":
        pybindings_dir = tmp_path / "extension" / "pybindings"
        pybindings_dir.mkdir(parents=True, exist_ok=True)
        (pybindings_dir / "_C.cpython-311-x86_64-linux-gnu.so").write_bytes(b"\x7fELF")

    # Production always passes the fourth argument (TORCH_TENSORRT_DELEGATE_RUNPATH_COLONS, the
    # RUNPATH the build declares), which selects the exact-whole-RUNPATH branch. Drive that branch
    # here with the RUNPATH the build declares, except for the cases kept on the 3-argument
    # fallback so it stays covered too.
    argv = ["sh", str(guard), str(stub), str(target), str(runtime)]
    if case not in _THREE_ARG_CASES:
        argv.append(_GUARD_GOOD_RUNPATH)
        # The manylinux tag the row ships under. Without it the guard skips the symbol version
        # ceiling entirely, so every ceiling case would pass for the wrong reason.
        argv.append("manylinux_2_28")

    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
    )
    # Named branches whose parametrize case must reach that branch and no other. Asserting the
    # exit status alone let a case pass by any route that also exits non-zero: the three readelf
    # and no-RUNPATH cases each survived their own branch being deleted because a later check still
    # failed. Requiring the branch's own message pins each case to the branch it is named for.
    expected_messages = {
        "no_runpath": "carries no RUNPATH",
        "no_needed_executorch": "has no DT_NEEDED on libexecutorch.so",
        "no_needed_libstdcxx": "has no DT_NEEDED on libstdc++",
        "readelf_broken": "could not inspect",
        "readelf_v_broken": "could not read symbol versions of",
        "unversioned_cxx_undef": "unversioned undefined C++ runtime symbols",
        "readelf_dynsyms_broken": "could not read the dynamic symbols of",
        "no_pybindings_extension": "could not find the pybindings extension",
    }
    if expect_pass:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0, f"{case} was accepted:\n{result.stdout}"
        expected = expected_messages.get(case)
        if expected is not None:
            assert expected in result.stderr, (
                f"{case} failed, but not through its own branch: expected {expected!r} in\n"
                f"{result.stderr}"
            )


@pytest.mark.unit
def test_the_guard_is_wired_into_the_build():
    """The guard has to be invoked, not merely present.

    Replacing ``COMMAND sh`` with ``COMMAND true`` in the POST_BUILD rule disables the check
    completely and leaves every source-text assertion in this file green, so pin the wiring:
    a POST_BUILD command on the delegate that runs this script with the two artifacts.
    """
    cmake = (
        _REPO_ROOT / "py/torch-tensorrt-executorch-runtime/native/CMakeLists.txt"
    ).read_text(encoding="utf-8")
    code = "\n".join(
        line for line in cmake.splitlines() if not line.lstrip().startswith("#")
    )
    invocation = re.search(
        r"add_custom_command\(\s*TARGET\s+executorch_backend_tensorrt\s+POST_BUILD\s+"
        r"COMMAND\s+sh\s+\"\$\{CMAKE_CURRENT_LIST_DIR\}/check_imports_executorch_runtime\.sh\"",
        code,
    )
    assert invocation, "the guard is not invoked by a POST_BUILD command running sh"
    # The enclosing condition too. Mitigated by the FATAL_ERROR above it on Linux, but the
    # invocation being present says nothing about whether it is reached, and if(FALSE) here left
    # every other assertion in this test green.
    # Track block depth to the invocation rather than matching the nearest condition, so that
    # every enclosing condition is checked and not just the innermost.
    enclosing: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        opened = re.match(_CMAKE_BLOCK_OPEN, stripped, re.IGNORECASE)
        if opened:
            # Strip whatever the command name actually was, since a fixed-width slice assumes
            # one spelling and garbles "if (X)" and every command longer than "if".
            enclosing.append(stripped[opened.end() :].strip().rstrip(")"))
        elif re.match(r"else\s*\(|elseif\s*\(", stripped, re.IGNORECASE):
            # Case-insensitive like the opens and closes. An uppercase ELSE() was a false accept:
            # it left the recorded condition untouched while control moved into the else branch.
            if enclosing:
                enclosing[-1] = stripped
        elif re.match(_CMAKE_BLOCK_CLOSE, stripped, re.IGNORECASE):
            if enclosing:
                enclosing.pop()
        elif "check_imports_executorch_runtime.sh" in stripped:
            break
    else:
        raise AssertionError(
            "the guard invocation was not found while scanning conditions"
        )
    # And that nothing reassigns the variable before the block reads it: the condition being
    # spelled correctly says nothing if TORCH_TENSORRT_READELF is cleared one line above. The
    # variable is only ever meant to come from find_program, so reject any set() of it before the
    # guard regardless of the value. Enumerating CMake's falsy literals missed the quoted forms
    # set(TORCH_TENSORRT_READELF "OFF") and set(TORCH_TENSORRT_READELF "" CACHE INTERNAL ""), each
    # of which is falsy to if() and disables the whole block.
    guard_at = code.index("check_imports_executorch_runtime.sh")
    disabled = re.search(
        r"set\(\s*TORCH_TENSORRT_READELF\b",
        code[:guard_at],
        re.IGNORECASE,
    )
    assert (
        not disabled
    ), "TORCH_TENSORRT_READELF is reassigned before the guard block, so the guard may never run"
    assert enclosing == ["TORCH_TENSORRT_READELF"], (
        "the guard's POST_BUILD command must be reached whenever readelf exists, but it sits "
        f"under {enclosing}"
    )
    # Both artifacts, or the symbol-floor comparison silently degrades to the two-argument form.
    assert "$<TARGET_FILE:executorch_backend_tensorrt>" in code
    assert "$<TARGET_FILE:executorch::runtime>" in code
    # And the build's own RUNPATH string, or the guard falls back to spot-checking one entry and a
    # delegate missing tensorrt_libs or the CUDA entry ships. Scoped to the invocation's argument
    # list, since the variable is also set earlier in the file where patchelf consumes it.
    arguments = code[invocation.end() : code.index("VERBATIM", invocation.end())]
    assert "${TORCH_TENSORRT_DELEGATE_RUNPATH_COLONS}" in arguments, (
        "the guard is not given the RUNPATH the build asks for, so it cannot compare the whole "
        "set and a missing entry ships"
    )


@pytest.mark.unit
def test_the_delegate_is_exported_the_way_executorch_exports_its_backends():
    """The delegate must be linkable in-tree as executorch::backend_tensorrt.

    ExecuTorch exports every backend under that spelling, so a project already linking
    executorch::backend_cuda should not need a second convention for this one. In-tree only:
    find_package would need a generated package config file, and the wheel ships only the .so,
    so an install(EXPORT) here would produce targets files no consumer ever sees.
    """
    cmake = (
        _REPO_ROOT / "py/torch-tensorrt-executorch-runtime/native/CMakeLists.txt"
    ).read_text(encoding="utf-8")

    # The target, and therefore the shipped libexecutorch_backend_tensorrt.so.
    assert "add_library(executorch_backend_tensorrt SHARED" in cmake
    assert (
        "add_library(executorch::backend_tensorrt ALIAS executorch_backend_tensorrt)"
        in cmake
    )
    # The old name would ship the library as libtorch_tensorrt_executorch_backend.so.
    assert "torch_tensorrt_executorch_backend" not in cmake
    # An export set without a package config file is dead weight: find_package cannot resolve
    # it and the wheel does not carry it.
    code = [line for line in cmake.splitlines() if not line.lstrip().startswith("#")]
    assert not [line for line in code if "install(EXPORT" in line]

    # The package withholds every imported target below this, without failing find_package,
    # so a lower floor would configure cleanly and then fail on the first executorch:: target.
    assert "cmake_minimum_required(VERSION 3.28)" in cmake


def _setup_tree():
    return ast.parse(_SETUP_PY.read_text(encoding="utf-8"))


def _runtime_setup_tree():
    return ast.parse(_RUNTIME_SETUP_PY.read_text(encoding="utf-8"))


def _assignment_value(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return node.value
    raise AssertionError(f"Could not find assignment for {name}")


def _function_def(tree, name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Could not find function {name}")


@pytest.mark.unit
def test_runtime_wheel_uses_public_torch_version():
    function = _function_def(_runtime_setup_tree(), "public_version")
    namespace = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "<setup.py>", "exec"),
        namespace,
    )

    assert namespace["public_version"]("2.14.0.dev20260726+cu132") == (
        "2.14.0.dev20260726"
    )


@pytest.mark.unit
def test_runtime_wheel_version_refuses_the_unpublishable_placeholder(monkeypatch):
    """A from-source build must not record torch-tensorrt==2.15.0a0.

    version.txt carries that in-development placeholder, the root build strips the suffix for real
    artifacts, and it is published on no index, so a wheel that requires it cannot be installed by
    its own README command. The environment variable CI exports is honoured; its absence, with no
    generated _version.py, must raise rather than fall back to the placeholder.
    """
    function = _function_def(_runtime_setup_tree(), "torchtrt_version")
    namespace = {
        "os": os,
        "re": re,
        "REPO_ROOT": Path("/nonexistent-torch-tensorrt-checkout"),
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "<setup.py>", "exec"),
        namespace,
    )

    monkeypatch.setenv(
        "TORCH_TENSORRT_EXECUTORCH_RUNTIME_VERSION", "2.15.0.dev20260824"
    )
    assert namespace["torchtrt_version"]() == "2.15.0.dev20260824"

    monkeypatch.delenv("TORCH_TENSORRT_EXECUTORCH_RUNTIME_VERSION", raising=False)
    with pytest.raises(RuntimeError, match="2.15.0a0"):
        namespace["torchtrt_version"]()


@pytest.mark.unit
def test_runtime_readme_build_recipe_sets_the_version():
    """The README's from-source recipe must set the version the build now requires.

    torchtrt_version() fails closed rather than fall back to the unpublishable version.txt
    placeholder, so a recipe that does not export TORCH_TENSORRT_EXECUTORCH_RUNTIME_VERSION would
    stop at the build step. Assert the export inside the fenced recipe, not anywhere in the prose,
    so a sentence merely mentioning the variable cannot satisfy it. The value has to be non-empty
    and the line uncommented: an empty assignment or a commented-out export names the variable
    while setting nothing, so a substring check on the name alone passed over a recipe that would
    still stop at the build step.
    """
    readme = (_REPO_ROOT / "py/torch-tensorrt-executorch-runtime/README.md").read_text(
        encoding="utf-8"
    )
    recipes = [
        block
        # Any language tag, not just bash/sh: the README also carries a ```cmake block, and a
        # pattern that does not recognise an opener treats it as content, which shifts every
        # fence pair after it and hides the recipe this test exists to read.
        for block in re.findall(r"```[a-zA-Z]*\n(.*?)```", readme, re.DOTALL)
        if "pip wheel" in block
    ]
    assert recipes, "the README no longer carries a pip wheel build recipe"
    variable = "TORCH_TENSORRT_EXECUTORCH_RUNTIME_VERSION"
    for recipe in recipes:
        exports = [
            line
            for line in recipe.splitlines()
            if re.match(rf"\s*(?:export\s+)?{variable}=", line)
            and not line.lstrip().startswith("#")
        ]
        assert exports, (
            f"the build recipe does not set {variable} on an uncommented line, so the build would "
            "stop rather than record the version the wheel requires"
        )
        for line in exports:
            value = line.split("=", 1)[1].strip().strip("\"'")
            assert value, (
                f"the build recipe sets {variable} to an empty value, so the build records no "
                "version and stops"
            )


def _co_names_and_consts(code) -> list[str]:
    """Every name and string constant the compiled code actually carries, nested code included.

    Reading the code object rather than the source is what makes the reachability check meaningful:
    a commented-out tail leaves no trace here, while it stays fully visible to a substring search of
    the text.
    """
    import types as _types

    found = [*code.co_names, *code.co_varnames]
    for constant in code.co_consts:
        if isinstance(constant, str):
            found.append(constant)
        elif isinstance(constant, _types.CodeType):
            found.extend(_co_names_and_consts(constant))
    return found


@pytest.mark.unit
def test_ci_exercises_the_device_resident_boundary():
    """The device-resident path needs its own lane, and it is easy to lose silently.

    Every other program in this suite crosses the method boundary on the host, so nothing else
    would notice if ``skip_h2d_for_method_inputs`` stopped taking effect. Two pieces have to stay
    wired for that coverage to mean anything: the export, which asserts the serialized program
    contains no boundary copy operator, and the Python runner, which feeds a CUDA tensor and
    asserts the output never left the device.

    Also asserted: the device-resident program is NOT passed to the C++ reference runner. That
    runner feeds host tensors, which this program's contract forbids, so adding it there would
    look like more coverage while actually testing the wrong thing.
    """
    workflow = (_REPO_ROOT / ".github/workflows/executorch-test-linux.yml").read_text(
        encoding="utf-8"
    )

    assert "export_device_resident.py" in workflow, (
        "the device-resident program is never exported, so no lane checks that the "
        "skip_h2d/skip_d2h flags remove the boundary copies"
    )
    assert "load_model_device_resident.py" in workflow, (
        "the device-resident program is exported but never executed, so nothing checks "
        "that a CUDA input stays on the device at runtime"
    )

    reference_runner = re.search(
        r"verify-executorch-reference-runner\.sh((?:[^\n]*\\\n)*[^\n]*)", workflow
    )
    assert reference_runner, "the C++ reference runner invocation is gone"
    assert "device-resident" not in reference_runner.group(1), (
        "the device-resident .pte is handed to the C++ reference runner, which feeds host "
        "tensors; that program requires CUDA inputs, so this would test the wrong contract"
    )

    # The export must keep asserting on the program, not just set the flags and hope.
    export = (
        _REPO_ROOT / "examples/torchtrt_executorch_example/export_device_resident.py"
    ).read_text(encoding="utf-8")
    for fragment in (
        "skip_h2d_for_method_inputs=True",
        "skip_d2h_for_method_outputs=True",
        "alloc_graph_input=False",
        "alloc_graph_output=False",
        "_h2d_copy",
        "_d2h_copy",
        "plan.operators",
    ):
        assert fragment in export, (
            f"export_device_resident.py no longer contains {fragment!r}, so it does not "
            "prove the boundary copies are absent"
        )


@pytest.mark.unit
def test_ci_runtime_check_asserts_the_delegate_loads_and_registers():
    """The one CI step that runs the built delegate must keep proving it registers.

    executorch-test-linux.yml imports the delegate package, queries ExecuTorch's registry, and fails unless the
    TensorRT delegate and the stock backends are all registered. It is the only end-to-end proof
    that this change's delegate works rather than merely being shaped right, and nothing else
    exercises it, so the whole one-liner could be reduced to `import sys` unnoticed. Read the
    string the workflow runs and assert the pieces that make it a real check: it resolves the
    runtime, queries the registry, names the delegate backend, and fails through sys.exit rather
    than assert so `python -O` cannot compile the check away.
    """
    import yaml

    workflow_text = (
        _REPO_ROOT / ".github/workflows/executorch-test-linux.yml"
    ).read_text(encoding="utf-8")

    match = re.search(r"runtime_check='([^']*)'", workflow_text)
    assert (
        match
    ), "executorch-test-linux.yml no longer defines a runtime_check one-liner"
    runtime_check = match.group(1)

    # Every required fragment has to be REACHABLE, not merely present. The one-liner is Python, so a
    # single "#" anywhere in it comments out the rest of the statement while leaving the text intact
    # for the substring checks below: `import sys, torch; # ...the whole check...` satisfies all of
    # them and exits 0 on a machine with no delegate installed. Commenting out the leading print is
    # the natural first move when debugging this step, and the print comes first in the string. The
    # sibling reachability test strips whole-line YAML comments for the same reason, but a mid-line
    # "#" here is a Python comment, so that pass does not see it.
    #
    # Compile the string and read the code object, which is the only view that agrees with what the
    # interpreter will actually run.
    compiled = compile(runtime_check, "<runtime_check>", "exec")
    reachable = "\n".join(
        instruction
        for instruction in _co_names_and_consts(compiled)
        if isinstance(instruction, str)
    )

    for fragment in (
        "torch_tensorrt_executorch_runtime",
        "BACKEND_NAME",
        "Runtime",
        "is_available",
        "XnnpackBackend",
        "CudaBackend",
        "sys.exit",
    ):
        assert fragment in runtime_check, (
            f"the CI runtime check no longer contains {fragment!r}, so it no longer proves the "
            f"delegate loads and registers: {runtime_check}"
        )
        # The attribute chains appear in the code object split across names, so compare on the last
        # segment, which is the part a comment would remove.
        needle = fragment.rsplit(".", 1)[-1]
        assert needle in reachable, (
            f"the CI runtime check mentions {fragment!r} but the interpreter never reaches it, so "
            "the step would pass without loading the delegate. A '#' inside the one-liner comments "
            f"out the rest of the statement: {runtime_check}"
        )
    # assert would be compiled out under python -O, which is why the check uses sys.exit; guard
    # that reasoning too, so a rewrite back to assert is caught.
    assert "assert " not in runtime_check, (
        "the CI runtime check uses assert, which python -O compiles out, so a runtime with no "
        "backends registered would pass"
    )

    # The string has to actually run in a step, or asserting its content proves nothing. Parse the
    # workflow and require a script that both defines it and runs it as the invocation whose exit
    # status becomes the step's. A second, identical-looking call sits inside
    # `if [[ "${check_status}" -ne 0 ]]` as a gdb backtrace and ends in `|| true`, so it runs only
    # after the check has already failed and can never fail the job; a plain substring search is
    # satisfied by that decoy even when the real call is neutered. Anchor on the executing form:
    # the env-prefixed call that begins its line. The gdb copy begins with `--args python`, so it
    # does not match, and replacing the real call at line 88 with `true ||` turns this red.
    document = yaml.safe_load(workflow_text)
    scripts = [
        text
        for job in document["jobs"].values()
        if isinstance(job, dict)
        for text in (
            [str((job.get("with") or {}).get("script") or "")]
            + [
                str(step.get("run") or "")
                for step in job.get("steps") or []
                if isinstance(step, dict)
            ]
        )
    ]
    executing = [
        text
        for text in scripts
        if "runtime_check='" in text
        and re.search(
            r'^\s*PYTHONFAULTHANDLER=1 python -u -X faulthandler -c "\$\{runtime_check\}"',
            text,
            re.MULTILINE,
        )
    ]
    assert executing, (
        "no workflow script both defines runtime_check and runs it as the status-bearing "
        "invocation, so the delegate load check does not execute in CI"
    )


@pytest.mark.unit
def test_runtime_wheel_pins_its_native_dependencies_per_cuda_major():
    """The wheel names the TensorRT distribution matching the CUDA it was built against.

    Hardcoding tensorrt-cu13 was correct while only CUDA 13 shipped, and wrong once CUDA 12.6 came
    back: a cu126 row would then have declared a dependency on the CUDA 13 distribution. The value
    is resolved at build time instead, so the pairing follows whatever row is building.
    """
    setup_source = _RUNTIME_SETUP_PY.read_text(encoding="utf-8")
    assert "TENSORRT_DISTRIBUTION = tensorrt_distribution()" in setup_source, (
        "the TensorRT distribution is no longer resolved from the build's own CUDA version, so a "
        "CUDA 12.6 row would declare the wrong dependency"
    )
    assert (
        '"tensorrt-cu12"' in setup_source and '"tensorrt-cu13"' in setup_source
    ), "both CUDA channels must be mappable, since the matrix builds cu126 alongside cu130/cu132"
    assert 'CUDA_RUNTIME_DISTRIBUTION = "nvidia-cuda-runtime"' in setup_source
    assert "torch=={public_version(torch.__version__)}" in setup_source
    assert "{TENSORRT_DISTRIBUTION}=={public_version(tensorrt_version)}" in setup_source
    assert (
        "{CUDA_RUNTIME_DISTRIBUTION}=={public_version(cuda_runtime_version)}"
        in setup_source
    )
    assert "nvidia-cuda-runtime-cu12" not in setup_source


@pytest.mark.unit
def test_packaging_declares_executorch_extra():
    tree = _setup_tree()
    extras = _assignment_value(tree, "EXTRAS_REQUIRE")
    assert isinstance(extras, ast.Dict)

    extras_by_name = {
        key.value: value
        for key, value in zip(extras.keys, extras.values)
        if isinstance(key, ast.Constant)
    }
    for extra_name in ("executorch", "all"):
        assert extra_name in extras_by_name
        requirements = extras_by_name[extra_name]
        assert isinstance(requirements, ast.List)
        assert len(requirements.elts) == 1
        assert isinstance(requirements.elts[0], ast.Name)
        assert requirements.elts[0].id == "EXECUTORCH_REQUIREMENT"

    setup_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    )
    extras_keyword = next(
        (keyword for keyword in setup_call.keywords if keyword.arg == "extras_require"),
        None,
    )
    assert extras_keyword is not None
    assert isinstance(extras_keyword.value, ast.Name)
    assert extras_keyword.value.id == "EXTRAS_REQUIRE"


@pytest.mark.unit
def test_executorch_is_not_base_install_requirement():
    tree = _setup_tree()
    for function_name in (
        "get_jetpack_requirements",
        "get_sbsa_requirements",
        "get_x86_64_requirements",
        "get_requirements",
    ):
        function = _function_def(tree, function_name)
        assert not any(
            isinstance(node, ast.Name) and node.id == "EXECUTORCH_REQUIREMENT"
            for node in ast.walk(function)
        )


@pytest.mark.unit
def test_main_wheel_does_not_package_executorch_delegate_headers():
    assert "include/torch_tensorrt/executorch/*.h" not in _SETUP_PY.read_text(
        encoding="utf-8"
    )


def _stub_node(name, target=None):
    return types.SimpleNamespace(name=name, target=name if target is None else target)


def _stub_exported_program(constants, name_to_fqn=None):
    sig = (
        None
        if name_to_fqn is None
        else types.SimpleNamespace(inputs_to_lifted_custom_objs=name_to_fqn)
    )
    return types.SimpleNamespace(constants=constants, graph_signature=sig)


@pytest.mark.unit
def test_resolve_lifted_custom_obj_via_signature_fqn():
    # Modern torch.export: placeholder name differs from the constants FQN key.
    sentinel = object()
    ep = _stub_exported_program({"engine_fqn": sentinel}, {"obj_engine": "engine_fqn"})
    assert _resolve_lifted_custom_obj(ep, _stub_node("obj_engine")) is sentinel


@pytest.mark.unit
def test_resolve_lifted_custom_obj_legacy_fallback():
    # No signature mapping: fall back to a direct name/target lookup.
    sentinel = object()
    ep = _stub_exported_program({"engine": sentinel}, name_to_fqn=None)
    assert _resolve_lifted_custom_obj(ep, _stub_node("engine")) is sentinel


@pytest.mark.unit
def test_resolve_lifted_custom_obj_signature_present_name_absent_is_none():
    # A present-but-incomplete mapping must not bind a different object by name.
    ep = _stub_exported_program({"engine": object()}, {"some_other_obj": "x"})
    assert _resolve_lifted_custom_obj(ep, _stub_node("engine")) is None


@pytest.mark.unit
def test_resolve_lifted_custom_obj_missing_is_none():
    ep = _stub_exported_program({}, name_to_fqn=None)
    assert _resolve_lifted_custom_obj(ep, _stub_node("missing")) is None


@pytest.mark.unit
def test_resolve_lifted_custom_obj_unwraps_fake_script_object():
    class _Real:
        pass

    fake = FakeScriptObject(object(), "Engine", _Real())
    ep = _stub_exported_program({"engine_fqn": fake}, {"obj_engine": "engine_fqn"})
    resolved = _resolve_lifted_custom_obj(ep, _stub_node("obj_engine"))
    assert not isinstance(resolved, FakeScriptObject)
    assert isinstance(resolved, _Real)


# --- per-partition target_device (TensorRTPartitioner) -----------------------
# These exercise the partitioner directly, so they need ExecuTorch installed;
# they run in the dedicated executorch CI job and skip elsewhere.


@pytest.mark.unit
def test_resolve_target_device_uses_partition_engine(monkeypatch):
    """The device comes from the partition's own engine node, read as metadata only.

    DEVICE_IDX is the only slot wanted, and this runs once per partition on every
    export that does not pin ``target_device`` -- the default -- so reading the full
    record would re-serialize each engine to look at one string.
    """
    pytest.importorskip("executorch.exir")
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import DEVICE_IDX
    from torch_tensorrt.executorch import partitioner as P

    part = P.TensorRTPartitioner()
    engine_node = object()
    monkeypatch.setattr(P, "_get_engine_nodes_in", lambda nodes: [engine_node])
    info = ["0"] * (DEVICE_IDX + 1)
    info[DEVICE_IDX] = "2"
    calls = []

    def _spy(ep, n, **kwargs):
        calls.append(kwargs)
        return info

    monkeypatch.setattr(P, "_get_engine_info_for_node", _spy)

    partition = types.SimpleNamespace(id=0, nodes=[engine_node])
    assert part._resolve_target_device_for_partition(object(), partition) == b"cuda:2"
    assert calls == [{"metadata_only": True}]


@pytest.mark.unit
def test_resolve_target_device_falls_back_when_not_one_engine(monkeypatch):
    pytest.importorskip("executorch.exir")
    from torch_tensorrt.executorch import partitioner as P

    part = P.TensorRTPartitioner()
    partition = types.SimpleNamespace(id=1, nodes=[])

    monkeypatch.setattr(P, "_get_engine_nodes_in", lambda nodes: [])
    assert part._resolve_target_device_for_partition(object(), partition) == b"cuda:0"

    monkeypatch.setattr(P, "_get_engine_nodes_in", lambda nodes: [object(), object()])
    assert part._resolve_target_device_for_partition(object(), partition) == b"cuda:0"


@pytest.mark.unit
def test_per_partition_distinct_target_devices(monkeypatch):
    pytest.importorskip("executorch.exir")
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import DEVICE_IDX
    from torch_tensorrt.executorch import partitioner as P

    part = P.TensorRTPartitioner()
    # Each partition's engine node carries its own device id as its value.
    monkeypatch.setattr(P, "_get_engine_nodes_in", lambda nodes: [nodes[0]])

    def fake_info(ep, node, **kwargs):
        info = ["0"] * (DEVICE_IDX + 1)
        info[DEVICE_IDX] = str(node)
        return info

    monkeypatch.setattr(P, "_get_engine_info_for_node", fake_info)
    d0 = part._resolve_target_device_for_partition(
        object(), types.SimpleNamespace(id=0, nodes=["0"])
    )
    d1 = part._resolve_target_device_for_partition(
        object(), types.SimpleNamespace(id=1, nodes=["1"])
    )
    assert d0 == b"cuda:0"
    assert d1 == b"cuda:1"
    assert d0 != d1


# --- lift() preserves constant dtype/device ----------------------------------
# lift() replaces get_attr constants with placeholders and copies a fake meta
# "val". It must fakify the source constant through the graph's own fake_mode
# (fake_mode.from_tensor), preserving dtype, device and stride; otherwise a
# non-fp32 or non-CPU lifted constant silently gets fp32/cpu meta.


def _traced_gm_with_parameter(dtype, device, requires_grad=False):
    """A symbolically-traced GraphModule with one get_attr parameter (`c`) of the
    given dtype/device, plus a stub graph_signature lift() can mutate."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c = torch.nn.Parameter(
                torch.zeros(3, 3, dtype=dtype, device=device),
                requires_grad=requires_grad,
            )

        def forward(self, x):
            return x + self.c

    gm = torch.fx.symbolic_trace(M())
    # lift() calls detect_fake_mode over the placeholder "val" metas, so the user
    # input needs a FakeTensor meta.
    fake_mode = FakeTensorMode()
    with fake_mode:
        fake_x = torch.empty(3, 3)
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            node.meta["val"] = fake_x
    sig = types.SimpleNamespace(input_specs=[], output_specs=[], user_inputs=["x"])
    return gm, sig


def _lifted_constant_meta(gm, sig):
    lifted_gm, _, _, _ = lift(gm, sig)
    lifted = [
        n for n in lifted_gm.graph.nodes if n.op == "placeholder" and n.name != "x"
    ]
    assert len(lifted) == 1, f"expected 1 lifted constant, got {len(lifted)}"
    return lifted[0].meta["val"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype, device",
    [
        (torch.bfloat16, "cpu"),
        (torch.float32, "cuda"),
    ],
)
def test_lift_preserves_constant_dtype_device(dtype, device):
    # Runtime gate (not a module-level skipif, which resolves at collection time
    # and is fragile on remote-GPU runners): skip the CUDA case only when no GPU.
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    gm, sig = _traced_gm_with_parameter(dtype, device)
    val = _lifted_constant_meta(gm, sig)
    assert isinstance(val, FakeTensor)
    assert val.dtype == dtype
    assert val.device.type == device
    # The source constant is a contiguous 3x3, so from_tensor must carry its real
    # (3, 1) stride onto the meta, not the old all-ones synthetic stride.
    assert val.stride() == torch.zeros(3, 3).stride()


# --- lift() preserves parameter kind and requires_grad -----------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "dtype, requires_grad",
    [
        (torch.uint8, False),
        (torch.int8, False),
        (torch.float32, False),
        # A trainable weight, so hardcoding either flag value fails the test.
        (torch.float32, True),
    ],
)
def test_lift_keeps_parameter_for_any_dtype(dtype, requires_grad):
    gm, sig = _traced_gm_with_parameter(dtype, "cpu", requires_grad)
    _, graph_signature, state_dict, constants = lift(gm, sig)

    kinds = [spec.kind for spec in graph_signature.input_specs]
    assert InputKind.PARAMETER in kinds, f"expected a PARAMETER, got {kinds}"

    # The weight belongs in the state dict as a parameter, not in constants, or a
    # caller can no longer load a checkpoint into the exported module.
    assert "c" in state_dict, f"weight missing from state_dict: {list(state_dict)}"
    assert isinstance(state_dict["c"], torch.nn.Parameter)
    assert state_dict["c"].dtype == dtype
    assert "c" not in constants

    assert state_dict["c"].requires_grad is requires_grad


# --- save(output_format="executorch") forwards ExecuTorch lowering kwargs -------
# After #4440, _save_as_executorch no longer runs the lowering itself: it delegates to
# torch_tensorrt.executorch.export() (which owns the TRT graph surgery + lowering and
# defaults compile_config to get_edge_compile_config()) and then calls
# edge_program.to_executorch(config=backend_config). save() must forward each of
# transform_passes / constant_methods / compile_config / generate_etrecord /
# partitioners / compile_specs into export(), let export() apply
# get_edge_compile_config() when compile_config is omitted (_check_ir_validity=False,
# since the TRT execute_engine placeholder graph fails edge IR validation), and route
# backend_config to to_executorch(). generate_etrecord persists a "<base>_etrecord.bin"
# next to the .pte.


def _patch_executorch_lowering(monkeypatch, captured):
    """Spy the seams _save_as_executorch delegates to after #4440, without running a
    real TRT lowering. Records the kwargs forwarded into
    torch_tensorrt.executorch.export() (captured["export_kwargs"]); lets the real
    export() run against an engine-free program while stubbing only the innermost
    ExecuTorch lowering, so the kwargs export() finally hands that lowering --
    including the get_edge_compile_config() it substitutes for an omitted
    compile_config -- land in `captured`; and records backend_config from
    to_executorch(). Fills `captured`; returns nothing."""
    import executorch.exir as exir
    import torch_tensorrt._compile as tc
    import torch_tensorrt.executorch as tte

    class _FakeETRecord:
        def save(self, path):
            with open(path, "wb") as fh:
                fh.write(b"etrecord")

    class _FakeExec:
        def write_to_file(self, f):
            f.write(b"")

        def get_etrecord(self):
            return _FakeETRecord()

    class _FakeEdge:
        def to_executorch(self, config=None):
            captured["backend_config"] = config
            return _FakeExec()

    # Innermost real lowering seam, called inside export(): capture the kwargs it
    # receives, notably compile_config already resolved to its default when omitted.
    def _fake_lower(exp_program, **kw):
        captured.update(kw)
        return _FakeEdge()

    monkeypatch.setattr(exir, "to_edge_transform_and_lower", _fake_lower)

    # Record what _save_as_executorch forwards into export(), then run the real
    # export() so its default-application and graph staging still execute.
    real_export = tte.export

    def _spy_export(source, **kw):
        captured["export_kwargs"] = kw
        return real_export(source, **kw)

    monkeypatch.setattr(tte, "export", _spy_export)
    monkeypatch.setattr(tc, "_write_external_tensor_data", lambda prog, path: None)
    # ENABLED_FEATURES is an immutable namedtuple; swap the whole module attribute.
    monkeypatch.setattr(
        tc, "ENABLED_FEATURES", types.SimpleNamespace(torch_tensorrt_runtime=True)
    )


@pytest.mark.unit
def test_save_executorch_forwards_lowering_kwargs(monkeypatch, tmp_path):
    pytest.importorskip("executorch.exir")
    import torch_tensorrt._compile as tc
    from executorch.exir import EdgeCompileConfig

    captured = {}
    _patch_executorch_lowering(monkeypatch, captured)

    sentinel_passes = [object()]
    sentinel_methods = {"get_max_seq_len": 128}
    caller_cfg = EdgeCompileConfig(_check_ir_validity=True)
    out = str(tmp_path / "model.pte")

    ep = torch.export.export(_AddOne(), (torch.randn(2, 2),))
    tc._save_as_executorch(
        ep,
        out,
        partitioners=[],
        compile_specs=[],
        backend_config=None,
        constant_methods=sentinel_methods,
        transform_passes=sentinel_passes,
        compile_config=caller_cfg,
        generate_etrecord=True,
    )

    # Every ExecuTorch lowering kwarg is forwarded verbatim into export().
    export_kwargs = captured["export_kwargs"]
    assert export_kwargs["transform_passes"] is sentinel_passes
    assert export_kwargs["constant_methods"] is sentinel_methods
    assert export_kwargs["compile_config"] is caller_cfg
    assert export_kwargs["generate_etrecord"] is True
    assert export_kwargs["partitioners"] == []
    assert export_kwargs["compile_specs"] == []
    # A caller-supplied compile_config reaches the lowering verbatim (explicit override
    # respected, not replaced with the default even though it sets
    # _check_ir_validity=True).
    assert captured["compile_config"] is caller_cfg
    assert captured["compile_config"]._check_ir_validity is True
    # backend_config flows to edge_program.to_executorch(config=...).
    assert captured["backend_config"] is None
    # ETRecord persisted next to the .pte per ET's "<base>_etrecord.bin" convention.
    assert (tmp_path / "model_etrecord.bin").exists()


@pytest.mark.unit
def test_save_executorch_defaults_when_lowering_kwargs_omitted(monkeypatch, tmp_path):
    pytest.importorskip("executorch.exir")
    import torch_tensorrt._compile as tc

    captured = {}
    _patch_executorch_lowering(monkeypatch, captured)

    out = str(tmp_path / "model.pte")
    ep = torch.export.export(_AddOne(), (torch.randn(2, 2),))
    tc._save_as_executorch(ep, out)

    # _save_as_executorch forwards None for every omitted lowering kwarg, delegating
    # the defaults to export().
    export_kwargs = captured["export_kwargs"]
    assert export_kwargs["compile_config"] is None
    assert export_kwargs["transform_passes"] is None
    assert export_kwargs["constant_methods"] is None
    assert export_kwargs["partitioners"] is None
    assert export_kwargs["compile_specs"] is None
    assert export_kwargs["generate_etrecord"] is False
    # export() substitutes get_edge_compile_config() for the omitted compile_config
    # (_check_ir_validity=False) and that config reaches the lowering.
    assert captured["compile_config"]._check_ir_validity is False
    assert captured["transform_passes"] is None
    assert captured["generate_etrecord"] is False
    # No etrecord written when generate_etrecord is falsy.
    assert not (tmp_path / "model_etrecord.bin").exists()


# --- the same lowering kwargs flow through the *public* torch_tensorrt.save() -----
# save() pops the ExecuTorch-only kwargs and forwards them to _save_as_executorch
# from three dispatch branches: an ExportedProgram input, a GraphModule with
# retrace=True (re-exported here), and a GraphModule with retrace=False (routed
# through the dynamo exporter). Each must extract the options and forward them.


class _AddOne(torch.nn.Module):
    def forward(self, x):
        return x + 1


def _stub_save_as_executorch(monkeypatch):
    """Capture the (module, file_path, kwargs) that save() forwards to
    _save_as_executorch without running the real lowering."""
    import torch_tensorrt._compile as tc

    calls = []

    def _fake(module, file_path, **kw):
        calls.append({"module": module, "file_path": file_path, "kwargs": kw})

    monkeypatch.setattr(tc, "_save_as_executorch", _fake)
    return calls


def _assert_lowering_kwargs_forwarded(kw, methods, passes, cfg):
    assert kw["constant_methods"] is methods
    assert kw["transform_passes"] is passes
    assert kw["compile_config"] is cfg
    assert kw["generate_etrecord"] is True
    assert kw["partitioners"] == []
    assert kw["compile_specs"] == []
    assert kw["backend_config"] is None


@pytest.mark.unit
def test_public_save_forwards_lowering_kwargs_exported_program(monkeypatch, tmp_path):
    pytest.importorskip("executorch.exir")
    import torch_tensorrt
    from executorch.exir import EdgeCompileConfig

    calls = _stub_save_as_executorch(monkeypatch)
    methods = {"get_max_seq_len": 128}
    passes = [object()]
    cfg = EdgeCompileConfig(_check_ir_validity=False)
    out = str(tmp_path / "ep.pte")

    ep = torch.export.export(_AddOne(), (torch.randn(2, 2),))
    torch_tensorrt.save(
        ep,
        out,
        output_format="executorch",
        partitioners=[],
        compile_specs=[],
        backend_config=None,
        constant_methods=methods,
        transform_passes=passes,
        compile_config=cfg,
        generate_etrecord=True,
    )

    assert len(calls) == 1
    assert calls[0]["module"] is ep
    _assert_lowering_kwargs_forwarded(calls[0]["kwargs"], methods, passes, cfg)


@pytest.mark.unit
def test_public_save_forwards_lowering_kwargs_graphmodule_retrace(
    monkeypatch, tmp_path
):
    pytest.importorskip("executorch.exir")
    import torch_tensorrt
    from executorch.exir import EdgeCompileConfig

    calls = _stub_save_as_executorch(monkeypatch)
    methods = {"get_max_seq_len": 128}
    passes = [object()]
    cfg = EdgeCompileConfig(_check_ir_validity=False)
    out = str(tmp_path / "gm_retrace.pte")

    gm = torch.export.export(_AddOne(), (torch.randn(2, 2),)).module()
    torch_tensorrt.save(
        gm,
        out,
        output_format="executorch",
        retrace=True,
        arg_inputs=(torch.randn(2, 2),),
        partitioners=[],
        compile_specs=[],
        backend_config=None,
        constant_methods=methods,
        transform_passes=passes,
        compile_config=cfg,
        generate_etrecord=True,
    )

    assert len(calls) == 1
    _assert_lowering_kwargs_forwarded(calls[0]["kwargs"], methods, passes, cfg)


@pytest.mark.unit
def test_public_save_forwards_lowering_kwargs_graphmodule_no_retrace(
    monkeypatch, tmp_path
):
    pytest.importorskip("executorch.exir")
    import torch_tensorrt
    import torch_tensorrt.dynamo._exporter as _exporter
    from executorch.exir import EdgeCompileConfig

    calls = _stub_save_as_executorch(monkeypatch)
    # retrace=False routes through the dynamo exporter (TRT-specific graph surgery);
    # stub it so the test isolates save()'s option extraction + forwarding. The stub
    # returns a real ExportedProgram because save() runs
    # _declare_aliased_kv_mutations_on_ep over the exporter's result before forwarding
    # it, and that pass reads .graph_module / .graph_signature. This program has no
    # engine nodes, so the pass returns it unchanged and the identity assertion below
    # still pins exactly what the exporter produced.
    stub_ep = torch.export.export(_AddOne(), (torch.randn(2, 2),))
    monkeypatch.setattr(_exporter, "export", lambda *a, **k: stub_ep)

    methods = {"get_max_seq_len": 128}
    passes = [object()]
    cfg = EdgeCompileConfig(_check_ir_validity=False)
    out = str(tmp_path / "gm_no_retrace.pte")

    gm = torch.export.export(_AddOne(), (torch.randn(2, 2),)).module()
    torch_tensorrt.save(
        gm,
        out,
        output_format="executorch",
        retrace=False,
        partitioners=[],
        compile_specs=[],
        backend_config=None,
        constant_methods=methods,
        transform_passes=passes,
        compile_config=cfg,
        generate_etrecord=True,
    )

    assert len(calls) == 1
    assert calls[0]["module"] is stub_ep
    _assert_lowering_kwargs_forwarded(calls[0]["kwargs"], methods, passes, cfg)


@pytest.mark.unit
def test_save_executorch_real_etrecord_is_inspector_consumable(tmp_path):
    """A real lowering with generate_etrecord=True writes a sidecar that ExecuTorch's
    devtools can parse back into an Inspector-consumable ETRecord."""
    pytest.importorskip("executorch.exir")
    pytest.importorskip("executorch.devtools")
    import torch_tensorrt
    from executorch.devtools.etrecord import parse_etrecord
    from torch_tensorrt._features import ENABLED_FEATURES

    if not ENABLED_FEATURES.torch_tensorrt_runtime:
        pytest.skip("output_format='executorch' requires the torch_tensorrt runtime")

    ep = torch.export.export(_AddOne(), (torch.randn(4, 4),))
    out = str(tmp_path / "tiny.pte")
    torch_tensorrt.save(ep, out, output_format="executorch", generate_etrecord=True)

    etrecord_path = tmp_path / "tiny_etrecord.bin"
    assert etrecord_path.exists()

    record = parse_etrecord(str(etrecord_path))
    assert record is not None
    # The parsed record carries the edge-dialect program the Inspector correlates
    # runtime events against.
    assert getattr(record, "edge_dialect_program", None) is not None


@pytest.mark.unit
@pytest.mark.parametrize(
    "case,should_pass",
    [
        ("well_formed", True),
        ("manylinux_tag", True),
        ("bundles_a_stowaway", False),
        ("bundles_the_executorch_runtime", False),
        ("bundles_the_executorch_runtime_under_a_non_so_name", False),
        ("payload_carries_a_mangled_name", False),
        ("ships_no_cmake_package", False),
        ("declares_itself_pure_python", False),
        ("platform_independent_tag", False),
        ("windows_compound_tag", False),
        ("alien_architecture_tag", False),
        ("requires_an_unpinned_executorch", False),
        ("requires_no_executorch", False),
        ("requires_a_mismatched_executorch_pin", False),
        ("requires_no_torch_tensorrt", False),
        ("requires_no_torch", False),
        ("requires_no_tensorrt", False),
        ("requires_an_unpinned_torch_tensorrt", False),
        ("requires_an_unpinned_cuda_runtime", False),
        ("requirement_carries_a_local_label", False),
        ("no_metadata_at_all", False),
    ],
)
def test_the_wheel_checker_rejects_a_bad_wheel(tmp_path, case, should_pass):
    """The wheel checker has to reject, not just pass on the artifact of the day.

    It is the only thing enforcing wheel contents, platform tag, purelib and Requires-Dist, and
    the first time it rejected anything it rejected this project's own wheel -- a local version
    label on one requirement -- with no test having exercised a rejecting case. Both halves of
    that need a fixture: the label case, and the pin mismatch that motivates reading METADATA at
    all, since setup.py derives every dependency from whatever happens to be installed. That same
    derivation applies to tensorrt-cu13 and nvidia-cuda-runtime, so a wheel that drops one or
    loosens it to a range is exercised too.

    The checker is lifted from the workflow rather than restated, so a change there cannot leave
    this test asserting against a copy that no longer exists.
    """
    workflow = (_REPO_ROOT / ".github/workflows/executorch-build-linux.yml").read_text(
        encoding="utf-8"
    )
    body = re.search(
        r"python - \"\$\(ls dist/torch_tensorrt_executorch_runtime-\*\.whl\)\" <<'PY'\n(.*?)\n        PY\n",
        workflow,
        re.DOTALL,
    )
    assert body, "the wheel checker is no longer identifiable in the workflow"

    # Lifting the body proves the rules work; it cannot see the invocation being made
    # unreachable. Parse the document and check the script with "bash -n", never "bash -c": the
    # prologue downloads bazelisk and pip installs ExecuTorch, so executing it is not an option.
    import yaml

    document = yaml.safe_load(workflow)
    marker = 'python - "$(ls dist/torch_tensorrt_executorch_runtime-*.whl)"'
    scripts = [
        text
        for job in document["jobs"].values()
        if isinstance(job, dict)
        for text in (
            [str((job.get("with") or {}).get("script") or "")]
            + [
                str(step.get("run") or "")
                for step in job.get("steps") or []
                if isinstance(step, dict)
            ]
        )
        if marker in text
    ]
    assert scripts, "no job script carries the wheel checker, so nothing runs it"
    # Every script that carries it, not just the first. PyYAML preserves document order, so taking
    # scripts[0] meant a decoy job declared earlier in the file was checked while the real
    # invocation went unexamined.
    for script in scripts:
        prologue = script[: script.index(marker)]
        _assert_the_checker_is_reachable(prologue)

    script = scripts[0]
    prologue = script[: script.index(marker)]
    checker = tmp_path / "checker.py"
    checker.write_text(
        "\n".join(
            line[8:] if line.startswith(" " * 8) else line
            for line in body.group(1).splitlines()
        ),
        encoding="utf-8",
    )

    pin = re.search(
        r'^__executorch_version__:\s*"?([^"\s]+)"?\s*$',
        (_REPO_ROOT / "dev_dep_versions.yml").read_text(encoding="utf-8"),
        re.MULTILINE,
    ).group(1)
    package = "torch_tensorrt_executorch_runtime/"
    payload = [package + "lib/libexecutorch_backend_tensorrt.so"]
    requires = [
        f"executorch=={pin}",
        "torch==2.15.0.dev20260824",
        "torch-tensorrt==2.15.0.dev20260824",
        "tensorrt-cu13==11.2.1",
        "nvidia-cuda-runtime==13.2.0",
    ]
    purelib, tag = "false", "linux_x86_64"

    if case == "manylinux_tag":
        tag = "manylinux_2_28_x86_64"
    elif case == "bundles_a_stowaway":
        payload.append(package + "libnvinfer.so.10")
    elif case == "bundles_the_executorch_runtime":
        payload.append(package + "libexecutorch.so")
    elif case == "bundles_the_executorch_runtime_under_a_non_so_name":
        # The count check only matches names ending in .so or .so.<n>, so an ExecuTorch component
        # shipped under any other name slips past it. Only the forbidden-component list, which
        # matches every name in the archive, catches this, so deleting that list opens a real hole.
        payload.append(package + "executorch/lib/libexecutorch.so.debug")
    elif case == "payload_carries_a_mangled_name":
        # Exactly one object, but under the setuptools-mangled name the build_py redesign exists to
        # prevent. The count check passes on it, so only the exact-name branch can reject it: with
        # that branch gone the wheel ships a delegate pip cannot import under the expected name.
        payload = [
            package + "_executorch_backend_tensorrt.cpython-310-x86_64-linux-gnu.so"
        ]
    elif case == "declares_itself_pure_python":
        purelib = "true"
    elif case == "platform_independent_tag":
        tag = "any"
    elif case == "windows_compound_tag":
        # The tag is split on "." and each part must match the linux-arch pattern alone, so a
        # compound tag carrying a win_amd64 part is rejected even though a plain substring test
        # would accept it for the "linux_x86_64" part beside it.
        tag = "win_amd64.linux_x86_64"
    elif case == "alien_architecture_tag":
        # The architecture allowlist is the other half of the split-and-match rule: a linux tag for
        # an architecture the wheel is not built for has to be rejected, not just non-linux tags.
        tag = "linux_ppc64le"
    elif case == "requires_an_unpinned_executorch":
        requires = [f"executorch=={pin.split('.')[0]}.0.0"]
    elif case == "requires_no_executorch":
        requires = ["torch==2.15.0.dev20260824"]
    elif case == "requires_a_mismatched_executorch_pin":
        # Every requirement present and exactly pinned, but executorch names a different version
        # than the repository. The presence loop is satisfied, so only the pin comparison can
        # reject it: with that branch gone the wheel ships requiring an executorch it was not built
        # against and every other check still passes. The wrong version is derived from the pin
        # rather than written as a literal so the repository-wide "==<digit>" pin scan does not read
        # this fixture as a real, mispinned requirement site.
        wrong_pin = pin.rsplit(".dev", 1)[0] + ".dev20200101"
        requires = [
            f"executorch=={wrong_pin}",
            "torch==2.15.0.dev20260824",
            "torch-tensorrt==2.15.0.dev20260824",
            "tensorrt-cu13==11.2.1",
            "nvidia-cuda-runtime==13.2.0",
        ]
    elif case == "requires_no_torch_tensorrt":
        # setup.py derives torch-tensorrt the same way it derives executorch, and it is the
        # requirement that binds this runtime wheel to the producer that emitted the program, so a
        # wheel that drops it ships with that binding missing and every content check still passes.
        requires = [
            f"executorch=={pin}",
            "torch==2.15.0.dev20260824",
            "tensorrt-cu13==11.2.1",
            "nvidia-cuda-runtime==13.2.0",
        ]
    elif case == "requires_no_torch":
        requires = [
            f"executorch=={pin}",
            "torch-tensorrt==2.15.0.dev20260824",
            "tensorrt-cu13==11.2.1",
            "nvidia-cuda-runtime==13.2.0",
        ]
    elif case == "requires_no_tensorrt":
        # setup.py derives tensorrt-cu13 the same way it derives executorch, so a wheel that drops
        # it ships with the dependency missing and every content check above still passes.
        requires = [
            f"executorch=={pin}",
            "torch==2.15.0.dev20260824",
            "torch-tensorrt==2.15.0.dev20260824",
            "nvidia-cuda-runtime==13.2.0",
        ]
    elif case == "requires_an_unpinned_torch_tensorrt":
        # A derived requirement loosened to a range no longer binds the wheel to the exact producer
        # it was built beside, which is the whole reason the metadata is read.
        requires = [
            f"executorch=={pin}",
            "torch==2.15.0.dev20260824",
            "torch-tensorrt>=2.15.0.dev20260824",
            "tensorrt-cu13==11.2.1",
            "nvidia-cuda-runtime==13.2.0",
        ]
    elif case == "requires_an_unpinned_cuda_runtime":
        # A derived requirement loosened to a range no longer binds the wheel to the version it was
        # built beside, which is the whole reason the metadata is read.
        requires = [
            f"executorch=={pin}",
            "torch==2.15.0.dev20260824",
            "torch-tensorrt==2.15.0.dev20260824",
            "tensorrt-cu13==11.2.1",
            "nvidia-cuda-runtime>=13.2.0",
        ]
    elif case == "requirement_carries_a_local_label":
        # The exact rejection this PR's own CI hit: binds the wheel to one CUDA train. Relabel the
        # torch-tensorrt entry in place rather than appending a duplicate requirement.
        requires = [
            r + "+cu130" if r.startswith("torch-tensorrt==") else r for r in requires
        ]

    wheel = tmp_path / f"torch_tensorrt_executorch_runtime-1.0-cp310-cp310-{tag}.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in payload:
            archive.writestr(name, b"\x7fELF")
        # The CMake package a C++ consumer links through. Present in every case except the one that
        # deliberately drops it, so the other cases fail for their own reason rather than this one.
        if case != "ships_no_cmake_package":
            for cmake_name in (
                "torchtrt_executorch-config.cmake",
                "torchtrt_executorch-config-version.cmake",
            ):
                archive.writestr(
                    f"{package}lib/cmake/torchtrt_executorch/{cmake_name}", "# stub\n"
                )
        info = "torch_tensorrt_executorch_runtime-1.0.dist-info"
        archive.writestr(f"{info}/WHEEL", f"Root-Is-Purelib: {purelib}\n")
        if case != "no_metadata_at_all":
            archive.writestr(
                f"{info}/METADATA",
                "Metadata-Version: 2.1\nName: torch-tensorrt-executorch-runtime\n"
                + "".join(f"Requires-Dist: {r}\n" for r in requires),
            )

    completed = subprocess.run(
        [sys.executable, str(checker), str(wheel)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    accepted = completed.returncode == 0
    assert accepted is should_pass, (
        f"{case}: checker exited {completed.returncode}, expected "
        f"{'acceptance' if should_pass else 'rejection'}\n{completed.stdout}{completed.stderr}"
    )
    # Named branches whose case must reject through that branch and no other. Exit-status alone let
    # a case pass by any route that also rejects: the mangled-name and pin-mismatch payloads each
    # survived their own branch being deleted because an earlier check rejected a sibling payload
    # that also dropped other requirements. Requiring the branch's own message pins each to it.
    expected_messages = {
        "payload_carries_a_mangled_name": (
            "expected torch_tensorrt_executorch_runtime/lib/libexecutorch_backend_tensorrt.so"
        ),
        "ships_no_cmake_package": "the wheel ships no CMake package",
        "requires_a_mismatched_executorch_pin": "the repository pins executorch==",
    }
    expected_message = expected_messages.get(case)
    if expected_message is not None:
        assert expected_message in completed.stderr, (
            f"{case} was rejected, but not through its own branch: expected "
            f"{expected_message!r} in\n{completed.stderr}"
        )


@pytest.mark.unit
def test_the_wheel_build_resolves_the_delegate_from_its_installed_location():
    """Something has to ask the loader, not just inspect the ELF headers.

    The link-time guard compares the whole RUNPATH against what the build asked for and checks one
    ExecuTorch symbol, but it cannot resolve anything: a Bazel output tree has no sibling
    site-packages to load against, so it reasons about the artifact's metadata. Measured: a pin
    bump that drops some other ExecuTorch export the delegate imports reaches an undefined symbol
    at import time that the link-time guard cannot see. ``ldd -r`` in the installed layout rejects
    that and accepts the real artifact, so the wheel-build step runs it there.
    """
    workflow = (_REPO_ROOT / ".github/workflows/executorch-build-linux.yml").read_text(
        encoding="utf-8"
    )
    # Strip whole-line shell comments before matching run-step content. Commenting out a step is
    # how it actually gets disabled, and the raw text would let a commented-out `ldd -r`, `exit 1`
    # or the import line, or a decoy comment naming "undefined symbol" beside a narrowed grep,
    # satisfy these checks while the step no longer runs. Line count is preserved so the block
    # regex below still spans structurally.
    workflow = "\n".join(
        "" if line.lstrip().startswith("#") else line for line in workflow.splitlines()
    )

    # Bound the match to the if-block, from its `if ... ldd -r` to the closing `fi`. A scan that ran
    # to the first `exit 1` anywhere below instead swallowed ~20 lines and matched an unrelated
    # `exit 1` in a later loop, so deleting this block's own `exit 1` left the test green while a
    # FATAL over an unresolvable delegate no longer failed the step. The command runs under
    # `env -u LD_LIBRARY_PATH` so the CUDA directory the test lane exports cannot resolve a missing
    # RUNPATH entry that a user's process would not have.
    resolves = re.search(
        r"^([ \t]*)if env -u LD_LIBRARY_PATH ldd -r [^\n]*\n(?:[^\n]*\n)*?\1fi\b",
        workflow,
        re.MULTILINE,
    )
    assert resolves, (
        "no step resolves the installed delegate with `env -u LD_LIBRARY_PATH ldd -r`, so a "
        "missing RUNPATH entry or an ExecuTorch symbol the pin no longer exports would ship"
    )
    assert re.search(r"^\s*exit 1\n", resolves.group(0), re.MULTILINE), (
        "the ldd -r block does not exit non-zero on unresolved symbols, so under set -e a FATAL "
        "still passes the step"
    )
    assert "not found" in resolves.group(0) and "undefined symbol" in resolves.group(
        0
    ), "the resolution check ignores one of the two failure kinds it exists to catch"
    # It has to run against the installed wheel, not the build tree, or the siblings are absent
    # and every dependency is "not found".
    assert re.search(
        r"pip install[^\n]*dist/torch_tensorrt_executorch_runtime-\*\.whl", workflow
    ), "the delegate is not installed before being resolved, so the check cannot pass"
    assert re.search(
        r'python -c "import torch_tensorrt_executorch_runtime"', workflow
    ), (
        "nothing imports the package after installing it, which is the check a user's first "
        "import performs, and importing it IS the registration"
    )


@pytest.mark.unit
def test_the_install_script_puts_a_cuda_runtime_on_the_library_path_for_both_majors():
    """Every CUDA row gets a CUDA runtime directory, not just the CUDA 13 ones.

    The two majors ship different layouts: nvidia-cuda-runtime-cu12 installs
    nvidia/cuda_runtime/lib/libcudart.so.12 while the CUDA 13 line installs
    nvidia/cu13/lib/libcudart.so.13. Naming only the cu13 path left every CUDA 12 row with no
    CUDA runtime on the search path, and the ExecuTorch reference runner died at startup with
    "libcudart.so.12: cannot open shared object file" while the package was installed all along.
    The runtime wheel's own RPATH already lists both directories; this keeps the CI install
    script consistent with it.
    """
    script = (_REPO_ROOT / ".github/scripts/install-torch-tensorrt.sh").read_text(
        encoding="utf-8"
    )
    for directory in ("nvidia/cuda_runtime/lib", "nvidia/cu13/lib"):
        assert directory in script, (
            f"{directory} is not on LD_LIBRARY_PATH, so a row of that CUDA major cannot resolve "
            "libcudart at run time"
        )
    # Matched on the major, so a future cu128 or cu134 row is covered without another edit.
    for pattern in ("cu12*)", "cu13*)"):
        assert (
            pattern in script
        ), f"{pattern} is gone, so a CUDA row of that major would fall through to no directory"
