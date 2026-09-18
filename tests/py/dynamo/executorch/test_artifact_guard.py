# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only behavior checks for the companion's native artifact guard."""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[4]
_NATIVE = _ROOT / "py/torch-tensorrt-executorch-runtime/native"
_GUARD = _NATIVE / "check_imports_executorch_runtime.sh"
_RUNPATH = (
    "$ORIGIN:$ORIGIN/../../executorch/lib:$ORIGIN/../../tensorrt_libs:"
    "$ORIGIN/../../nvidia/cu13/lib"
)
_REGISTER = "_ZN10executorch7runtime16register_backendERKNS0_7BackendE"
_X86 = "manylinux_2_28_x86_64"
_ARM = "manylinux_2_35_aarch64"
_BASE_VERSIONS = "CXXABI_1.3 GLIBCXX_3.4.21 GLIBC_2.17 GCC_3.0"
# What the tests that build for real need, and which of those a Linux CI image already carries. The
# rest has to be installed by the job, or those tests skip and take the job's green with them.
_NATIVE_TOOLS = ("cmake", "c++", "readelf", "patchelf")
_IMAGE_TOOLS = ("cmake", "c++", "readelf")


def _run(argv, **kwargs):
    return subprocess.run(argv, text=True, capture_output=True, **kwargs)


def _ok(result):
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def artifact(tmp_path):
    target = tmp_path / "libdelegate.so"
    runtime = tmp_path / "executorch/lib/libexecutorch.so"
    extension = runtime.with_name("libexecutorch_extension_cuda.so")
    kernels = runtime.with_name("libkernels.so")
    pybindings = runtime.parent.parent / "extension/pybindings/_C.test.so"
    for path in (target, runtime, extension, kernels, pybindings):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    data = {
        "dialect": "gnu",
        "runpath": _RUNPATH,
        "path_tag": "RUNPATH",
        "needed": [
            "libexecutorch.so",
            "libexecutorch_extension_cuda.so",
            "libstdc++.so.6",
        ],
        "versions": _BASE_VERSIONS,
        "runtime_versions": _BASE_VERSIONS,
        "kernel_versions": _BASE_VERSIONS,
        "failure": None,
    }
    config = tmp_path / "readelf.json"
    reader = tmp_path / "readelf"
    reader.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        f"d = json.loads(Path({str(config)!r}).read_text())\n"
        "flag, name = sys.argv[1], Path(sys.argv[-1]).name\n"
        "if d['failure'] == [flag, name]:\n"
        "    print('deliberate read failure', file=sys.stderr)\n"
        "    sys.exit(17)\n"
        "eu = d['dialect'] == 'elfutils'\n"
        "if flag == '-d':\n"
        "    needed = d['needed'] if name == 'libdelegate.so' else (\n"
        "        ['libkernels.so'] if name == '_C.test.so' else [])\n"
        "    for lib in needed:\n"
        "        print((' NEEDED' if eu else ' 0x0001 (NEEDED)') + ' Shared library: [' + lib + ']')\n"
        "    if name == 'libdelegate.so' and d['runpath'] is not None:\n"
        "        tag = d['path_tag']\n"
        "        print((' ' + tag if eu else ' 0x001d (' + tag + ')') + ' Library runpath: [' + d['runpath'] + ']')\n"
        "elif flag in ('-Ws', '--dyn-syms'):\n"
        "    ndx = ('UNDEF' if eu else 'UND') if name == 'libdelegate.so' else '12'\n"
        f"    print(' 1: 00000000 8 FUNC GLOBAL DEFAULT ' + ndx + ' {_REGISTER}')\n"
        "elif flag == '-V':\n"
        "    key = 'versions' if name == 'libdelegate.so' else (\n"
        "        'kernel_versions' if name == 'libkernels.so' else 'runtime_versions')\n"
        "    for node in d[key].split():\n"
        "        print(' 0x0010: Name: ' + node + ' Flags: none Version: 2')\n"
        "else:\n"
        "    sys.exit(18)\n"
    )
    reader.chmod(0o755)

    def invoke(*, options=None, guard=_GUARD):
        config.write_text(json.dumps(data))
        if options is None:
            options = [str(runtime), _RUNPATH, _X86]
        return _run(["sh", str(guard), str(reader), str(target), *options])

    return data, invoke, runtime


@pytest.mark.parametrize("dialect", ["gnu", "llvm", "elfutils"])
@pytest.mark.parametrize("with_expected", [False, True])
def test_cuda_13_uses_actual_runpath(artifact, dialect, with_expected):
    data, invoke, runtime = artifact
    data["dialect"] = dialect
    data["needed"].append("libcudart.so.13")
    options = [str(runtime)] + ([_RUNPATH, _X86] if with_expected else [])
    _ok(invoke(options=options))


@pytest.mark.parametrize("major", [12, 14, 99])
def test_unsupported_cuda_major(artifact, major):
    data, invoke, _ = artifact
    data["needed"].append(f"libcudart.so.{major}")
    result = invoke()
    assert result.returncode != 0
    assert "requires CUDA 13" in result.stderr


@pytest.mark.parametrize("suffix", ["", "-backup", "/not-the-lib-dir"])
def test_cuda_path_must_be_an_actual_entry(artifact, suffix):
    data, invoke, runtime = artifact
    data["needed"].append("libcudart.so.13")
    data["runpath"] = _RUNPATH.rsplit(":", 1)[0]
    if suffix:
        data["runpath"] += ":$ORIGIN/../../nvidia/cu13/lib" + suffix
    result = invoke(options=[str(runtime)])
    assert result.returncode != 0
    assert "RUNPATH carries no nvidia/cu13/lib" in result.stderr


def test_expected_runpath_is_only_an_equality_check(artifact):
    data, invoke, runtime = artifact
    data["needed"].append("libcudart.so.13")
    result = invoke(options=[str(runtime), "$ORIGIN/../../executorch/lib"])
    assert result.returncode != 0
    assert "RUNPATH the build did not ask for" in result.stderr


@pytest.mark.parametrize("position", [0, 1, 2])
def test_supplied_empty_option_is_rejected(artifact, position):
    _, invoke, runtime = artifact
    options = [str(runtime), _RUNPATH, _X86][: position + 1]
    options[position] = ""
    result = invoke(options=options)
    assert result.returncode != 0
    assert "must not be empty" in result.stderr


@pytest.mark.parametrize(
    "tag",
    [
        "manylinux_2_34",
        "manylinux_2_28",
        "manylinux_2_39",
        "linux_x86_64",
        "manylinux_2_39_aarch64",
        "manylinux_2_39_x86_64",
        "unknown",
    ],
)
def test_unsupported_supplied_tag_is_rejected(artifact, tag):
    _, invoke, runtime = artifact
    result = invoke(options=[str(runtime), _RUNPATH, tag])
    assert result.returncode != 0
    assert "unsupported manylinux tag" in result.stderr


def test_omitted_optional_arguments_are_explicit(artifact):
    _, invoke, runtime = artifact
    result = invoke(options=[])
    _ok(result)
    assert "no runtime given" in result.stderr
    result = invoke(options=[str(runtime)])
    _ok(result)
    assert "no manylinux tag" in result.stderr


# Boundary and gap cases from auditwheel 6.8.2's architecture-specific policy.
@pytest.mark.parametrize(
    "tag,node,allowed",
    [
        (_X86, "GLIBCXX_3.4.23", True),
        (_X86, "GLIBCXX_3.4.24", True),
        (_X86, "GLIBCXX_3.4.25", False),
        (_ARM, "GLIBCXX_3.4.24", True),
        (_ARM, "GLIBCXX_3.4.25", True),
        (_ARM, "GLIBCXX_3.4.34", False),
        (_X86, "CXXABI_1.3.10", True),
        (_X86, "CXXABI_1.3.11", True),
        (_X86, "CXXABI_1.3.12", False),
        (_ARM, "CXXABI_1.3.11", True),
        (_ARM, "CXXABI_1.3.12", True),
        (_ARM, "CXXABI_1.3.16", False),
        (_X86, "GLIBC_2.27", True),
        (_X86, "GLIBC_2.28", True),
        (_X86, "GLIBC_2.29", False),
        (_ARM, "GLIBC_2.28", True),
        (_ARM, "GLIBC_2.29", True),
        (_ARM, "GLIBC_2.39", False),
        (_X86, "GCC_4.8.0", True),
        (_X86, "GCC_7.0.0", True),
        (_X86, "GCC_7.1.0", False),
        (_ARM, "GCC_4.5.0", True),
        (_ARM, "GCC_7.0.0", True),
        (_ARM, "GCC_14.0.0", False),
        (_ARM, "GCC_15.0.0", False),
        (_X86, "GLIBC_2.19", False),
        (_ARM, "GLIBC_2.37", False),
        (_X86, "GCC_4.5.0", False),
        (_ARM, "GCC_4.8.0", False),
        (_X86, "CXXABI_TM_1", True),
        (_ARM, "CXXABI_TM_1", True),
        (_X86, "CXXABI_FLOAT128", True),
        (_ARM, "CXXABI_FLOAT128", False),
        (_X86, "GLIBC_ABI_DT_RELR", False),
        (_ARM, "GLIBC_ABI_DT_RELR", False),
        (_X86, "GLIBC_PRIVATE", False),
        (_ARM, "GLIBC_PRIVATE", False),
    ],
)
def test_manylinux_policy_membership(artifact, tag, node, allowed):
    data, invoke, runtime = artifact
    data["versions"] += " " + node
    result = invoke(options=[str(runtime), _RUNPATH, tag])
    if allowed:
        _ok(result)
    else:
        assert result.returncode != 0
        assert node in result.stderr
        assert tag in result.stderr


@pytest.mark.parametrize(
    "flag,name",
    [
        ("-d", "libdelegate.so"),
        ("--dyn-syms", "libdelegate.so"),
        ("-Ws", "libdelegate.so"),
        ("-V", "libdelegate.so"),
        ("-Ws", "libexecutorch.so"),
        ("-d", "_C.test.so"),
        ("-d", "libexecutorch.so"),
        ("-d", "libexecutorch_extension_cuda.so"),
        ("-d", "libkernels.so"),
        ("-V", "libexecutorch.so"),
        ("-V", "libkernels.so"),
    ],
)
def test_every_read_failure_is_fatal(artifact, flag, name):
    data, invoke, _ = artifact
    data["failure"] = [flag, name]
    result = invoke()
    assert result.returncode != 0
    assert name in result.stderr


@pytest.mark.parametrize("dialect", ["gnu", "llvm", "elfutils"])
def test_legacy_rpath_is_rejected(artifact, dialect):
    data, invoke, _ = artifact
    data.update(dialect=dialect, path_tag="RPATH")
    result = invoke()
    assert result.returncode != 0
    assert "DT_RPATH" in result.stderr


def test_untagged_named_nodes_use_pybindings_closure(artifact):
    data, invoke, runtime = artifact
    data["versions"] += " CXXABI_TM_1"
    result = invoke(options=[str(runtime)])
    assert result.returncode != 0
    data["kernel_versions"] += " CXXABI_TM_1"
    _ok(invoke(options=[str(runtime)]))


@pytest.mark.parametrize(
    "extra", [[], ["reader"], ["reader", "target", "runtime", "path", _X86, "extra"]]
)
def test_invalid_argument_count(extra):
    result = _run(["sh", str(_GUARD), *extra])
    assert result.returncode != 0
    assert "expected <readelf> <shared-object>" in result.stderr


def test_missing_runtime_is_fatal(artifact):
    _, invoke, runtime = artifact
    result = invoke(options=[str(runtime.with_name("missing.so"))])
    assert result.returncode != 0
    assert "does not exist" in result.stderr


def test_runtime_version_output_must_not_be_empty(artifact):
    data, invoke, _ = artifact
    data["runtime_versions"] = ""
    data["kernel_versions"] = ""
    result = invoke()
    assert result.returncode != 0
    assert "could not read symbol versions beside" in result.stderr


def test_missing_pybindings_is_fatal(artifact):
    _, invoke, runtime = artifact
    (runtime.parent.parent / "extension/pybindings/_C.test.so").unlink()
    result = invoke()
    assert result.returncode != 0
    assert "could not find the pybindings extension" in result.stderr


@pytest.fixture
def native_tools():
    tools = {name: shutil.which(name) for name in _NATIVE_TOOLS}
    if sys.platform != "linux" or not all(tools.values()):
        pytest.skip("needs Linux, CMake >= 3.28, a C++ compiler, readelf and patchelf")
    return tools


@pytest.mark.unit
def test_the_lane_that_runs_this_file_installs_the_tools_it_needs() -> None:
    """A tool the job does not have skips the real-build tests, and a skip leaves the job green.

    They are the only tests here that configure the native project, build it and read the result
    back with the platform's own tools. With patchelf absent from the lint job, eleven of them did
    nothing on every pull request while the job reported success. Whatever the runner image does not
    carry has to be installed by the job, and what the job installs is the lint dependency group.
    """
    import tomllib

    declared = {
        canonicalize_name(Requirement(dependency).name)
        for dependency in tomllib.loads(
            (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )["dependency-groups"]["lint"]
    }
    missing = [
        tool
        for tool in _NATIVE_TOOLS
        if tool not in _IMAGE_TOOLS and canonicalize_name(tool) not in declared
    ]
    assert not missing, (
        f"the lint group installs none of {missing}, so the real-build tests in this file skip "
        "there and nothing that can block a pull request exercises the guard"
    )


def _native_project(tmp_path, tools, *, mutation=None, static_cuda=False):
    native = tmp_path / "native"
    shutil.copytree(_NATIVE, native)
    cmake_file = native / "CMakeLists.txt"
    cmake = cmake_file.read_text()
    guard_command = '    COMMAND sh\n      "${CMAKE_CURRENT_LIST_DIR}/check_imports_executorch_runtime.sh"'
    if mutation == "remove":
        start = cmake.index(
            "  add_custom_command", cmake.index("if(TORCH_TENSORRT_READELF)")
        )
        end = cmake.index("    VERBATIM)", start) + len("    VERBATIM)")
        cmake = cmake[:start] + cmake[end:]
    elif mutation == "early_return":
        marker = "find_program(TORCH_TENSORRT_READELF NAMES"
        assert marker in cmake
        cmake = cmake.replace(marker, "return()\n" + marker, 1)
    elif mutation == "inert":
        assert guard_command in cmake
        cmake = cmake.replace(
            guard_command, '    COMMAND "${CMAKE_COMMAND}" -E true', 1
        )
    elif mutation == "static_check":
        marker = 'if(_extension_cuda_type STREQUAL "STATIC_LIBRARY")'
        assert marker in cmake
        cmake = cmake.replace(marker, "if(FALSE)", 1)
    elif mutation == "retention":
        marker = '"LINKER:--push-state,--no-as-needed,$<TARGET_FILE:executorch_backend_tensorrt>,--pop-state"'
        assert marker in cmake
        cmake = cmake.replace(marker, '"LINKER:--as-needed"', 1)
    cmake_file.write_text(cmake)

    source = tmp_path / "source"
    modules = source / "cmake/Modules"
    sources = source / "cpp/src/torch_tensorrt/executorch"
    modules.mkdir(parents=True)
    sources.mkdir(parents=True)
    (sources / "TensorRTBackend.cpp").write_text(
        "#include <string>\n"
        "namespace executorch { namespace runtime {\n"
        "struct Backend {}; void register_backend(const Backend&);\n"
        "}}\n"
        'extern "C" void extension_cuda();\n'
        'extern "C" void cuda_fixture();\n'
        'namespace { struct R { std::string s; R() : s("fixture") {\n'
        "executorch::runtime::register_backend({}); extension_cuda(); cuda_fixture();\n"
        "} } r; }\n"
    )
    for name in ("TensorRTBlobHeader.cpp", "WeightStreamingBudget.cpp"):
        (sources / name).write_text("\n")
    runtime_dir = tmp_path / "executorch/lib"
    runtime_dir.mkdir(parents=True)
    library_sources = {
        "libexecutorch.so": '#include <cstdio>\nnamespace executorch { namespace runtime { struct Backend {}; void register_backend(const Backend&) { puts("registered"); } }}\n',
        "libexecutorch_extension_cuda.so": 'extern "C" void extension_cuda() {}\n',
        "libcudart.so.13": 'extern "C" void cuda_fixture() {}\n',
        "libunrelated.so": 'extern "C" void unrelated() {}\n',
    }
    for name, body in library_sources.items():
        cpp = tmp_path / (name + ".cpp")
        cpp.write_text(body)
        _ok(
            _run(
                [
                    tools["c++"],
                    "-shared",
                    "-fPIC",
                    str(cpp),
                    "-o",
                    str(runtime_dir / name),
                    f"-Wl,-soname,{name}",
                ]
            )
        )
    pybindings = runtime_dir.parent / "extension/pybindings"
    pybindings.mkdir(parents=True)
    shutil.copy2(
        runtime_dir / "libexecutorch_extension_cuda.so", pybindings / "_C.test.so"
    )
    (modules / "FindTensorRT.cmake").write_text(
        "add_library(TensorRT::nvinfer INTERFACE IMPORTED)\n"
    )
    (modules / "FindCUDAToolkit.cmake").write_text(
        "add_library(CUDA::cudart SHARED IMPORTED)\n"
        f'set_target_properties(CUDA::cudart PROPERTIES IMPORTED_LOCATION "{runtime_dir}/libcudart.so.13")\n'
    )
    prefix = runtime_dir.parent / "share/cmake"
    prefix.mkdir(parents=True)
    (prefix / "executorch-config.cmake").write_text(
        "add_library(executorch::runtime SHARED IMPORTED)\n"
        f'set_target_properties(executorch::runtime PROPERTIES IMPORTED_LOCATION "{runtime_dir}/libexecutorch.so" INTERFACE_LINK_OPTIONS "-Wl,-rpath,{runtime_dir}")\n'
        f'add_library(executorch::extension_cuda {"STATIC" if static_cuda else "SHARED"} IMPORTED)\n'
        f'set_target_properties(executorch::extension_cuda PROPERTIES IMPORTED_LOCATION "{runtime_dir}/libexecutorch_extension_cuda.so")\n'
    )
    (tmp_path / "main.cpp").write_text("int main() { return 0; }\n")
    (tmp_path / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.28)\nproject(guard_fixture LANGUAGES CXX)\n"
        "add_subdirectory(native)\n"
        "add_executable(consumer main.cpp)\n"
        'target_link_options(consumer PRIVATE "LINKER:--as-needed")\n'
        f'target_link_libraries(consumer PRIVATE executorch::backend_tensorrt "{runtime_dir}/libunrelated.so")\n'
        f'set_target_properties(consumer PROPERTIES BUILD_RPATH "{runtime_dir};${{CMAKE_BINARY_DIR}}/native")\n'
    )
    build = tmp_path / "build"
    configure = _run(
        [
            tools["cmake"],
            "-S",
            str(tmp_path),
            "-B",
            str(build),
            f"-DTORCH_TENSORRT_SOURCE_DIR={source}",
            f"-DCMAKE_PREFIX_PATH={prefix}",
            f"-DTORCH_TENSORRT_PATCHELF={tools['patchelf']}",
            f"-DTORCH_TENSORRT_READELF={tools['readelf']}",
        ]
    )
    return configure, build, runtime_dir


@pytest.mark.parametrize("mutation", [None, "remove", "early_return", "inert"])
def test_production_post_build_rejects_bad_artifact(tmp_path, native_tools, mutation):
    tools = dict(native_tools)
    real_patchelf = tools["patchelf"]
    wrapper = tmp_path / "patchelf-with-bad-needed"
    wrapper.write_text(
        "#!/bin/sh\nset -eu\n"
        f'{shlex.quote(real_patchelf)} "$@"\n'
        'if [ "$1" = "--set-rpath" ]; then\n'
        f'  {shlex.quote(real_patchelf)} --replace-needed libcudart.so.13 libcudart.so.14 "$3"\n'
        "fi\n"
    )
    wrapper.chmod(0o755)
    tools["patchelf"] = str(wrapper)
    configure, build, _ = _native_project(tmp_path, tools, mutation=mutation)
    _ok(configure)
    result = _run(
        [
            tools["cmake"],
            "--build",
            str(build),
            "--target",
            "executorch_backend_tensorrt",
        ]
    )
    if mutation is None:
        assert result.returncode != 0, result.stdout + result.stderr
        assert "requires CUDA 13" in result.stdout + result.stderr
    else:
        _ok(result)
        dyn = _run(
            [
                tools["readelf"],
                "-d",
                str(build / "native/libexecutorch_backend_tensorrt.so"),
            ]
        )
        _ok(dyn)
        assert "libcudart.so.14" in dyn.stdout


@pytest.mark.parametrize("mutation", [None, "retention"])
def test_production_alias_retains_registration(tmp_path, native_tools, mutation):
    configure, build, runtime_dir = _native_project(
        tmp_path, native_tools, mutation=mutation
    )
    _ok(configure)
    _ok(_run([native_tools["cmake"], "--build", str(build)]))
    consumer = build / "consumer"
    dyn = _run([native_tools["readelf"], "-d", str(consumer)])
    _ok(dyn)
    assert ("libexecutorch_backend_tensorrt.so" in dyn.stdout) == (mutation is None)
    assert "libunrelated.so" not in dyn.stdout
    result = _run(
        [str(consumer)], env={**os.environ, "LD_LIBRARY_PATH": str(runtime_dir)}
    )
    _ok(result)
    assert ("registered" in result.stdout) == (mutation is None)
    artifact = build / "native/libexecutorch_backend_tensorrt.so"
    runpath = _run([native_tools["patchelf"], "--print-rpath", str(artifact)])
    _ok(runpath)
    assert runpath.stdout.strip() == _RUNPATH


@pytest.mark.parametrize("mutation", [None, "static_check"])
def test_static_cuda_target_is_rejected_at_configure(tmp_path, native_tools, mutation):
    configure, _, _ = _native_project(
        tmp_path, native_tools, mutation=mutation, static_cuda=True
    )
    if mutation is None:
        assert configure.returncode != 0
        assert (
            "executorch::extension_cuda is a STATIC_LIBRARY"
            in configure.stdout + configure.stderr
        )
    else:
        _ok(configure)


@pytest.mark.parametrize("reader_name", ["readelf", "llvm-readelf", "eu-readelf"])
def test_real_readers_accept_and_reject_elf(tmp_path, native_tools, reader_name):
    reader = shutil.which(reader_name)
    if reader is None:
        pytest.skip(f"{reader_name} is not installed")
    tools = {**native_tools, "readelf": reader}
    configure, build, runtime_dir = _native_project(tmp_path, tools)
    _ok(configure)
    _ok(
        _run(
            [
                tools["cmake"],
                "--build",
                str(build),
                "--target",
                "executorch_backend_tensorrt",
            ]
        )
    )
    target = build / "native/libexecutorch_backend_tensorrt.so"
    argv = [
        "sh",
        str(_GUARD),
        reader,
        str(target),
        str(runtime_dir / "libexecutorch.so"),
    ]
    _ok(_run(argv))
    _ok(
        _run([tools["patchelf"], "--force-rpath", "--set-rpath", _RUNPATH, str(target)])
    )
    result = _run(argv)
    assert result.returncode != 0
    assert "DT_RPATH" in result.stderr


@pytest.mark.parametrize(
    "mutation", ["cuda", "cuda_path", "policy_loop", "gcc", "empty", "unknown", "exit"]
)
def test_guard_removal_controls(artifact, tmp_path, mutation):
    data, invoke, runtime = artifact
    guard = tmp_path / "mutated-guard.sh"
    source = _GUARD.read_text()
    options = None
    if mutation == "cuda":
        old, new = "libcudart.so.13) ;;", "libcudart.so.13|libcudart.so.14) ;;"
        data["needed"].append("libcudart.so.14")
    elif mutation == "cuda_path":
        old = "*':$ORIGIN/../../nvidia/cu13/lib:'*) ;;"
        new = "*) ;;"
        data["needed"].append("libcudart.so.13")
        data["runpath"] = _RUNPATH.rsplit(":", 1)[0]
        options = [str(runtime)]
    elif mutation == "policy_loop":
        old = 'for node in $(versions "${target_versions}"); do'
        new = "for node in; do"
        data["versions"] += " GCC_7.1.0"
    elif mutation == "gcc":
        old, new = (
            "(GLIBCXX|CXXABI|GLIBC|GCC|LIBATOMIC|ZLIB)",
            "(GLIBCXX|CXXABI|GLIBC|LIBATOMIC|ZLIB)",
        )
        data["versions"] += " GCC_7.1.0"
    elif mutation == "empty":
        old = '[ -n "${argument}" ] || fail "supplied arguments must not be empty"'
        new = ":"
        options = [str(runtime), _RUNPATH, ""]
    elif mutation == "unknown":
        old = '*) fail "unsupported manylinux tag: ${manylinux_tag}" ;;'
        new = '*) manylinux_tag="" ;;'
        options = [str(runtime), _RUNPATH, "unknown"]
    else:
        old, new = "set -u", "exit 0\nset -u"
        data["needed"].remove("libexecutorch.so")
    assert source.count(old) == 1
    # The intact guard must reject the same input before the control disables that check.
    assert invoke(options=options).returncode != 0
    guard.write_text(source.replace(old, new, 1))
    result = invoke(options=options, guard=guard)
    if mutation == "empty":
        assert "must not be empty" not in result.stderr
        assert "unsupported manylinux tag" in result.stderr
    else:
        _ok(result)


@pytest.mark.unit
def test_every_site_naming_the_platform_tag_agrees() -> None:
    """Three files name the tag, and they have to say the same thing.

    The workflow tags the wheel, the native build passes a tag to the guard, and the guard decides
    which tags it accepts. When one moved and another did not, the build failed with an unsupported
    tag well after the code was otherwise correct.
    """
    cmake = (_NATIVE / "CMakeLists.txt").read_text(encoding="utf-8")
    guard = _GUARD.read_text(encoding="utf-8")
    workflow = (_ROOT / ".github/workflows/build_linux.yml").read_text(encoding="utf-8")
    checker = (_ROOT / ".github/scripts/check-executorch-runtime-wheel.py").read_text(
        encoding="utf-8"
    )
    readme = (_ROOT / "py/torch-tensorrt-executorch-runtime/README.md").read_text(
        encoding="utf-8"
    )
    for arch, tag in (("aarch64", _ARM), ("x86_64", _X86)):
        floor = tag.removeprefix("manylinux_").removesuffix(f"_{arch}")
        other = tag.replace(floor, "2_28" if floor != "2_28" else "2_35")
        assert f'"{tag}"' in cmake, f"the native build does not name {tag}"
        assert tag in guard, f"the guard does not accept {tag}"
        assert f"platform_tag={tag}" in workflow, f"the workflow does not apply {tag}"
        # The checker derives the tag from a per-architecture floor rather than naming it whole.
        assert (
            f'"{arch}": "{floor}"' in checker
        ), f"the checker's floor for {arch} is not {floor}"
        assert tag in readme, f"the documentation does not name {tag}"
        assert other not in cmake, f"the native build still names {other}"
        assert (
            f"platform_tag={other}" not in workflow
        ), f"the workflow still applies {other}"
        assert other not in readme, f"the documentation still names {other}"


@pytest.mark.unit
def test_the_shipped_binaries_can_find_the_libraries_they_need() -> None:
    """A program in the wheel is not launched through Python, so nothing prepares its search path.

    The example runner shipped without entries for TensorRT and the CUDA runtime, which live in their
    own distributions beside this one, so it exited before main with a loader error naming a library
    that was installed the whole time. The delegate library in the companion wheel already carried
    the right entries, which is why it loaded and the binary did not.
    """
    build = (_ROOT / "examples/executorch_reference_runner/BUILD").read_text(
        encoding="utf-8"
    )
    binaries = [
        block
        for block in build.split("cc_binary(")[1:]
        if "kv_cache_decode_check" in block or "example_executorch_runner" in block
    ]
    assert len(binaries) == 2, f"expected two shipped binaries, found {len(binaries)}"
    for block in binaries:
        name = block.split('name = "', 1)[1].split('"', 1)[0]
        for needed in (
            "$$ORIGIN/../lib",
            "$$ORIGIN/../../tensorrt_libs",
            "$$ORIGIN/../../nvidia/cu13/lib",
        ):
            assert needed in block, f"{name} has no run path entry for {needed}"


def _drop_needed(data, name):
    data["needed"] = [lib for lib in data["needed"] if lib != name]


@pytest.mark.parametrize(
    "break_it,expected",
    [
        (
            lambda d: _drop_needed(d, "libexecutorch.so"),
            "no DT_NEEDED on libexecutorch.so",
        ),
        (
            lambda d: _drop_needed(d, "libexecutorch_extension_cuda.so"),
            "no DT_NEEDED on libexecutorch_extension_cuda.so",
        ),
        (
            lambda d: _drop_needed(d, "libstdc++.so.6"),
            "no DT_NEEDED on libstdc++",
        ),
        (
            lambda d: d.update(path_tag="RPATH"),
            "carries DT_RPATH rather than DT_RUNPATH",
        ),
        (lambda d: d.update(runpath=None), "carries no RUNPATH"),
    ],
)
@pytest.mark.unit
def test_each_linkage_check_rejects_what_it_is_for(artifact, break_it, expected):
    """Each of these checks passed its own suite with the check deleted.

    The guard is what stands between a wheel that cannot load and whoever installs it, so a check
    nothing exercises is the same as no check. One case per rejection, each crafting the input that
    rejection exists for.
    """
    data, invoke, runtime = artifact
    break_it(data)
    result = invoke()
    assert result.returncode != 0, result.stdout
    assert expected in result.stderr, result.stderr


@pytest.mark.parametrize(
    "symbol", ["ZLIB_1.2.13", "LIBATOMIC_1.3", "GLIBCXX_3.4.40", "GLIBC_2.99"]
)
@pytest.mark.unit
def test_a_symbol_above_the_platform_ceiling_is_rejected(artifact, symbol):
    """Two of the platform's six symbol families were missing from the tables and the collector.

    A requirement above the ceiling in either of those two passed in silence, which is the one thing
    the tag is a promise about.
    """
    data, invoke, runtime = artifact
    data["versions"] = f"{_BASE_VERSIONS} {symbol}"
    result = invoke()
    assert result.returncode != 0, result.stdout
    assert symbol in result.stderr, result.stderr


@pytest.mark.unit
def test_a_library_outside_the_allowed_set_is_rejected(artifact):
    """Nothing enumerated what the delegate links, so an unintended link passed in silence.

    libpython is the one that matters most. The wheel is tagged for any Python 3 because the payload
    has no Python ABI, and linking libpython would make that tag wrong while the wheel still installed
    everywhere.
    """
    data, invoke, runtime = artifact
    data["needed"].append("libpython3.12.so.1.0")
    result = invoke()
    assert result.returncode != 0, result.stdout
    assert "libpython3.12.so.1.0" in result.stderr, result.stderr


@pytest.mark.unit
def test_the_native_build_runs_the_guard_after_linking() -> None:
    """Deleting the step that runs the guard left this suite green wherever patchelf was missing.

    The cases that build for real skip without it, so nothing noticed that the built library had
    stopped being checked at all. This reads the build file instead, which is weaker than running it
    but is the only check that holds where the toolchain is absent.
    """
    build = (
        _ROOT / "py/torch-tensorrt-executorch-runtime/native/CMakeLists.txt"
    ).read_text(encoding="utf-8")
    commands = [
        block
        for block in build.split(
            "add_custom_command(TARGET executorch_backend_tensorrt POST_BUILD"
        )[1:]
    ]
    assert commands, "nothing runs after the delegate is linked"
    guard = [b for b in commands if "check_imports_executorch_runtime.sh" in b]
    assert (
        guard
    ), f"the guard is not run after linking: {len(commands)} post-build steps"
    # It has to receive the reader and the built library, or it checks nothing useful.
    assert "TORCH_TENSORRT_READELF" in guard[0], guard[0][:300]
    assert "TARGET_FILE:executorch_backend_tensorrt" in guard[0], guard[0][:300]


_LITERAL_OR_COMMENT = re.compile(
    r'"(?:\\.|[^"\\])*"' r"|'(?:\\.|[^'\\])*'" r"|//[^\n]*" r"|/\*.*?\*/", re.S
)


def _code_only(source: str) -> str:
    """The source with its comments removed, so no check below can be satisfied by a comment.

    String and character literals are matched first and kept, so a ``//`` inside one survives. The
    delegate sources use no raw string literals, which this would not handle.
    """
    return _LITERAL_OR_COMMENT.sub(
        lambda match: match.group(0) if match.group(0)[0] in "\"'" else " ", source
    )


def _definition_body(source: str, signature: str) -> str:
    """The braced body of the definition introduced by ``signature``.

    Reading one body rather than the whole file is what lets a check say where something happens
    instead of only that the file mentions it somewhere.
    """
    start = source.index(signature)
    depth = 0
    for index in range(source.index("{", start), len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unbalanced braces after {signature!r}")


def _assert_one_shared_runtime(source: str, header: str) -> None:
    code = _code_only(source)
    accessor = "nvinfer1::IRuntime* shared_runtime()"
    assert accessor in code, "no shared runtime accessor"
    # Exactly one place builds it, and it is that accessor. The availability check used to build a
    # second one, which is what made TensorRT report an ignored logger on every load, and it ran
    # first: the runtime asks whether the backend is available before it initialises anything, so
    # the logger TensorRT kept for the process was that one's, on a stack frame already gone.
    builds = [line for line in code.splitlines() if "createInferRuntime" in line]
    assert len(builds) == 1, builds
    assert builds[0] in _definition_body(code, accessor), builds[0]
    availability = _definition_body(code, "bool TensorRTBackend::is_available() const")
    assert "shared_runtime()" in availability, availability
    assert "createInferRuntime" not in availability, availability
    # And no lock around the deserialize, because TensorRT lists that call as thread safe.
    assert (
        "deserialize_lock" not in code
    ), "a lock was added around a documented-safe call"
    # And the handle no longer carries one of its own.
    assert "IRuntime> runtime;" not in _code_only(
        header
    ), "the handle still owns a runtime"


@pytest.mark.parametrize("mutation", [None, "availability-builds-one", "commented-out"])
@pytest.mark.unit
def test_the_backend_shares_one_tensorrt_runtime(mutation) -> None:
    """A runtime per program bought nothing and warned on every load after the first.

    TensorRT documents a runtime as sharable across threads for nonmodifying use, and logs that a
    second one's logger is ignored because it already has one. Deserializing from a runtime is on its
    thread-safe list, so no lock is taken around it; what it says to serialize is the modifying
    setters, and this backend calls none of them.

    This reads the arrangement of the code, not a running backend: TensorRT and the ExecuTorch
    headers are absent from every lane that can run this file, so nothing here can compile or call
    the delegate. What it does establish is where the runtime is built and who asks for it, with
    comments stripped first and each definition read on its own, so neither a comment nor a mention
    elsewhere in the file can stand in for the code. That the file compiles at all is established by
    the lane that builds the wheel, and what TensorRT then does with one runtime by the device lanes.
    """
    source = (
        _ROOT / "cpp/src/torch_tensorrt/executorch/TensorRTBackend.cpp"
    ).read_text(encoding="utf-8")
    header = (
        _ROOT / "cpp/include/torch_tensorrt/executorch/TensorRTBackend.h"
    ).read_text(encoding="utf-8")
    if mutation is None:
        _assert_one_shared_runtime(source, header)
        return
    if mutation == "availability-builds-one":
        # The arrangement this change replaced: its own runtime, from a logger on the stack.
        original = "return shared_runtime() != nullptr;"
        assert original in source
        source = source.replace(
            original,
            "TRTLogger logger;\n"
            "  TRTUniquePtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));\n"
            "  return runtime != nullptr;",
        )
    else:
        # The text still says it, in a comment, and nothing builds a runtime.
        original = "  static TRTUniquePtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));"
        assert original in source
        source = source.replace(original, "//" + original)
    with pytest.raises(AssertionError):
        _assert_one_shared_runtime(source, header)
