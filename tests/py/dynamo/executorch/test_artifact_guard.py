"""CPU-only behavior checks for the companion's native artifact guard."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

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
_ARM = "manylinux_2_39_aarch64"
_BASE_VERSIONS = "CXXABI_1.3 GLIBCXX_3.4.21 GLIBC_2.17 GCC_3.0"


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
        "manylinux_2_28_aarch64",
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
        (_ARM, "GLIBCXX_3.4.32", True),
        (_ARM, "GLIBCXX_3.4.33", True),
        (_ARM, "GLIBCXX_3.4.34", False),
        (_X86, "CXXABI_1.3.10", True),
        (_X86, "CXXABI_1.3.11", True),
        (_X86, "CXXABI_1.3.12", False),
        (_ARM, "CXXABI_1.3.14", True),
        (_ARM, "CXXABI_1.3.15", True),
        (_ARM, "CXXABI_1.3.16", False),
        (_X86, "GLIBC_2.27", True),
        (_X86, "GLIBC_2.28", True),
        (_X86, "GLIBC_2.29", False),
        (_ARM, "GLIBC_2.38", True),
        (_ARM, "GLIBC_2.39", True),
        (_ARM, "GLIBC_2.40", False),
        (_X86, "GCC_4.8.0", True),
        (_X86, "GCC_7.0.0", True),
        (_X86, "GCC_7.1.0", False),
        (_ARM, "GCC_13.0.0", True),
        (_ARM, "GCC_14.0", True),
        (_ARM, "GCC_14.0.0", True),
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
        (_ARM, "GLIBC_ABI_DT_RELR", True),
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
    tools = {
        name: shutil.which(name) for name in ("cmake", "c++", "readelf", "patchelf")
    }
    if sys.platform != "linux" or not all(tools.values()):
        pytest.skip("needs Linux, CMake >= 3.28, a C++ compiler, readelf and patchelf")
    return tools


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
        old, new = "(GLIBCXX|CXXABI|GLIBC|GCC)", "(GLIBCXX|CXXABI|GLIBC)"
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
