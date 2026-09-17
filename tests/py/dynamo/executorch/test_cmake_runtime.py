# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise the installed companion CMake target with a CPU-only native fixture."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit
_CONFIG = (
    Path(__file__).resolve().parents[4]
    / "py/torch-tensorrt-executorch-runtime/cmake/torchtrt_executorch-config.cmake"
)


@pytest.fixture
def linker_tools():
    tools = {name: shutil.which(name) for name in ("cmake", "c++", "readelf")}
    if sys.platform != "linux" or not all(tools.values()):
        pytest.skip("needs cmake, a C++ compiler, readelf, and a Linux linker")
    return tools


def _run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def _installed_consumer(tmp_path, tools, config, old_dtags, example=None):
    prefix = tmp_path / "prefix"
    config_dir = prefix / "lib/cmake/torchtrt_executorch"
    config_dir.mkdir(parents=True)
    (config_dir / _CONFIG.name).write_text(config)
    libraries = ["executorch_backend_tensorrt", "unrelated"]
    if example is not None:
        components = ("backend_cuda", "kernels_optimized")
        libraries.extend(f"executorch_{name}" for name in components)
        et_config = prefix / "lib/cmake/executorch"
        et_config.mkdir()
        (et_config / "executorch-config.cmake").write_text(
            "add_library(executorch::runtime INTERFACE IMPORTED)\n"
            + "".join(
                f"add_library(executorch::{name} SHARED IMPORTED)\n"
                f"set_target_properties(executorch::{name} PROPERTIES\n"
                f'  IMPORTED_LOCATION "{prefix}/lib/libexecutorch_{name}.so"\n'
                '  INTERFACE_LINK_OPTIONS "LINKER:--push-state,--no-as-needed,'
                f'{prefix}/lib/libexecutorch_{name}.so,--pop-state")\n'
                for name in components
            )
        )
    for name in libraries:
        source = tmp_path / f"{name}.cpp"
        source.write_text(
            "#include <cstdio>\nnamespace { struct Registration { "
            f'Registration() {{ std::puts("{name}"); }}'
            " } registration; }\n"
        )
        _run(
            [
                tools["c++"],
                "-shared",
                "-fPIC",
                str(source),
                f"-Wl,-soname,lib{name}.so",
                "-o",
                str(prefix / f"lib/lib{name}.so"),
            ]
        )
    app = tmp_path / "app"
    app.mkdir()
    (app / "main.cpp").write_text("int main() { return 0; }\n")
    (app / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.28)\nproject(consumer LANGUAGES CXX)\n"
        "add_executable(my_app main.cpp)\n"
        'target_link_options(my_app PRIVATE "LINKER:--as-needed")\n'
        + (
            example
            if example is not None
            else "find_package(torchtrt_executorch REQUIRED)\n"
            "target_link_libraries(my_app PRIVATE torchtrt::executorch_backend)\n"
        )
        + f'\ntarget_link_libraries(my_app PRIVATE "{prefix}/lib/libunrelated.so")\n'
    )
    build = tmp_path / "build"
    flags = ["-DCMAKE_EXE_LINKER_FLAGS=-Wl,--disable-new-dtags"] if old_dtags else []
    _run(
        [
            tools["cmake"],
            "-S",
            str(app),
            "-B",
            str(build),
            f"-DCMAKE_PREFIX_PATH={prefix}",
            f"-DCMAKE_CXX_COMPILER={tools['c++']}",
            *flags,
        ]
    )
    _run([tools["cmake"], "--build", str(build)])
    dynamic = _run([tools["readelf"], "-dW", str(build / "my_app")])
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("LD_PRELOAD", None)
    output = _run([str(build / "my_app")], env=env)
    return dynamic, output


def _assert_consumer(dynamic, output):
    needed = re.findall(r"\(NEEDED\).*\[([^]]+)\]", dynamic)
    assert "libexecutorch_backend_tensorrt.so" in needed, "missing delegate dependency"
    assert "libunrelated.so" not in needed, "retained unrelated dependency"
    assert "(RUNPATH)" in dynamic, "missing RUNPATH"
    assert "(RPATH)" not in dynamic, "unexpected RPATH"
    assert output.strip() == "executorch_backend_tensorrt", "wrong static initializers"


@pytest.mark.parametrize("old_dtags", [False, True])
def test_installed_cmake_consumer(tmp_path, linker_tools, old_dtags):
    _assert_consumer(
        *_installed_consumer(tmp_path, linker_tools, _CONFIG.read_text(), old_dtags)
    )


@pytest.mark.parametrize(
    "removed", ["retention", "pop_order", "pop_state", "new_dtags"]
)
def test_installed_cmake_consumer_rejects_removed_guards(
    tmp_path, linker_tools, removed
):
    config = _CONFIG.read_text()
    retention = (
        '"LINKER:--push-state,--no-as-needed,'
        '${TORCHTRT_EXECUTORCH_BACKEND_LIBRARY},--pop-state"'
    )
    assert config.count(retention) == 1
    if removed == "retention":
        config = config.replace(retention, "")
        message = "missing delegate dependency"
    elif removed == "pop_order":
        config = config.replace(
            retention,
            '"LINKER:--push-state,--no-as-needed,--pop-state,'
            '${TORCHTRT_EXECUTORCH_BACKEND_LIBRARY}"',
        )
        message = "missing delegate dependency"
    elif removed == "pop_state":
        config = config.replace(retention, retention.replace(",--pop-state", ""))
        message = "retained unrelated dependency"
    else:
        assert config.count("--enable-new-dtags,") == 1
        config = config.replace("--enable-new-dtags,", "")
        message = "missing RUNPATH"
    dynamic, output = _installed_consumer(tmp_path, linker_tools, config, True)
    with pytest.raises(AssertionError, match=message):
        _assert_consumer(dynamic, output)


def _mixed_example(sample):
    if sample == "readme":
        text = (_CONFIG.parents[1] / "README.md").read_text()
        blocks = re.findall(r"```cmake\n(.*?)```", text, re.DOTALL)
        assert len(blocks) == 1
        return blocks[0]
    lines = _CONFIG.read_text().splitlines()
    start = next(
        i
        for i, line in enumerate(lines)
        if line.startswith("#   find_package(executorch")
    )
    commands = []
    for line in lines[start:]:
        if not line.startswith("#   "):
            break
        commands.append(line[4:])
    return "\n".join(commands)


@pytest.mark.parametrize("sample", ["readme", "config"])
@pytest.mark.parametrize("removed", [None, "backend_cuda", "kernels_optimized"])
def test_documented_mixed_consumer(tmp_path, linker_tools, sample, removed):
    example = _mixed_example(sample)
    if removed:
        assert example.count(f"executorch::{removed}") == 1
        example = example.replace(f"executorch::{removed}", "")
    dynamic, output = _installed_consumer(
        tmp_path, linker_tools, _CONFIG.read_text(), False, example
    )
    needed = re.findall(r"\(NEEDED\).*\[([^]]+)\]", dynamic)
    assert "libexecutorch_backend_tensorrt.so" in needed
    assert "libunrelated.so" not in needed
    assert "executorch_backend_tensorrt" in output.splitlines()
    assert "unrelated" not in output.splitlines()
    for name in ("backend_cuda", "kernels_optimized"):
        assert (f"libexecutorch_{name}.so" in needed) is (removed != name)
        assert (f"executorch_{name}" in output.splitlines()) is (removed != name)


@pytest.mark.unit
def test_the_config_refuses_a_target_it_did_not_create() -> None:
    """The in-tree build defines the same target name, as an interface over a private static copy.

    Reusing whatever is already there let a project that pulls in the in-tree target and then calls
    find_package link the private copy and never touch the wheel's shared library, with only a
    status message to say so. Only an imported shared library can be the target this config made.
    """
    config = _CONFIG.read_text(encoding="utf-8")
    assert "get_target_property" in config, config
    assert 'STREQUAL "SHARED_LIBRARY"' in config, config
    assert "FATAL_ERROR" in config, config


@pytest.mark.unit
def test_the_embedded_run_path_can_be_turned_off() -> None:
    """The run path is this machine's absolute path and it reaches every consumer binary.

    That suits building against an installed wheel, which is what this package is for, and ruins
    anything redistributable, so a consumer has to be able to decline it.
    """
    config = _CONFIG.read_text(encoding="utf-8")
    assert "TORCHTRT_EXECUTORCH_EMBED_RUNPATH" in config, config
    # Honoured rather than declared with option(), which inside a package config creates a cache
    # entry in the consumer's project and can override a plain variable they already set.
    assert "if(NOT DEFINED TORCHTRT_EXECUTORCH_EMBED_RUNPATH)" in config, config
    assert "option(TORCHTRT_EXECUTORCH_EMBED_RUNPATH" not in config, config
    # The run path lives inside the branch the switch controls, so turning it off omits it.
    branch = config.split("if(TORCHTRT_EXECUTORCH_EMBED_RUNPATH)", 1)[1].split(
        "else()", 1
    )[0]
    assert "INTERFACE_LINK_OPTIONS" in branch, branch[:300]
    # Turning it off does not leave a binary free of this machine's paths: linking an imported
    # library makes CMake record its directory as a run path anyway. Measured on Linux, a consumer
    # with no link options from this package still carries DT_RUNPATH, and only
    # CMAKE_SKIP_BUILD_RPATH drops it. The off branch has to say so, or the switch reads as doing
    # more than it does.
    off_branch = config.split("else()", 1)[1]
    assert "CMAKE_SKIP_BUILD_RPATH" in off_branch, off_branch[:400]


@pytest.mark.unit
def test_the_collision_guard_checks_which_library_the_target_points_at() -> None:
    """Matching the type is not identifying the target.

    A shared imported target of the same name pointing at a different file passes a type check, and
    the consumer then links that file while believing it linked the one this package found.
    """
    config = _CONFIG.read_text(encoding="utf-8")
    assert "IMPORTED_LOCATION" in config, config
    assert (
        'STREQUAL "${TORCHTRT_EXECUTORCH_BACKEND_LIBRARY}"' in config
    ), "the guard does not compare against the library this package found"
    # Both guards, so a same-type impostor is still refused.
    assert config.count("FATAL_ERROR") >= 2, config
