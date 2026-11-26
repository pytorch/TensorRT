# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compare native header layouts without constructing an engine or using TorchScript.

Requires a Linux C++ compiler and the TensorRT/CUDA development headers. For a
non-system TensorRT SDK, set TensorRT_ROOT or add its include directory to
CPPFLAGS. The existing py-core CI suite discovers this test.
"""

import itertools
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(sys.platform != "linux", reason="GCC/Clang layout probe"),
]
_ROOT = Path(__file__).resolve().parents[3]
_PROBE = _ROOT / "tests/core/runtime/runtime_layout_probe.cpp"


def _run(command, **kwargs):
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=120, **kwargs
    )
    assert result.returncode == 0, (
        f"Command failed: {shlex.join(map(str, command))}\n"
        f"{result.stdout}\n{result.stderr}"
    )
    return result.stdout


def test_runtime_layout_independent_of_build_macros(tmp_path):
    from torch.utils.cpp_extension import CUDA_HOME, include_paths

    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or shutil.which(compiler[0]) is None or CUDA_HOME is None:
        pytest.skip("requires a C++ compiler and CUDA development headers")

    command = [
        *compiler,
        "-std=c++20",
        "-O2",
        *shlex.split(os.environ.get("CPPFLAGS", "")),
        *shlex.split(os.environ.get("CXXFLAGS", "")),
        f"-I{_ROOT}",
        *(f"-I{path}" for path in include_paths(device_type="cuda")),
    ]
    if trt_root := os.environ.get("TensorRT_ROOT"):
        command.append(f"-I{Path(trt_root) / 'include'}")

    # Wheels alone need not install the development headers. Skip only that
    # missing prerequisite, not errors compiling Torch-TensorRT's own headers.
    sdk_check = subprocess.run(
        [*command, "-fsyntax-only", "-x", "c++", "-"],
        input="#include <NvInfer.h>\n",
        capture_output=True,
        text=True,
        timeout=120,
    )
    if "NvInfer.h: No such file or directory" in sdk_check.stderr:
        pytest.skip("requires TensorRT development headers (set TensorRT_ROOT)")
    assert sdk_check.returncode == 0, sdk_check.stderr

    layouts = {}
    for nccl, runtime_config in itertools.product((False, True), repeat=2):
        variant = f"nccl_{int(nccl)}_runtime_config_{int(runtime_config)}"
        executable = tmp_path / variant
        # Clear inherited flags first, including the derived NCCL flag. Exercise
        # both the library's build and a consumer without its Bazel definitions.
        defines = [
            "-UUSE_C10D_NCCL",
            "-UENABLE_TRT_NCCL_COLLECTIVES",
            "-UTRT_HAS_IRUNTIME_CONFIG",
        ]
        if nccl:
            defines.append("-DUSE_C10D_NCCL=1")
        if runtime_config:
            defines.append("-DTRT_HAS_IRUNTIME_CONFIG=1")
        _run(
            [
                *command,
                *defines,
                "-Wno-invalid-offsetof",
                _PROBE,
                "-o",
                executable,
            ]
        )
        layouts[variant] = dict(
            line.split("=", 1) for line in _run([executable]).splitlines()
        )

    baseline = layouts["nccl_0_runtime_config_0"]
    for variant, layout in layouts.items():
        assert layout == baseline, f"Native runtime layout differs for {variant}"
