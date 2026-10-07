# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""NVRTC artifact handling independent of the installed CUDA driver version."""

from types import SimpleNamespace

import pytest

from torch_tensorrt.kernels import KernelSpec, _derive, _nvrtc


@pytest.mark.parametrize("load_eager_kernel", [False, True])
def test_ptx_is_not_loaded_and_eager_uses_separate_cubin(
    monkeypatch, load_eager_kernel
):
    """A newer PTX ISA must not prevent registration or CUBIN eager loading."""
    device = SimpleNamespace(arch="120", set_current=lambda: current.append(True))
    current = []
    artifacts = []
    kernel = object()

    class Program:
        def __init__(self, source, *, code_type, options):
            assert current == [True]
            assert source == "kernel source" and code_type == "c++"
            assert options == {
                "std": "c++20",
                "arch": "sm_90",
                "include_path": ["/cuda/include"],
            }
            self.compiled = False

        def compile(self, artifact, *, name_expressions):
            assert not self.compiled, "NVRTC requires a fresh Program per artifact"
            assert name_expressions == ("k",)
            self.compiled = True
            artifacts.append(artifact)

            def get_kernel(name):
                assert artifact == "cubin", "PTX loading would invoke the driver JIT"
                assert name == "k"
                return kernel

            return SimpleNamespace(code=b"ptx", get_kernel=get_kernel)

    monkeypatch.setattr(
        _nvrtc,
        "_cuda_core_imports",
        lambda: (lambda: device, Program, lambda **kwargs: kwargs, None, None),
    )
    args = ("kernel source", "k", ["/cuda/include"], "c++20", "sm_90")
    if load_eager_kernel:
        result = _derive._compile_kernel(
            KernelSpec(
                kernel_source=args[0],
                kernel_name=args[1],
                include_paths=args[2],
                compile_std=args[3],
                arch_override=args[4],
            )
        )
    else:
        result = _nvrtc.compile_to_ptx(*args)

    assert result == (b"ptx", device, kernel if load_eager_kernel else None)
    assert artifacts == (["ptx", "cubin"] if load_eager_kernel else ["ptx"])
