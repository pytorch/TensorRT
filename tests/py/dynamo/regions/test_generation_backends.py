# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Backend contracts and reconstruction across the evaluator process boundary."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from torch_tensorrt.dynamo.region_compilation._generation._backends import get_backend
from torch_tensorrt.dynamo.region_compilation._generation._candidate import (
    make_registration,
    validate_candidate,
)
from torch_tensorrt.dynamo.region_compilation._generation._cuda import _validate_ptx

CUDA_SOURCE = """extern "C" __global__ void generated(const float* arg0, float* out0) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 257) out0[index] = arg0[index] + 1.0f;
}
"""


def candidate(**updates):
    return (
        dict(
            source=CUDA_SOURCE,
            kernel_name="generated",
            grid=[3],
            block=[128],
            shared_mem_bytes=0,
        )
        | updates
    )


def test_backend_schemas_only_expose_applicable_launch_fields():
    triton = get_backend("triton").proposal_schema
    cutile = get_backend("cutile").proposal_schema
    cuda = get_backend("cuda_cpp").proposal_schema
    assert "num_warps" in triton["properties"]
    assert "num_warps" not in cutile["properties"]
    assert "constants" in cutile["properties"]
    assert set(cuda["properties"]) == set(candidate())
    cuda["properties"].clear()
    assert get_backend("cuda_cpp").proposal_schema["properties"]


def test_backend_extension_is_resolved_in_a_fresh_interpreter(tmp_path, monkeypatch):
    (tmp_path / "test_kernel_backend.py").write_text(
        "from torch_tensorrt.dynamo.region_compilation._generation._backends import get_backend\n"
        "backend = get_backend('triton')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    name = "test_kernel_backend:backend"
    assert get_backend(name).name == "triton"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from torch_tensorrt.dynamo.region_compilation._generation._backends import get_backend; "
            f"assert get_backend({name!r}).name == 'triton'",
        ],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "name", ["cuda_python", "numba", "invalid:path.extra", "a:b:c", None]
)
def test_unknown_backend_fails_with_actionable_contract(name):
    with pytest.raises(ValueError, match="backend|Backend"):
        get_backend(name)


def test_extension_must_implement_backend_contract(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "invalid_kernel_backend", SimpleNamespace(backend=object())
    )
    with pytest.raises(ValueError, match="GenerationBackend"):
        get_backend("invalid_kernel_backend:backend")


@pytest.mark.parametrize(
    "updates",
    [
        {"grid": [False]},
        {"grid": [1, 65536]},
        {"block": [1024, 2]},
        {"block": [1, 1, 65]},
        {"block": [0]},
        {"shared_mem_bytes": True},
        {"shared_mem_bytes": 49153},
        {"num_warps": 4},
        {"source": CUDA_SOURCE.replace("const float* arg0", "int arg0")},
        {"source": CUDA_SOURCE.replace("float* out0", "const float* out0")},
        {"source": CUDA_SOURCE.replace("arg0, float* out0", "out0, float* arg0")},
        {"source": "#include <stdio.h>\n" + CUDA_SOURCE},
        {"source": "#define float int\n" + CUDA_SOURCE},
        {"source": CUDA_SOURCE + CUDA_SOURCE.replace("generated", "second")},
    ],
)
def test_cuda_rejects_unsupported_abi_and_launch_metadata(updates):
    with pytest.raises(ValueError):
        validate_candidate(candidate(**updates), "cuda_cpp", 1, 1)


def test_cuda_normalized_launch_lists_are_independent():
    original = candidate()
    normalized = validate_candidate(original, "cuda_cpp", 1, 1)
    normalized["block"].append(1)
    assert original["block"] == [128]


def test_cuda_compiled_entry_contract_is_checked():
    ptx = b".visible .entry generated(.param .u64 .ptr .align 1 a, .param .u64 b) {}"
    _validate_ptx(ptx, "generated", 2, (128, 1, 1))
    for invalid in (
        ptx.replace(b".u64 b", b".u32 b"),
        ptx.replace(b"generated", b"other"),
        ptx + b"\n.visible .entry second() {}",
        ptx.replace(b") {", b") .reqntid 64 {"),
    ):
        with pytest.raises(ValueError):
            _validate_ptx(invalid, "generated", 2, (128, 1, 1))


def test_cuda_pointer_dtype_must_match_captured_boundary(tmp_path):
    spec = {"dtype": "float16", "shape": [257], "device": "cuda:0"}
    with pytest.raises(ValueError, match="boundary dtype"):
        make_registration(candidate(), "cuda_cpp", [spec], [spec], tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_registration_compiles_real_source_and_derives_trusted_launch(
    tmp_path, monkeypatch
):
    pytest.importorskip("cuda.core")
    import tensorrt.plugin as trtp

    from torch_tensorrt.kernels import KernelSpec

    spec = {"dtype": "float32", "shape": [257], "device": "cuda:0"}
    registration = make_registration(candidate(), "cuda_cpp", [spec], [spec], tmp_path)
    assert isinstance(registration, KernelSpec)
    monkeypatch.setattr(trtp, "KernelLaunchParams", SimpleNamespace)
    params, extras = registration.geometry.fn([], [], None)
    assert params.grid_x == 3
    assert params.block_x == 128
    assert len(extras) == 0
    assert len(list(tmp_path.glob("*.cu"))) == 1
