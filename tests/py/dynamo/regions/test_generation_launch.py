# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Launch validated candidates directly, as the evaluator's guard-band check does."""

import pytest
import torch

from torch_tensorrt.dynamo.region_compilation import _plugins
from torch_tensorrt.dynamo.region_compilation._generation import _cuda
from torch_tensorrt.dynamo.region_compilation._generation._backends import get_backend

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# 258 elements leave the last 128-element block partial.
SHAPE = (3, 86)
NUMEL = 258
TRITON = """import triton
import triton.language as tl

@triton.jit
def generated(arg0, out0, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(arg0 + offsets, offsets < N)
    tl.store(out0 + offsets, values * 2 + 1, offsets < N)
"""
CUTILE = """import cuda.tile as ct

@ct.kernel
def generated(arg0, out0, BLOCK: ct.Constant[int]):
    values = ct.load(arg0, index=(ct.bid(0),), shape=(BLOCK,))
    ct.store(out0, index=(ct.bid(0),), tile=values * 2 + 1)
"""
CUDA = """extern "C" __global__ void generated(const float* arg0, float* out0) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 258) out0[index] = arg0[index] * 2.0f + 1.0f;
}
"""
CANDIDATES = {
    "triton": dict(
        source=TRITON,
        kernel_name="generated",
        grid=[3],
        constants=[{"name": "N", "value": NUMEL}, {"name": "BLOCK", "value": 128}],
        num_warps=4,
        num_stages=None,
    ),
    "cutile": dict(
        source=CUTILE,
        kernel_name="generated",
        grid=[3],
        constants=[{"name": "BLOCK", "value": 128}],
    ),
    "cuda_cpp": dict(
        source=CUDA, kernel_name="generated", grid=[3], block=[128], shared_mem_bytes=0
    ),
}


def _backend(name):
    if name == "triton":
        pytest.importorskip("triton")
    elif name == "cutile":
        pytest.importorskip("cuda.tile")
        from torch_tensorrt.kernels._cutile import validate_cutile_toolchain

        try:
            validate_cutile_toolchain("generated kernel launch tests")
        except RuntimeError as error:
            pytest.skip(str(error))
    else:
        pytest.importorskip("cuda.core")
    backend = get_backend(name)
    return backend, backend.validate_candidate(CANDIDATES[name], 1, 1)


def _guarded(fill, guard=512):
    """A sentinel-filled buffer and its interior view with the captured shape."""
    buffer = torch.full((guard + NUMEL + guard,), fill, device="cuda")
    return buffer, buffer[guard : guard + NUMEL].view(SHAPE)


@pytest.mark.parametrize("name", list(CANDIDATES))
def test_launch_writes_outputs_on_the_current_stream(name, tmp_path):
    backend, candidate = _backend(name)
    inputs = torch.randn(SHAPE, device="cuda")
    output = torch.empty_like(inputs)
    backend.launch(candidate, [inputs], [output], tmp_path)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, inputs * 2 + 1)
    # Capture fails if the launch synchronizes or uses another stream.
    output.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        backend.launch(candidate, [inputs], [output], tmp_path)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, inputs * 2 + 1)


@pytest.mark.parametrize("name", list(CANDIDATES))
def test_launch_stays_inside_interior_views(name, tmp_path):
    backend, candidate = _backend(name)
    source, inputs = _guarded(0.0)
    inputs.copy_(torch.randn(SHAPE, device="cuda"))
    original = source.clone()
    buffer, output = _guarded(float("nan"))
    backend.launch(candidate, [inputs], [output], tmp_path)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, inputs * 2 + 1)
    assert torch.equal(source, original)
    assert torch.isnan(buffer[:512]).all() and torch.isnan(buffer[-512:]).all()


def test_triton_constants_do_not_collide_with_launch_arguments(tmp_path):
    backend, candidate = _backend("triton")
    for name in ("grid", "num_warps", "debug", "instrumentation_mode"):
        candidate["source"] = TRITON.replace("BLOCK", name)
        candidate["constants"][1]["name"] = name
        if name not in ("grid",):
            with pytest.raises(ValueError, match=f"cannot be named \\['{name}'\\]"):
                backend.validate_candidate(candidate, 1, 1)
            continue
        candidate = backend.validate_candidate(candidate, 1, 1)
        inputs = torch.randn(SHAPE, device="cuda")
        output = torch.empty_like(inputs)
        backend.launch(candidate, [inputs], [output], tmp_path / name)
        torch.cuda.synchronize()
        torch.testing.assert_close(output, inputs * 2 + 1)


def test_unmasked_triton_kernel_writes_past_an_interior_output_view(tmp_path):
    backend, candidate = _backend("triton")
    candidate["source"] = TRITON.replace(", offsets < N)", ")")
    candidate = backend.validate_candidate(candidate, 1, 1)
    source, inputs = _guarded(0.0)
    buffer, output = _guarded(float("nan"))
    backend.launch(candidate, [inputs], [output], tmp_path)
    torch.cuda.synchronize()
    # Three 128-element blocks cover 384 elements: 126 land past the output,
    # where the evaluator's guard band, not another tensor, receives them.
    trailing = buffer[512 + NUMEL :]
    assert not torch.isnan(trailing[:126]).any()
    assert torch.isnan(trailing[126:]).all()
    assert torch.isnan(buffer[:512]).all()


def test_cuda_registration_and_launch_compile_once(tmp_path, monkeypatch):
    backend, candidate = _backend("cuda_cpp")
    from torch_tensorrt.kernels import _nvrtc

    compilations = []
    compile_to_ptx = _nvrtc.compile_to_ptx

    def counted(*args, **kwargs):
        compilations.append(args[1])
        return compile_to_ptx(*args, **kwargs)

    monkeypatch.setattr(_nvrtc, "compile_to_ptx", counted)
    monkeypatch.setattr(_cuda, "_KERNELS", {})
    inputs = torch.randn(SHAPE, device="cuda")
    output = torch.empty_like(inputs)
    spec = _plugins._TensorBoundary(SHAPE, torch.float32, inputs.device)
    backend.make_registration(candidate, [spec], [spec], tmp_path)
    backend.launch(candidate, [inputs], [output], tmp_path)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, inputs * 2 + 1)
    assert compilations == ["generated"]
