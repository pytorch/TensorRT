# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise generated source through the isolated evaluator and final TRT engine."""

import json

import pytest
import torch

import torch_tensorrt
from torch_tensorrt import region_annotations
from torch_tensorrt.dynamo.region_compilation import get_region_records
from torch_tensorrt.dynamo.region_compilation._generation import _providers

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch_tensorrt.ENABLED_FEATURES.qdp_plugin,
    reason="CUDA and TensorRT QDP support are required",
)

SOURCES = {
    "triton": """import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice

@triton.jit
def generated(arg0, out0, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    dtype = arg0.dtype.element_ty
    values = tl.load(arg0 + offsets, offsets < N, other=0.0).to(tl.float32)
    scaled = libdevice.mul_rn(values, 0.75).to(dtype).to(tl.float32)
    shifted = libdevice.add_rn(scaled, -0.25).to(dtype)
    tl.store(out0 + offsets, tl.where(shifted < 0, 0.0, shifted), offsets < N)
""",
    "cutile": """import cuda.tile as ct

@ct.kernel
def generated(arg0, out0, N: ct.Constant[int], BLOCK: ct.Constant[int]):
    dtype = arg0.dtype
    values = ct.load(arg0, index=(ct.bid(0),), shape=(BLOCK,)).astype(ct.float32)
    scaled = ct.mul(values, 0.75, rounding_mode=ct.RoundingMode.RN).astype(dtype).astype(ct.float32)
    shifted = ct.add(scaled, -0.25, rounding_mode=ct.RoundingMode.RN).astype(dtype)
    ct.store(out0, index=(ct.bid(0),), tile=ct.where(shifted < 0, 0.0, shifted))
""",
}


def _cuda_source(dtype):
    scalar, load, store = {
        torch.float32: ("float", "float", "float"),
        torch.float16: ("__half", "__half2float", "__float2half_rn"),
        torch.bfloat16: ("__nv_bfloat16", "__bfloat162float", "__float2bfloat16_rn"),
    }[dtype]
    return f"""#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ void generated(const {scalar}* arg0, {scalar}* out0) {{
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset < 771) {{
        float value = {load}(arg0[offset]);
        float scaled = {load}({store}(__fmul_rn(value, 0.75f)));
        float shifted = {load}({store}(__fadd_rn(scaled, -0.25f)));
        out0[offset] = {store}(shifted < 0.0f ? 0.0f : shifted);
    }}
}}
"""


@pytest.mark.parametrize("backend", ["triton", "cutile", "cuda_cpp"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_generated_kernel_evaluation_engine_and_cache(
    backend, dtype, tmp_path, monkeypatch
):
    pytest.importorskip(
        {"triton": "triton", "cutile": "cuda.tile", "cuda_cpp": "cuda.bindings"}[
            backend
        ]
    )
    if backend == "cutile":
        try:
            from cuda.tile.compilation import export_kernel  # noqa: F401

            from torch_tensorrt.kernels._cutile import validate_cutile_toolchain

            validate_cutile_toolchain("generated kernel integration tests")
        except (ImportError, RuntimeError) as error:
            pytest.skip(str(error))
    tolerance = None if dtype == torch.float32 else 0
    generator = region_annotations.KernelGenerator(
        backend=backend,
        cache_dir=tmp_path,
        max_attempts=1,
        timeout_s=180,
        candidate_timeout_s=90,
        num_tests=2,
        benchmark_iters=3,
        rtol=tolerance,
        atol=tolerance,
    )
    requests = []

    def author(**request):
        requests.append(request)
        if backend == "cuda_cpp":
            return dict(
                source=_cuda_source(dtype),
                kernel_name="generated",
                grid=[7],
                block=[128],
                shared_mem_bytes=0,
            )
        return dict(
            source=SOURCES[backend],
            kernel_name="generated",
            constants=[{"name": "N", "value": 771}, {"name": "BLOCK", "value": 128}],
            grid=[7],
            num_warps=4 if backend == "triton" else None,
            num_stages=None,
        )

    monkeypatch.setattr(_providers, "propose", author)

    class Model(torch.nn.Module):
        def forward(self, x):
            x = x + 1
            with region_annotations.fuse_in_plugin(
                generator=generator, name="generated_activation"
            ):
                y = torch.relu(x * 0.75 - 0.25)
            return y - 0.5

    model = Model().eval()
    inputs = [torch.linspace(-2, 2, 771, device="cuda", dtype=dtype).reshape(3, 257)]
    if dtype != torch.float32:
        region_input = inputs[0] + 1
        rounded = torch.relu(region_input * 0.75 - 0.25)
        unrounded = torch.relu(region_input.float() * 0.75 - 0.25).to(dtype)
        assert torch.any(
            rounded != unrounded
        ), "Exercise observable intermediate rounding"
    with torch.inference_mode():
        # An ordinary eager invocation must not invoke the agent.
        expected = model(*inputs)
        assert not requests
        compiled = torch_tensorrt.compile(
            model,
            inputs=inputs,
            ir="dynamo",
            strict=True,
            min_block_size=1,
            offload_module_to_cpu=False,
        )
        torch.testing.assert_close(
            compiled(*inputs), expected, rtol=tolerance, atol=tolerance
        )
        special = inputs[0].clone()
        special.reshape(-1)[:3] = torch.tensor(
            [float("nan"), float("inf"), -float("inf")], device="cuda"
        )
        torch.testing.assert_close(
            compiled(special),
            model(special),
            equal_nan=True,
            rtol=tolerance,
            atol=tolerance,
        )
    assert len(requests) == 1
    assert "aten.relu" in requests[0]["prompt"]
    records = get_region_records(compiled)
    assert len(records) == 1 and records[0].route == "plugin"
    engine = compiled.get_submodule(records[0].owner)
    assert str(records[0].plugin_op) in engine.get_layer_info()
    manifests = list(tmp_path.glob("*/accepted.json"))
    assert len(manifests) == 1
    accepted = json.loads(manifests[0].read_text())
    assert accepted["report"]["status"] == "passed"
    assert accepted["report"]["latency_ms"] > 0

    def no_second_author(**request):
        pytest.fail("A verified cache hit must not call the agent")

    monkeypatch.setattr(_providers, "propose", no_second_author)
    with torch.inference_mode():
        cached = torch_tensorrt.compile(
            model,
            inputs=inputs,
            ir="dynamo",
            strict=True,
            min_block_size=1,
            offload_module_to_cpu=False,
        )
        torch.testing.assert_close(
            cached(*inputs), expected, rtol=tolerance, atol=tolerance
        )
    checks = list(tmp_path.glob("*/run-*/cache_check.json"))
    assert len(checks) == 1
    assert json.loads(checks[0].read_text())["status"] == "passed"
