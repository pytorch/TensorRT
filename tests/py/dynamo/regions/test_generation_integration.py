# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise generated source through the isolated evaluator and final TRT engine."""

import json
import os

import pytest
import torch

import torch_tensorrt
from torch_tensorrt import region
from torch_tensorrt.dynamo.region_compilation import get_region_records
from torch_tensorrt.dynamo.region_compilation._generation import _providers

from ._generation_utils import require_cuda_cpp_toolchain

pytestmark = pytest.mark.skipif(
    os.name != "posix"
    or not torch.cuda.is_available()
    or not torch_tensorrt.ENABLED_FEATURES.qdp_plugin,
    reason="A POSIX host, CUDA, and TensorRT QDP support are required",
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


def _proposal(candidate):
    return {
        "action": "propose",
        "candidates": [candidate],
        "resource": None,
        "offset": None,
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
        {"triton": "triton", "cutile": "cuda.tile", "cuda_cpp": "cuda.core"}[backend]
    )
    if backend == "cutile":
        try:
            from cuda.tile.compilation import export_kernel  # noqa: F401

            from torch_tensorrt.kernels._cutile import validate_cutile_toolchain

            validate_cutile_toolchain("generated kernel integration tests")
        except (ImportError, RuntimeError) as error:
            pytest.skip(str(error))
    if backend == "cuda_cpp":
        require_cuda_cpp_toolchain()
    tolerance = None if dtype == torch.float32 else 0
    generator = region.KernelGenerator(
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
            return _proposal(
                dict(
                    source=_cuda_source(dtype),
                    kernel_name="generated",
                    grid=[7],
                    block=[128],
                    shared_mem_bytes=0,
                )
            )
        candidate = dict(
            source=SOURCES[backend],
            kernel_name="generated",
            constants=[{"name": "N", "value": 771}, {"name": "BLOCK", "value": 128}],
            grid=[7],
        )
        if backend == "triton":
            candidate.update(num_warps=4, num_stages=None)
        return _proposal(candidate)

    monkeypatch.setattr(_providers, "propose", author)

    class Model(torch.nn.Module):
        def forward(self, x):
            x = x + 1
            with region.fused_kernel(generator=generator, name="generated_activation"):
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
    # A cache-only build, as CI would run it, reuses the revalidated kernel.
    monkeypatch.setenv("TORCH_TENSORRT_REGION_GENERATION", "0")
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


def test_cache_only_compilation_fails_before_any_agent_or_worker(tmp_path, monkeypatch):
    pytest.importorskip("triton")

    def no_author(**request):
        pytest.fail("Cache-only compilation must not call the agent")

    monkeypatch.setattr(_providers, "propose", no_author)
    generator = region.KernelGenerator(cache_dir=tmp_path, allow_generation=False)

    class Model(torch.nn.Module):
        def forward(self, x):
            with region.fused_kernel(generator=generator, name="uncached"):
                return torch.relu(x * 0.75 - 0.25)

    with pytest.raises(region.RegionError, match="disabled by allow_generation=False"):
        torch_tensorrt.compile(
            Model().eval(),
            inputs=[torch.randn(3, 257, device="cuda")],
            ir="dynamo",
            strict=True,
            min_block_size=1,
            offload_module_to_cpu=False,
        )
    (run,) = tmp_path.glob("*/run-*")
    # No evaluation ran: nothing was cached to revalidate.
    assert not list(run.glob("*/feedback.json"))
    assert not (run / "cache_check.json").exists()


def test_batched_tuning_selects_a_correct_kernel_inside_tensorrt(tmp_path, monkeypatch):
    pytest.importorskip("triton")
    generator = region.KernelGenerator(
        cache_dir=tmp_path,
        max_attempts=1,
        timeout_s=240,
        candidate_timeout_s=90,
        num_tests=2,
        benchmark_iters=3,
        diagnostics=True,
        search=region.SearchPolicy(
            batch_size=2,
            max_candidates=3,
            finalists=2,
            tuning={"num_warps": (4, 8)},
        ),
    )
    candidates = []
    for scale in (0.5, 0.75):
        candidates.append(
            dict(
                source=SOURCES["triton"].replace("0.75", str(scale)),
                kernel_name="generated",
                grid=[7],
                constants=[
                    {"name": "N", "value": 771},
                    {"name": "BLOCK", "value": 128},
                ],
                num_warps=32,
                num_stages=None,
            )
        )
    requests = []

    def author(**request):
        requests.append(request)
        return {"action": "propose", "candidates": candidates}

    monkeypatch.setattr(_providers, "propose", author)

    class Model(torch.nn.Module):
        def forward(self, x):
            with region.fused_kernel(generator=generator, name="batched_activation"):
                return torch.relu(x * 0.75 - 0.25)

    model = Model().eval()
    inputs = [torch.linspace(-2, 2, 771, device="cuda").reshape(3, 257)]
    with torch.inference_mode():
        compiled = torch_tensorrt.compile(
            model,
            inputs=inputs,
            ir="dynamo",
            strict=True,
            min_block_size=1,
            offload_module_to_cpu=False,
        )
        torch.testing.assert_close(compiled(*inputs), model(*inputs))
    assert len(requests) == 1
    accepted = json.loads(next(tmp_path.glob("*/accepted.json")).read_text())
    assert accepted["candidate"]["num_warps"] == 4
    assert accepted["validation"]["status"] == "passed"
    feedback = [
        json.loads(path.read_text())
        for path in tmp_path.glob("*/run-*/attempt-*/candidate-*/feedback.json")
    ]
    # The fourth variant exceeds max_candidates and is recorded, not evaluated.
    refused = [report for report in feedback if report.get("stage") == "budget"]
    reports = [report for report in feedback if report.get("stage") != "budget"]
    assert len(refused) == 1 and len(reports) == 3
    assert sum(report["status"] == "passed" for report in reports) == 1
    # The search screens by direct launches; the finalist is proven in an engine.
    passed = next(report for report in reports if report["status"] == "passed")
    assert passed["execution"] == "launch"
    assert accepted["validation"]["execution"] == "engine"
    assert all(
        report["diagnostics"]["candidate"]["compilation_ms"] > 0 for report in reports
    )
    records = get_region_records(compiled)
    assert len(records) == 1 and records[0].route == "plugin"
    assert (
        str(records[0].plugin_op)
        in compiled.get_submodule(records[0].owner).get_layer_info()
    )


# relu(x * 0.75 - 0.25) that multiplies by the mask: -inf * 0 gives NaN, not 0.
MASK_MULTIPLYING_TRITON = """import triton
import triton.language as tl

@triton.jit
def generated(arg0, out0, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    shifted = tl.load(arg0 + offsets, offsets < N, other=0.0) * 0.75 - 0.25
    tl.store(out0 + offsets, shifted * (shifted > 0), offsets < N)
"""


def test_real_validation_batches_reject_a_kernel_that_passes_synthetic_inputs(
    tmp_path, monkeypatch
):
    pytest.importorskip("triton")
    candidate = dict(
        source=MASK_MULTIPLYING_TRITON,
        kernel_name="generated",
        constants=[{"name": "N", "value": 771}, {"name": "BLOCK", "value": 128}],
        grid=[7],
        num_warps=4,
        num_stages=None,
    )
    monkeypatch.setattr(_providers, "propose", lambda **request: _proposal(candidate))
    # Real batches carry -inf, as masked scores do; synthetic inputs without the
    # nonfinite checks never do.
    masked = torch.randn(3, 257)
    masked[1, 5:9] = -float("inf")

    def compile(cache, validation_inputs):
        generator = region.KernelGenerator(
            cache_dir=tmp_path / cache,
            max_attempts=1,
            timeout_s=240,
            candidate_timeout_s=90,
            num_tests=2,
            benchmark_iters=3,
            check_nonfinite=False,
            validation_inputs=validation_inputs,
        )

        class Model(torch.nn.Module):
            def forward(self, x):
                x = x + 1
                with region.fused_kernel(generator=generator, name="masked"):
                    y = torch.relu(x * 0.75 - 0.25)
                return y - 0.5

        model = Model().eval()
        with torch.inference_mode():
            compiled = torch_tensorrt.compile(
                model,
                inputs=[torch.randn(3, 257, device="cuda")],
                ir="dynamo",
                strict=True,
                min_block_size=1,
                offload_module_to_cpu=False,
            )
        return model, compiled

    model, compiled = compile("synthetic", None)
    with torch.inference_mode():
        assert compiled(masked.cuda()).isnan().any()
        assert not model(masked.cuda()).isnan().any()
    with pytest.raises(region.RegionError, match="no valid kernel"):
        compile("sampled", [torch.randn(3, 257), masked])
    (feedback,) = (tmp_path / "sampled").glob("*/run-*/attempt-1/feedback.json")
    report = json.loads(feedback.read_text())
    assert report["stage"] == "correctness"
    assert "Output 0 is wrong for case 'sample 2': got nan at index (1, 5)" in (
        report["error"]
    )
    assert not list((tmp_path / "sampled").glob("*/accepted.json"))
