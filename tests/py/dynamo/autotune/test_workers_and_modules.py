# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Real worker processes and, on a GPU, module workloads.

On the CPU: worker jobs, their failures and timeouts. On a GPU: a module
workload, a whole session, region kernel generation and the roofline.
"""

import json
from pathlib import Path

import pytest
import torch
import torch_tensorrt
from torch_tensorrt.autotune import (
    Budget,
    NoiseControl,
    Timing,
    TuningSession,
    WorkloadFactory,
)
from torch_tensorrt.autotune._knobs import knobs_by_name, resolve
from torch_tensorrt.autotune._measure import measure_candidate

from ._workloads import create_mlp

WORKLOADS = Path(__file__).with_name("_workloads.py")


def worker_session(tmp_path, factory="create_fake"):
    """A session whose worker jobs the tests run one at a time."""
    return TuningSession(
        WorkloadFactory(f"{WORKLOADS}:{factory}"),
        directory=tmp_path / "s",
        budget=Budget(max_trials=0),
    )


def test_worker_processes_run_jobs_and_report_failures(tmp_path):
    session = worker_session(tmp_path)
    session.directory.mkdir(parents=True)
    reference = session._run_worker(
        {"kind": "reference"}, session.directory / "reference"
    )
    assert reference["status"] == "ok" and reference["workload"]["name"] == "fake"
    assert reference["environment"]["torch"] == torch.__version__
    trial = session._run_worker(
        {
            "kind": "measure",
            "config": {},
            "timing": {"warmup": 0, "rounds": 3, "calls": 1, "profile_calls": 0},
        },
        session.directory / "trial",
    )
    assert trial["status"] == "ok" and trial["accuracy"]["max_abs_error"] == 0
    assert (session.directory / "trial" / "worker.log").is_file()

    broken = worker_session(tmp_path / "broken", factory="missing_factory")
    failed = broken._run_worker({"kind": "describe"}, tmp_path / "broken" / "describe")
    assert failed["status"] == "failed" and failed["stage"] == "setup"
    assert "missing_factory" in failed["error"]


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@requires_cuda
@pytest.mark.parametrize("cudagraphs", [False, True])
def test_module_workload_is_checked_measured_and_profiled(cudagraphs):
    workload = create_mlp()
    knobs = knobs_by_name(workload.knobs())
    result = measure_candidate(
        workload,
        resolve({"runtime.cudagraphs": cudagraphs}, knobs),
        workload.reference(),
        Timing(warmup=1, rounds=3, calls=2, profile_calls=1),
    )
    assert result["status"] == "ok", result
    assert result["accuracy"]["cosine_similarity"] > 0.9999
    assert result["accuracy_after_timing"]["cosine_similarity"] > 0.9999
    assert result["structure"]["trt_engines"] == 1
    assert result["structure"]["torch_partitions"] == 0
    assert result["latency_ms"]["count"] == 3 and result["latency_ms"]["median"] > 0
    profile = result["profile"]
    assert profile["device_ms"] > 0 and profile["kernels"] >= 1
    assert profile["wall_ms"] == result["latency_ms"]["median"]
    assert 0 <= profile["host_overhead_fraction"] <= 1
    assert {hint["name"] for hint in result["bottleneck_hints"]} >= {"precision"}


@requires_cuda
def test_a_session_runs_end_to_end_in_worker_processes(tmp_path):
    session = TuningSession(
        WorkloadFactory(f"{WORKLOADS}:create_mlp"),
        directory=tmp_path / "mlp",
        budget=Budget(max_trials=2, trial_timeout_s=900),
        final_check_pairs=1,
        # One noise run and one head-to-head pair exercise both through real
        # workers. Other tests may share the GPU, so the session neither waits
        # for an idle GPU nor holds this tiny model's jitter to 3%.
        noise_control=NoiseControl(
            noise_runs=1, head_to_head_pairs=1, quiet_gpu_s=0, max_spread=0.5
        ),
    )
    result = session.run()
    trials = sorted((tmp_path / "mlp" / "trials").iterdir())
    assert len(trials) == 3
    for trial in trials:
        assert json.loads((trial / "result.json").read_text())["status"] == "ok"
    assert (tmp_path / "mlp" / "report.md").read_text() == result.report()
    assert result.artifact is not None
    loaded = torch_tensorrt.load(str(result.artifact / "model.ep")).module()
    workload = create_mlp()
    torch.testing.assert_close(
        loaded(*workload.inputs), workload.reference(), rtol=1e-2, atol=1e-2
    )


TRITON_RELU = """import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice

@triton.jit
def generated(arg0, out0, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(arg0 + offsets, offsets < N, other=0.0)
    shifted = libdevice.add_rn(libdevice.mul_rn(values, 0.75), -0.25)
    tl.store(out0 + offsets, tl.where(shifted < 0, 0.0, shifted), offsets < N)
"""


class Activation(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x * 0.75 - 0.25)


class RegionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.act = Activation()

    def forward(self, x):
        return self.act(x + 1) - 0.5


@requires_cuda
@pytest.mark.skipif(
    not torch_tensorrt.ENABLED_FEATURES.qdp_plugin, reason="needs TensorRT QDP"
)
def test_region_knobs_generate_kernels_through_the_kernel_generator(
    tmp_path, monkeypatch
):
    pytest.importorskip("triton")
    from torch_tensorrt.autotune import ModuleWorkload
    from torch_tensorrt.dynamo.region_compilation._generation import _providers
    from torch_tensorrt.region import KernelGenerator, generated_kernels

    requests = []

    def author(**request):
        requests.append(request)
        candidate = {
            "source": TRITON_RELU,
            "kernel_name": "generated",
            "constants": [{"name": "N", "value": 771}, {"name": "BLOCK", "value": 128}],
            "grid": [7],
            "num_warps": 4,
            "num_stages": None,
        }
        return {
            "action": "propose",
            "candidates": [candidate],
            "resource": None,
            "offset": None,
        }

    monkeypatch.setattr(_providers, "propose", author)
    generator = KernelGenerator(
        cache_dir=tmp_path,
        max_attempts=1,
        timeout_s=180,
        candidate_timeout_s=90,
        num_tests=2,
        benchmark_iters=3,
    )
    with pytest.raises(ValueError, match="need a KernelGenerator"):
        ModuleWorkload(RegionModel(), [], regions={"act": "act"})
    workload = ModuleWorkload(
        RegionModel().cuda(),
        [torch.linspace(-2, 2, 771, device="cuda").reshape(3, 257)],
        regions={"act": "act"},
        generator=generator,
        compile_options={"min_block_size": 1},
    )
    knobs = knobs_by_name(workload.knobs())
    timing = Timing(warmup=1, rounds=3, calls=2, profile_calls=0)
    native = measure_candidate(
        workload, resolve({}, knobs), workload.reference(), timing
    )
    assert native["status"] == "ok" and native["structure"]["plugin_regions"] == 0
    assert not requests
    registered = len(generated_kernels())
    generated = measure_candidate(
        workload,
        resolve({"kernels.act": "generated"}, knobs),
        workload.reference(),
        timing,
    )
    assert generated["status"] == "ok", generated
    assert generated["structure"]["plugin_regions"] == 1
    assert generated["accuracy"]["max_abs_error"] < 1e-5
    assert len(requests) == 1 and "aten.relu" in requests[0]["prompt"]
    # The user's module is not rewritten.
    assert "forward" not in vars(workload.module.act)
    # The build recorded the region's new kernel, with its measurement.
    (kernel,) = generated_kernels()[registered:]
    assert (kernel.region, kernel.origin, kernel.backend) == ("act", "new", "triton")
    assert kernel.baseline == "eager" and kernel.speedup > 0
    assert kernel.manifest is not None and kernel.manifest.endswith("accepted.json")


@requires_cuda
def test_roofline_compares_latency_with_the_bandwidth_floor():
    from torch_tensorrt.autotune._measure import memory_roofline, peak_read_gbps

    peak = peak_read_gbps()
    assert 50 < peak < 50000
    # Bytes that take exactly 1 ms at the measured peak, timed at 2 ms.
    line = memory_roofline(peak * 1e6, 2.0)
    assert line["floor_ms"] == pytest.approx(1.0)
    assert line["fraction"] == pytest.approx(0.5)
