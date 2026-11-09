# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted candidates and reports for generation controller tests."""

import os
import time
from dataclasses import replace
from pathlib import Path

import pytest
from torch.fx.experimental.proxy_tensor import make_fx

from torch_tensorrt.dynamo.region_compilation._generation import (
    _cuda,
    _factory,
    _reference,
)
from torch_tensorrt.dynamo.region_compilation._generation._backends import get_backend
from torch_tensorrt.region import RegionError

requires_proc = pytest.mark.skipif(
    not os.path.isdir("/proc/self"), reason="Process states are read from /proc"
)


def process_ended(pid):
    """Whether ``pid`` exited; an orphan's zombie may wait for its reaper."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as stat:
            return stat.read().rsplit(")", 1)[1].split()[0] == "Z"
    except FileNotFoundError:
        return True


def wait_until_ended(pid, timeout_s):
    deadline = time.monotonic() + timeout_s
    while not process_ended(pid):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def require_cuda_cpp_toolchain():
    """Skip unless the driver loads NVRTC's PTX, as CUDA C++ generation requires.

    CI runners can have a driver older than NVRTC; generation refuses that
    toolchain, so tests that compile and load real CUDA C++ skip there.
    """
    pytest.importorskip("cuda.core")
    try:
        _cuda.verify_toolchain("cuda")
    except RegionError as error:
        pytest.skip(str(error))


def save_reference(module, *example_inputs, path):
    """Save an ATen trace of ``module`` as the controller saves a captured region."""
    _reference.save_reference(make_fx(module)(*example_inputs), Path(path))


def validate_candidate(candidate, backend, num_inputs, num_outputs):
    return get_backend(backend).validate_candidate(candidate, num_inputs, num_outputs)


def make_registration(candidate, backend, input_specs, output_specs, directory):
    return get_backend(backend).make_registration(
        candidate, input_specs, output_specs, directory
    )


def _candidate(name):
    return {
        "source": f"""
import triton
import triton.language as tl

@triton.jit
def {name}(arg0, out0, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(arg0 + offsets, offsets < N, other=0.0)
    value = value + value
    tl.store(out0 + offsets, tl.where(value < 0, 0.0, value), offsets < N)
""",
        "kernel_name": name,
        "constants": [{"name": "N", "value": 7}, {"name": "BLOCK", "value": 32}],
        "grid": [1],
        "num_warps": 4,
        "num_stages": None,
    }


def _passed(latency):
    return {
        "status": "passed",
        "measurement": "cuda_graph",
        "latency_ms": latency,
        "reference_latency_ms": 1.0,
        "tests": 6,
    }


def _proposal(*candidates):
    return {
        "action": "propose",
        "candidates": list(candidates),
        "resource": None,
        "offset": None,
    }


def _generate(factory, **changes):
    return _factory.generate_registration(
        factory.graph, replace(factory.config, **changes), region_id="activation"
    )
