# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted candidates and reports for generation controller tests."""

from dataclasses import replace

from torch_tensorrt.dynamo.region_compilation._generation import _factory
from torch_tensorrt.dynamo.region_compilation._generation._backends import get_backend


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
