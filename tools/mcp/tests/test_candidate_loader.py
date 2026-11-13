# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import sys

import pytest
import torch
from torch_tensorrt.dynamo.lowering import TORCH_TRT_DECOMPOSITIONS


def test_baseline_loads_no_candidate(policy):
    assert policy.load_candidate(None) is None


def test_candidate_is_required_only_for_lowering_jobs(policy):
    request = {
        "timeout_seconds": 5.0,
        "model": "model",
        "inputs": "inputs",
        "graph_transformation_candidate_id": "a" * 32,
        "model_parameters": {},
        "input_parameters": {},
    }

    with pytest.raises(ValueError, match="regular file"):
        policy.validate_request("compile", request)

    policy.validate_model_request(request)


def test_generated_candidate_loads_once(policy):
    generated = policy.output_root / ".graph_transformation_candidates"
    graph_transformation_candidate_id = "a" * 32
    (
        generated
        / f"graph_transformation_candidate_{graph_transformation_candidate_id}.py"
    ).write_text(
        "IMPORT_COUNT = globals().get('IMPORT_COUNT', 0) + 1\n",
        encoding="utf-8",
    )

    first = policy.load_candidate(graph_transformation_candidate_id)
    second = policy.load_candidate(graph_transformation_candidate_id)

    assert first is second
    assert first.IMPORT_COUNT == 1
    sys.modules.pop(first.__name__, None)


def test_generated_candidate_registers_custom_decomposition(policy):
    generated = policy.output_root / ".graph_transformation_candidates"
    graph_transformation_candidate_id = "d" * 32
    candidate = generated / (
        f"graph_transformation_candidate_{graph_transformation_candidate_id}.py"
    )
    candidate.write_text(
        """
import torch
from torch_tensorrt.dynamo.lowering import (
    TORCH_TRT_DECOMPOSITIONS,
    register_torch_trt_decomposition,
)

@register_torch_trt_decomposition(
    torch.ops.aten.diff.default,
    registry=TORCH_TRT_DECOMPOSITIONS,
)
def decompose_diff(self, n=1, dim=-1, prepend=None, append=None):
    assert n == 1 and prepend is None and append is None
    return (
        torch.ops.aten.slice.Tensor(self, dim, 1)
        - torch.ops.aten.slice.Tensor(self, dim, 0, -1)
    )
""",
        encoding="utf-8",
    )
    op = torch.ops.aten.diff.default
    assert op not in TORCH_TRT_DECOMPOSITIONS

    module = policy.load_candidate(graph_transformation_candidate_id)
    try:
        decomposition = TORCH_TRT_DECOMPOSITIONS[op]
        value = torch.tensor([[1.0, 4.0, 9.0]])
        torch.testing.assert_close(
            decomposition(value),
            torch.diff(value, dim=-1),
        )
    finally:
        TORCH_TRT_DECOMPOSITIONS.pop(op, None)
        sys.modules.pop(module.__name__, None)


@pytest.mark.parametrize(
    "graph_transformation_candidate_id",
    ["", "../candidate", "A" * 32, "a" * 31, "a" * 33, 123],
)
def test_graph_transformation_candidate_id_is_strictly_validated(
    policy, graph_transformation_candidate_id
):
    with pytest.raises(ValueError, match="32 lowercase hexadecimal"):
        policy.load_candidate(graph_transformation_candidate_id)


def test_candidate_symlink_is_rejected(policy, tmp_path):
    generated = policy.output_root / ".graph_transformation_candidates"
    graph_transformation_candidate_id = "b" * 32
    outside = tmp_path.parent / "outside_candidate.py"
    outside.write_text("VALUE = 1\n", encoding="utf-8")
    (
        generated
        / f"graph_transformation_candidate_{graph_transformation_candidate_id}.py"
    ).symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        policy.load_candidate(graph_transformation_candidate_id)
