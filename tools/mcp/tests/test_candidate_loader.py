# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import sys

import pytest


def test_baseline_loads_no_candidate(policy):
    assert policy.load_candidate(None) is None


def test_candidate_is_required_only_for_lowering_jobs(policy):
    request = {
        "timeout_seconds": 5.0,
        "model": "model",
        "inputs": "inputs",
        "lowering_pass_candidate_id": "a" * 32,
        "model_parameters": {},
        "input_parameters": {},
    }

    with pytest.raises(ValueError, match="regular file"):
        policy.validate_request("compile", request)

    policy.validate_model_request(request)


def test_generated_candidate_loads_once(policy):
    generated = policy.output_root / ".lowering_pass_candidates"
    lowering_pass_candidate_id = "a" * 32
    (generated / f"lowering_pass_candidate_{lowering_pass_candidate_id}.py").write_text(
        "IMPORT_COUNT = globals().get('IMPORT_COUNT', 0) + 1\n",
        encoding="utf-8",
    )

    first = policy.load_candidate(lowering_pass_candidate_id)
    second = policy.load_candidate(lowering_pass_candidate_id)

    assert first is second
    assert first.IMPORT_COUNT == 1
    sys.modules.pop(first.__name__, None)


@pytest.mark.parametrize(
    "lowering_pass_candidate_id",
    ["", "../candidate", "A" * 32, "a" * 31, "a" * 33, 123],
)
def test_lowering_pass_candidate_id_is_strictly_validated(
    policy, lowering_pass_candidate_id
):
    with pytest.raises(ValueError, match="32 lowercase hexadecimal"):
        policy.load_candidate(lowering_pass_candidate_id)


def test_candidate_symlink_is_rejected(policy, tmp_path):
    generated = policy.output_root / ".lowering_pass_candidates"
    lowering_pass_candidate_id = "b" * 32
    outside = tmp_path.parent / "outside_candidate.py"
    outside.write_text("VALUE = 1\n", encoding="utf-8")
    (generated / f"lowering_pass_candidate_{lowering_pass_candidate_id}.py").symlink_to(
        outside
    )

    with pytest.raises(ValueError, match="symlink"):
        policy.load_candidate(lowering_pass_candidate_id)
