# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Search budgets and backend-owned tuning domains, without model/GPU calls."""

import copy
import json
from dataclasses import FrozenInstanceError, asdict, replace
from types import SimpleNamespace

import pytest

from torch_tensorrt.dynamo.region_compilation._generation import _backends, _factory
from torch_tensorrt.dynamo.region_compilation._generation._backends import (
    candidate_variants,
    validate_tuning_space,
)
from torch_tensorrt.region import KernelGenerator, SearchPolicy

from ._generation_utils import validate_candidate


def test_search_policy_defaults_preserve_single_proposal_behavior():
    first, second = KernelGenerator(), KernelGenerator()
    assert first.search == SearchPolicy()
    assert first.search is not second.search
    assert first.search.tuning is not second.search.tuning
    assert first.search.batch_size == first.search.finalists == 1
    assert first.search.max_candidates is None
    assert first.search.max_no_improvement is None
    assert first.search.screening == "launch"
    assert first.diagnostics is False
    assert hash(first) == hash(second)
    with pytest.raises(FrozenInstanceError):
        first.search.batch_size = 2


def test_search_policy_snapshots_nested_user_tuning():
    grid = [3]
    options = {"grid": [grid, [6]], "constants.BLOCK": (64, 128)}
    policy = SearchPolicy(tuning=options)
    grid.append(2)
    options["grid"].append([9])
    options["constants.BLOCK"] = (256,)
    assert policy.tuning == {"grid": ([3], [6]), "constants.BLOCK": (64, 128)}


@pytest.mark.parametrize(
    "options,error",
    [
        ({"batch_size": True}, TypeError),
        ({"batch_size": 0}, ValueError),
        ({"max_candidates": False}, TypeError),
        ({"max_candidates": 0}, ValueError),
        ({"max_no_improvement": 1.5}, TypeError),
        ({"max_no_improvement": 0}, ValueError),
        ({"max_variants": None}, TypeError),
        ({"max_variants": 129}, ValueError),
        ({"finalists": True}, TypeError),
        ({"finalists": 0}, ValueError),
        ({"resume": "yes"}, TypeError),
        ({"screening": True}, TypeError),
        ({"screening": "eager"}, ValueError),
        ({"tuning": []}, TypeError),
        ({"tuning": {"num_warps": ()}}, ValueError),
        ({"tuning": {"num_warps": "4"}}, TypeError),
        ({"tuning": {"": (4,)}}, ValueError),
        ({"tuning": {"grid": ({(1,): [1]},)}}, TypeError),
        ({"tuning": {"num_warps": (float("nan"),)}}, ValueError),
        ({"tuning": {"num_warps": (float("inf"),)}}, ValueError),
        ({"tuning": {"grid": ({"nested": float("inf")},)}}, ValueError),
        ({"tuning": {"grid": (object(),)}}, TypeError),
        ({"tuning": {"grid": (lambda: 1,)}}, TypeError),
    ],
)
def test_invalid_search_configuration_is_rejected(options, error):
    with pytest.raises(error):
        SearchPolicy(**options)


def test_recursive_tuning_input_is_rejected():
    values = []
    values.append(values)
    with pytest.raises(ValueError, match="Circular reference"):
        SearchPolicy(tuning={"grid": (values,)})


def test_generator_requires_typed_search_and_boolean_diagnostics():
    with pytest.raises(TypeError, match="SearchPolicy"):
        KernelGenerator(search={"batch_size": 2})
    with pytest.raises(TypeError, match="diagnostics must be a bool"):
        KernelGenerator(diagnostics=1)


@pytest.mark.parametrize(
    "backend,tuning",
    [
        ("triton", {"source": ("replacement",)}),
        ("triton", {"kernel_name": ("replacement",)}),
        ("triton", {"unknown": (1,)}),
        ("cutile", {"num_warps": (4,)}),
        ("cuda_cpp", {"constants.BLOCK": (128,)}),
        ("triton", {"constants.__class__": (128,)}),
        ("triton", {"constants.for": (128,)}),
        ("triton", {"constants.BLOCK.more": (128,)}),
        ("triton", {"constants": ([],), "constants.BLOCK": (128,)}),
        ("triton", {"num_warps": (True,)}),
        ("triton", {"grid": ("lambda: 1",)}),
        ("triton", {"grid": ([0],)}),
        ("cuda_cpp", {"shared_mem_bytes": (-1,)}),
        # The schema carries the limits the backend validators enforce.
        ("triton", {"num_warps": (3, 6)}),
        ("triton", {"num_warps": (4.0, 8.0)}),
        ("triton", {"num_stages": (0,)}),
        ("triton", {"grid": ([2**31],)}),
        ("cuda_cpp", {"block": ([2048],)}),
        ("cuda_cpp", {"shared_mem_bytes": (49153,)}),
    ],
)
def test_tuning_fields_and_values_follow_selected_backend_schema(backend, tuning):
    with pytest.raises(ValueError):
        validate_tuning_space(backend, tuning)


def test_tuning_accepts_every_value_the_validators_accept():
    tuning = {
        "num_warps": (None, 1, 2, 4, 8, 16, 32),
        "num_stages": (None, 1, 4),
        "grid": ([2**31 - 1, 65535, 65535],),
    }
    assert validate_tuning_space("triton", tuning) == {
        name: list(choices) for name, choices in sorted(tuning.items())
    }


def test_variants_override_every_configured_knob_without_mutating_the_proposal():
    candidate = {
        "source": "source",
        "kernel_name": "kernel",
        "grid": [99],
        "num_warps": 32,
    }
    variants = candidate_variants(
        candidate, "triton", {"num_warps": (4, 8), "grid": ([1], [2])}, limit=3
    )
    assert [(item["grid"], item["num_warps"]) for item in variants] == [
        ([1], 4),
        ([1], 8),
        ([2], 4),
    ]
    assert all(item != candidate for item in variants)
    variants[0]["grid"].append(2)
    assert candidate["grid"] == [99]
    assert variants[1]["grid"] == [1]


def test_named_constants_are_tuned_without_mutating_source_or_other_constants():
    candidate = {
        "source": "source",
        "kernel_name": "kernel",
        "constants": [{"name": "N", "value": 257}, {"name": "BLOCK", "value": 512}],
    }
    variants = candidate_variants(
        candidate, "cutile", {"constants.BLOCK": (64, 128)}, 8
    )
    assert [item["constants"][1]["value"] for item in variants] == [64, 128]
    assert all(item["constants"][0] == {"name": "N", "value": 257} for item in variants)
    assert candidate["constants"][1]["value"] == 512
    with pytest.raises(ValueError, match="declare exactly one"):
        candidate_variants(candidate, "cutile", {"constants.MISSING": (64,)}, 8)


def test_large_product_is_bounded_and_uses_extension_declared_schema(monkeypatch):
    fields = {f"knob{i:02}": {"type": "integer"} for i in range(32)}
    monkeypatch.setattr(
        _backends,
        "get_backend",
        lambda name: SimpleNamespace(proposal_schema={"properties": fields}),
    )
    space = {name: tuple(range(128)) for name in fields}
    variants = candidate_variants({}, "extension:backend", space, 128)
    assert len(variants) == 128
    assert all(set(item) == set(space) for item in variants)
    # Without authored values, declared orders apply; the product is walked by
    # total rank, so every knob takes its second choice before any its third.
    assert variants[0] == dict.fromkeys(space, 0)
    assert [sum(item.values()) for item in variants[:33]] == [0] + [1] * 32
    assert [max(item, key=item.get) for item in variants[1:33]] == sorted(space)[::-1]
    assert len({json.dumps(item, sort_keys=True) for item in variants}) == 128


def test_variants_start_from_the_authored_configuration_and_vary_every_knob_early():
    proposal = {
        "constants": [{"name": "N", "value": 257}, {"name": "BLOCK", "value": 256}],
        "num_stages": 2,
        "num_warps": 4,
    }
    tuning = {
        "constants.BLOCK": (64, 128, 256, 512, 1024),
        "num_stages": (1, 2, 3, 4),
        "num_warps": (1, 2, 4, 8),
    }
    variants = candidate_variants(proposal, "triton", tuning, 8)
    settings = [
        (item["constants"][1]["value"], item["num_stages"], item["num_warps"])
        for item in variants
    ]
    # Nearest choices first, the earlier one on ties: BLOCK 256, 128, 512, ...
    assert settings == [
        (256, 2, 4),
        (256, 2, 2),
        (256, 1, 4),
        (128, 2, 4),
        (256, 2, 8),
        (256, 1, 2),
        (256, 3, 4),
        (128, 2, 2),
    ]
    assert variants[0] == proposal
    # Values match exactly: True is not the choice 1, so declared order remains.
    flagged = dict(proposal, num_warps=True)
    assert candidate_variants(flagged, "triton", {"num_warps": (2, 1)}, 2) == [
        dict(proposal, num_warps=2),
        dict(proposal, num_warps=1),
    ]


def test_empty_space_returns_one_snapshot_and_zero_limit_returns_none():
    proposal = {"source": "source", "grid": [3]}
    variants = candidate_variants(proposal, "triton", {}, 8)
    assert variants == [proposal]
    assert variants[0] is not proposal
    assert candidate_variants(proposal, "triton", {}, 0) == []
    with pytest.raises(ValueError, match="Variant limit"):
        candidate_variants(proposal, "triton", {}, True)


def test_complete_variant_still_passes_through_existing_backend_validator():
    proposal = {
        "source": "import triton\nimport triton.language as tl\n@triton.jit\ndef generated(arg0, out0):\n    pass\n",
        "kernel_name": "generated",
        "grid": [1],
        "constants": [],
        "num_warps": 4,
        "num_stages": None,
    }
    # The schema bounds each grid item; per-axis limits belong to the validator
    # that checks the complete variant.
    variant = candidate_variants(proposal, "triton", {"grid": ([1, 70000],)}, 1)[0]
    with pytest.raises(ValueError, match="grid"):
        validate_candidate(variant, "triton", 1, 1)


@pytest.mark.parametrize(
    "schema,choices",
    [
        (
            {
                "oneOf": [
                    {"type": "integer", "const": 1},
                    {"type": "integer", "const": 2},
                ]
            },
            (1, 2),
        ),
        (
            {
                "oneOf": [
                    {"type": "string", "pattern": "^a"},
                    {"type": "string", "pattern": "^b"},
                ]
            },
            ("alpha", "beta"),
        ),
        ({"oneOf": [{"$ref": "#/$defs/a"}, {"$ref": "#/$defs/b"}]}, (1, "two")),
        ({"type": "integer"}, (1, 2)),
        (True, ({"custom": [1]},)),
    ],
)
def test_extension_shape_check_does_not_reject_valid_overlapping_schema_branches(
    monkeypatch, schema, choices
):
    monkeypatch.setattr(
        _backends,
        "get_backend",
        lambda name: SimpleNamespace(proposal_schema={"properties": {"knob": schema}}),
    )
    assert validate_tuning_space("extension:backend", {"knob": choices}) == {
        "knob": list(choices)
    }


def _task_identity_example():
    spec = {"shape": [257], "dtype": "float32", "device": "cuda:0"}
    return {
        "reference": "full(..., device=device(type='cuda', index=0))",
        "backend": "triton",
        "backend_version": "1",
        "proposal_schema_sha256": "schema",
        "instructions_sha256": "instructions",
        "objective": "candidate",
        "min_speedup": 1.0,
        "inputs": [spec],
        "outputs": [spec],
        "validation": {"rtol": 0.01, "atol": 0.001},
        "compile_options": {"disable_tf32": False, "optimization_level": None},
        "search": asdict(SearchPolicy(tuning={"num_warps": (4, 8)})),
        "budget": {"max_attempts": 4, "timeout_s": 600, "use_cache": True},
        "diagnostics": False,
    }


def test_task_identity_ignores_scheduling_budget_diagnostics_and_device_index():
    task = _task_identity_example()
    original = copy.deepcopy(task)
    other_device = [{**task["inputs"][0], "device": "cuda:1"}]
    changed = {
        **task,
        "search": asdict(
            SearchPolicy(
                batch_size=2,
                max_candidates=20,
                max_no_improvement=3,
                max_variants=4,
                finalists=2,
                resume=True,
                tuning={"num_warps": (4, 8)},
                screening="engine",
            )
        ),
        "budget": {"max_attempts": 8, "timeout_s": 60, "use_cache": False},
        "diagnostics": True,
        "inputs": other_device,
        "outputs": other_device,
        "reference": task["reference"].replace("index=0", "index=1"),
    }
    assert _factory._task_digest(changed) == _factory._task_digest(task)
    assert _factory._task_digest(json.loads(json.dumps(task))) == _factory._task_digest(
        task
    )
    assert task == original


@pytest.mark.parametrize(
    "changes",
    [
        {"search": asdict(SearchPolicy(tuning={"num_warps": (4,)}))},
        {"search": asdict(SearchPolicy())},
        {"backend_version": "2"},
        {"proposal_schema_sha256": "new-schema"},
        {"instructions_sha256": "new-instructions"},
        {"inputs": [{"shape": [513], "dtype": "float32", "device": "cuda:0"}]},
        {"outputs": [{"shape": [257], "dtype": "float16", "device": "cuda:0"}]},
        {"validation": {"rtol": 0.001, "atol": 0.001}},
        {"compile_options": {"disable_tf32": True, "optimization_level": None}},
        {"objective": "eager"},
        {"min_speedup": 1.1},
    ],
)
def test_task_identity_binds_tuning_and_kernel_compatibility(changes):
    task = _task_identity_example()
    assert _factory._task_digest({**task, **changes}) != _factory._task_digest(task)


def _round(*stages):
    return [
        (
            {"result": {"status": "passed"}}
            if stage == "passed"
            else {
                "result": {
                    "status": "error",
                    **({} if stage is None else {"stage": stage}),
                }
            }
        )
        for stage in stages
    ]


@pytest.mark.parametrize(
    "stage,action",
    [
        ("source", "repair_compile"),
        ("compile", "repair_compile"),
        ("engine", "repair_compile"),
        ("correctness", "repair_correctness"),
        ("benchmark", "repair"),
        ("timeout", "repair"),
        ("author", "repair"),
        (None, "repair"),
        ("provider", "finalize"),
        ("setup", "finalize"),
        ("baseline", "finalize"),
        ("budget", "finalize"),
    ],
)
def test_search_action_uses_evaluation_stage_and_stagnation_budget(stage, action):
    config = KernelGenerator(search=SearchPolicy(max_no_improvement=2))
    assert _factory._next_action(_round(stage), False, 1, config) == action
    assert _factory._next_action(_round(stage), False, 2, config) == "finalize"


def test_search_action_distinguishes_initial_search_improvement_and_stagnation():
    config = KernelGenerator()
    passed = _round("passed")
    assert _factory._next_action([], False, 0, config) == "author"
    assert _factory._next_action(passed, True, 0, config) == "optimize"
    assert _factory._next_action(passed, False, 1, config) == "rewrite"
    limited = replace(config, search=SearchPolicy(max_no_improvement=2))
    assert _factory._next_action(passed, False, 2, limited) == "finalize"


@pytest.mark.parametrize(
    "stages,improved,action",
    [
        (("passed", "compile"), True, "optimize"),
        (("compile", "passed"), True, "optimize"),
        (("correctness", "compile"), False, "repair_compile"),
        (("compile", "correctness"), False, "repair_compile"),
        (("benchmark", "correctness", "passed"), False, "repair_correctness"),
        (("passed", "budget"), True, "finalize"),
        (("setup", "passed"), True, "finalize"),
    ],
)
def test_search_action_follows_the_whole_round_not_its_order(stages, improved, action):
    config = KernelGenerator()
    assert _factory._next_action(_round(*stages), improved, 0, config) == action
