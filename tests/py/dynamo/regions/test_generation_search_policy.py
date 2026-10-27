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
from torch_tensorrt.dynamo.region_compilation._generation._candidate import (
    validate_candidate,
)
from torch_tensorrt.region import KernelGenerator, SearchPolicy


def test_search_policy_defaults_preserve_single_proposal_behavior():
    first, second = KernelGenerator(), KernelGenerator()
    assert first.search == SearchPolicy()
    assert first.search is not second.search
    assert first.search.tuning is not second.search.tuning
    assert first.search.batch_size == first.search.finalists == 1
    assert first.search.max_candidates is None
    assert first.search.max_no_improvement is None
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
    "options",
    [
        {"batch_size": True},
        {"batch_size": 0},
        {"batch_size": 33},
        {"max_candidates": False},
        {"max_candidates": 0},
        {"max_candidates": 4097},
        {"max_no_improvement": 1.5},
        {"max_no_improvement": 0},
        {"max_variants": None},
        {"max_variants": 129},
        {"finalists": True},
        {"finalists": 0},
        {"resume": "yes"},
        {"tuning": []},
        {"tuning": {"num_warps": ()}},
        {"tuning": {"num_warps": "4"}},
        {"tuning": {"num_warps": tuple(range(129))}},
        {"tuning": {"": (4,)}},
        {"tuning": {"grid": ({1: [1]},)}},
        {"tuning": {"num_warps": (float("nan"),)}},
        {"tuning": {"num_warps": (float("inf"),)}},
        {"tuning": {"grid": ({"nested": float("inf")},)}},
        {"tuning": {"grid": (object(),)}},
        {"tuning": {"grid": (lambda: 1,)}},
        {"tuning": {"grid": ("x" * 65536,)}},
    ],
)
def test_invalid_search_configuration_is_rejected(options):
    with pytest.raises((TypeError, ValueError)):
        SearchPolicy(**options)


def test_recursive_tuning_input_is_rejected():
    values = []
    values.append(values)
    with pytest.raises(ValueError, match="nesting"):
        SearchPolicy(tuning={"grid": (values,)})


def test_generator_requires_typed_search_and_boolean_diagnostics():
    with pytest.raises(TypeError, match="SearchPolicy"):
        KernelGenerator(search={"batch_size": 2})
    with pytest.raises(ValueError, match="diagnostics"):
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
    ],
)
def test_tuning_fields_and_values_follow_selected_backend_schema(backend, tuning):
    with pytest.raises(ValueError):
        validate_tuning_space(backend, tuning)


def test_variants_override_every_configured_knob_and_do_not_emit_original():
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
    fields = {f"knob{i}": {"type": "integer"} for i in range(16)}
    monkeypatch.setattr(
        _backends,
        "get_backend",
        lambda name: SimpleNamespace(proposal_schema={"properties": fields}),
    )
    space = {name: tuple(range(128)) for name in fields}
    variants = candidate_variants({}, "extension:backend", space, 3)
    assert len(variants) == 3
    assert all(set(item) == set(space) for item in variants)
    assert [item[sorted(fields)[-1]] for item in variants] == [0, 1, 2]


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
    # The schema declares an integer; valid warp counts are enforced by the
    # compiler adapter on the complete variant, not guessed from a scalar alone.
    variant = candidate_variants(proposal, "triton", {"num_warps": (3,)}, 1)[0]
    with pytest.raises(ValueError):
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
        ({"type": "integer"}, (1, 2.0)),
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
    return {
        "backend": "triton",
        "backend_version": "1",
        "proposal_schema_sha256": "schema",
        "instructions_sha256": "instructions",
        "objective": "candidate",
        "min_speedup": 1.0,
        "inputs": [{"shape": [257], "dtype": "float32", "device": "cuda:0"}],
        "validation": {"rtol": 0.01, "atol": 0.001},
        "search": asdict(SearchPolicy(tuning={"num_warps": (4, 8)})),
        "diagnostics": False,
    }


def test_task_identity_ignores_scheduling_and_diagnostics_without_mutating_task():
    task = _task_identity_example()
    original = copy.deepcopy(task)
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
            )
        ),
        "diagnostics": True,
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
        {"validation": {"rtol": 0.001, "atol": 0.001}},
        {"objective": "eager"},
        {"min_speedup": 1.1},
    ],
)
def test_task_identity_binds_tuning_and_kernel_compatibility(changes):
    task = _task_identity_example()
    assert _factory._task_digest({**task, **changes}) != _factory._task_digest(task)


@pytest.mark.parametrize(
    "stage,action",
    [
        ("source", "repair_compile"),
        ("compile", "repair_compile"),
        ("correctness", "repair_correctness"),
        ("benchmark", "repair"),
        ("author", "repair"),
        ("provider", "finalize"),
        ("baseline", "finalize"),
        ("budget", "finalize"),
    ],
)
def test_search_action_uses_evaluation_stage_and_stagnation_budget(stage, action):
    history = [{"result": {"status": "error", "stage": stage}}]
    config = KernelGenerator(search=SearchPolicy(max_no_improvement=2))
    assert _factory._next_action(history, None, 1, config) == action
    assert _factory._next_action(history, None, 2, config) == "finalize"


def test_search_action_distinguishes_initial_search_improvement_and_stagnation():
    config = KernelGenerator()
    best = {"source": "verified"}
    history = [{"result": {"status": "passed"}}]
    assert _factory._next_action([], None, 0, config) == "author"
    assert _factory._next_action(history, best, 0, config) == "optimize"
    assert _factory._next_action(history, best, 1, config) == "rewrite"
    assert _factory._next_action(history, None, 0, config) == "rewrite"
    limited = replace(config, search=SearchPolicy(max_no_improvement=2))
    assert _factory._next_action(history, best, 2, limited) == "finalize"
