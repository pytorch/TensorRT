# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Settings, knobs and configs, and the helpers a session builds on.

Output comparison, code edits, bottleneck hints, rooflines, exports and the
provenance of the Torch-TensorRT source.
"""

import dataclasses
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch_tensorrt.autotune import (
    Budget,
    Goal,
    Knob,
    Limits,
    Timing,
    WorkloadFactory,
    compare_outputs,
)
from torch_tensorrt.autotune._knobs import (
    apply_changes,
    difference,
    is_member,
    knobs_by_name,
    resolve,
)
from torch_tensorrt.autotune._settings import digest, violations


def test_workload_factories_are_module_functions(tmp_path):
    with pytest.raises(ValueError, match="module:function"):
        WorkloadFactory("no_factory")
    with pytest.raises(ValueError, match="module:function"):
        WorkloadFactory("package.module:not-an-identifier")
    with pytest.raises(FileNotFoundError):
        WorkloadFactory(f"{tmp_path / 'missing.py'}:create")
    with pytest.raises(TypeError, match="JSON data"):
        WorkloadFactory("package.module:create", {"model": object()})


def test_file_factories_resolve_to_absolute_paths(tmp_path, monkeypatch):
    (tmp_path / "workloads.py").write_text("def create():\n    return 3\n")
    monkeypatch.chdir(tmp_path)
    factory = WorkloadFactory("workloads.py:create")
    assert factory.factory == f"{tmp_path / 'workloads.py'}:create"
    with pytest.raises(TypeError, match="not a torch_tensorrt.autotune.Workload"):
        factory.load()


def test_goal_measures_gains_in_the_metric_direction():
    latency = Goal(target=1.0)
    assert latency.gain(2.0, 1.5) == pytest.approx(0.25)
    assert latency.reached(0.9) and not latency.reached(1.1)
    throughput = Goal(metric="throughput", target=100.0)
    assert throughput.gain(50.0, 75.0) == pytest.approx(0.5)
    assert throughput.reached(100.0) and not throughput.reached(99.0)
    assert not Goal().reached(0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"metric": "speed"},
        {"min_improvement": 1.0},
        {"min_improvement": -0.1},
        {"target": -1.0},
    ],
)
def test_goal_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        Goal(**kwargs)


def test_limits_override_workload_defaults_and_report_every_miss():
    limits = Limits(accuracy={"top1": (">=", 0.99)})
    bounds = limits.bounds({"top1": (">=", 0.5), "kl": ("<=", 0.01)})
    assert bounds == {"top1": (">=", 0.99), "kl": ("<=", 0.01)}
    assert violations({"top1": 0.995, "kl": 0.001}, bounds) == []
    assert violations({"top1": 0.9, "kl": math.nan}, bounds) == [
        "kl is unavailable (needs <= 0.01)",
        "top1=0.9 (needs >= 0.99)",
    ]
    assert violations({}, {"top1": (">", 0.5)}) == ["top1 is unavailable (needs > 0.5)"]


@pytest.mark.parametrize(
    "bound", [("==", 1.0), (">=", True), (">=", math.inf), (">=",)]
)
def test_limits_reject_malformed_bounds(bound):
    with pytest.raises(ValueError):
        Limits(accuracy={"top1": bound})


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (Budget, {"max_trials": -1}),
        (Budget, {"patience": True}),
        (Budget, {"trial_timeout_s": 0}),
        (Budget, {"max_time_s": math.nan}),
        (Budget, {"prepare_timeout_s": 0}),
        (Budget, {"prepare_timeout_s": math.inf}),
        (Timing, {"rounds": 2}),
        (Timing, {"calls": 0}),
    ],
)
def test_budget_and_timing_reject_invalid_values(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)


@pytest.mark.parametrize(
    "args,match",
    [
        (("speed.option", (1,), 1, ""), "layer"),
        (("engine.option", (1, 1), 1, ""), "twice"),
        (("engine.option", (1, 2), 3, ""), "default"),
        (("engine.option", ([1],), [1], ""), "JSON scalars"),
        (("engine.option", (1,), 1, "", ("slow",)), "bottleneck hints"),
    ],
)
def test_knobs_validate_their_definition(args, match):
    with pytest.raises(ValueError, match=match):
        Knob(*args)


@pytest.mark.parametrize(
    "prepare,match",
    [
        (("fused",), "prepare lists 'fused', which is not one of its values"),
        (("none",), "prepare lists its default 'none'"),
        (("on", "on"), "prepare lists 'on' twice"),
        (("on", True), "prepare lists True, which is not one of its values"),
        ("on", "prepare must be a tuple of its values"),
        ((["on"],), r"prepare lists \['on'\], which is not one of its values"),
    ],
)
def test_prepared_values_must_be_non_default_values_of_the_knob(prepare, match):
    with pytest.raises(ValueError, match=match):
        Knob("kernels.k", ("none", "on", "off"), "none", "", prepare=prepare)


def test_prepared_values_keep_the_knobs_order_and_survive_json():
    knob = Knob("kernels.k", ("none", "on", 3), "none", "", prepare=[3, "on"])
    assert knob.prepare == ("on", 3)
    data = json.loads(json.dumps(knob.to_json()))
    assert data["prepare"] == ["on", 3]
    assert Knob.from_json(data) == knob
    # A knob without prepared values writes an empty list.
    plain = Knob("kernels.k", ("none", "on"), "none", "")
    assert plain.to_json()["prepare"] == []
    assert Knob.from_json(plain.to_json()).prepare == ()
    # A prepared value keeps its requirements.
    needs = Knob(
        "kernels.k",
        ("none", "on"),
        "none",
        "",
        requires={"runtime.cudagraphs": True},
        prepare=("on",),
    )
    assert needs.requirements("on") == {"runtime.cudagraphs": True}
    assert dataclasses.replace(needs, description="other").prepare == ("on",)


def test_module_workload_regions_generate_kernels_before_the_search():
    from torch_tensorrt.autotune import ModuleWorkload

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    workload = ModuleWorkload(
        model,
        [torch.ones(1, 2)],
        regions={"act": "1"},
        generator=SimpleNamespace(use_cache=True),
    )
    knob = knobs_by_name(workload.knobs())["kernels.act"]
    assert knob.values == ("native", "generated") and knob.prepare == ("generated",)
    # Every build after the preparation loads the kernel from the cache.
    with pytest.raises(ValueError, match="KernelGenerator with use_cache=True"):
        ModuleWorkload(
            model,
            [torch.ones(1, 2)],
            regions={"act": "1"},
            generator=SimpleNamespace(use_cache=False),
        )


def test_booleans_and_numbers_are_different_values():
    assert not is_member(True, (0, 1))
    assert not is_member(1, (True, False))
    assert not is_member(1.0, (1, 2))
    assert is_member(None, (None, "x"))


def test_configs_are_checked_and_normalized():
    knobs = knobs_by_name(
        [
            Knob("runtime.cudagraphs", (False, True), False, ""),
            Knob("engine.level", (3, 5), 3, ""),
        ]
    )
    config = apply_changes({}, {"runtime.cudagraphs": True}, knobs)
    assert config == {"runtime.cudagraphs": True}
    # Setting an option back to its default drops it, so equal configs match.
    assert apply_changes(config, {"runtime.cudagraphs": False}, knobs) == {}
    assert digest(apply_changes({"engine.level": 3}, {}, knobs)) == digest({})
    with pytest.raises(ValueError, match="Unknown knob"):
        apply_changes({}, {"engine.speed": 1}, knobs)
    with pytest.raises(ValueError, match="does not allow"):
        apply_changes({}, {"engine.level": 4}, knobs)
    with pytest.raises(ValueError, match="does not allow"):
        apply_changes({}, {"engine.level": True}, knobs)
    with pytest.raises(ValueError, match="unknown knob"):
        apply_changes({"engine.speed": 1}, {}, knobs)
    assert resolve({"engine.level": 5}, knobs) == {
        "runtime.cudagraphs": False,
        "engine.level": 5,
    }
    assert difference({"engine.level": 5}, {"runtime.cudagraphs": True}, knobs) == {
        "runtime.cudagraphs": True,
        "engine.level": 3,
    }
    with pytest.raises(ValueError, match="defined twice"):
        knobs_by_name(
            [Knob("engine.level", (3,), 3, ""), Knob("engine.level", (5,), 5, "")]
        )


def test_compare_outputs_reports_the_worst_leaf():
    reference = {"a": torch.ones(4), "ids": torch.tensor([1, 2, 3, 4])}
    same = compare_outputs(
        {"a": torch.ones(4), "ids": torch.tensor([1, 2, 3, 4])}, reference
    )
    assert same["max_abs_error"] == 0 and same["match_fraction"] == 1
    assert same["cosine_similarity"] == pytest.approx(1.0)
    off = compare_outputs(
        {"a": torch.tensor([1.0, 1.0, 1.0, 2.0]), "ids": torch.tensor([1, 2, 0, 0])},
        reference,
    )
    assert off["max_abs_error"] == pytest.approx(1.0)
    assert off["rel_l2_error"] == pytest.approx(0.5)
    assert off["match_fraction"] == pytest.approx(0.5)
    broken = compare_outputs(
        {"a": torch.full((4,), math.nan), "ids": reference["ids"]}, reference
    )
    assert broken["max_abs_error"] == math.inf and broken["cosine_similarity"] == -1.0
    with pytest.raises(ValueError, match="structure"):
        compare_outputs([torch.ones(4)], reference)
    with pytest.raises(ValueError, match="shape"):
        compare_outputs({"a": torch.ones(5), "ids": reference["ids"]}, reference)


def test_module_workload_knobs_default_to_its_compile_options():
    from torch_tensorrt.autotune import ModuleWorkload

    workload = ModuleWorkload(
        torch.nn.Linear(2, 2),
        [torch.ones(1, 2)],
        compile_options={"min_block_size": 1, "optimization_level": 4},
    )
    knobs = knobs_by_name(workload.knobs())
    assert knobs["graph.min_block_size"].default == 1
    assert knobs["engine.optimization_level"].default == 4
    assert knobs["engine.optimization_level"].values == (3, 5, 4)
    unlisted = ModuleWorkload(
        torch.nn.Linear(2, 2), [torch.ones(1, 2)], compile_options={"min_block_size": 7}
    )
    assert knobs_by_name(unlisted.knobs())["graph.min_block_size"].values == (
        7,
        5,
        1,
        3,
        10,
    )
    assert resolve({}, knobs)["graph.min_block_size"] == 1
    with pytest.raises(AttributeError):
        ModuleWorkload(
            torch.nn.Linear(2, 2), [], regions={"x": "missing"}, generator=object()
        )


def test_roofline_and_copy_hints_point_at_memory_bound_work():
    from torch_tensorrt.autotune._profile import bottleneck_hints

    found = bottleneck_hints(
        {
            "host_overhead_fraction": 0.0,
            "device_ms": 1.0,
            "copy_ms": 0.05,
            "host_syncs": 0,
            "kernels": 300,
            "top_kernels": [],
        },
        {"trt_engines": 1, "torch_partitions": 0},
        {},
        {
            "bytes_per_unit": 1e9,
            "peak_read_gbps": 1000.0,
            "floor_ms": 1.0,
            "fraction": 0.6,
        },
    )
    names = [hint["name"] for hint in found]
    assert names == ["roofline_gap", "copy_overhead"]
    assert "60% of memory-bandwidth-bound speed" in found[0]["detail"]
    assert "5% of GPU time is memory copies" in found[1]["detail"]
    close = bottleneck_hints(
        None,
        {},
        {},
        {
            "bytes_per_unit": 1.0,
            "peak_read_gbps": 1.0,
            "floor_ms": 1.0,
            "fraction": 0.9,
        },
    )
    assert close == []


def test_edits_stay_inside_their_fences():
    from torch_tensorrt.autotune._code_edits import (
        apply_edits,
        check_compiles,
        fenced_spans,
    )

    source = "a = 1\n# EDITABLE-START\nb = 2\nc = 3\n# EDITABLE-END\nd = 4\n"
    assert apply_edits(source, [{"search": "b = 2", "replace": "b = 5"}]).count("b = 5")
    for edits, match in (
        ([{"search": "a = 1", "replace": "a = 9"}], "not inside"),
        ([{"search": "= ", "replace": "="}], "occurs 2 times inside"),
        ([{"search": "b = 2", "replace": "# EDITABLE-END"}], "fences"),
        ([{"search": "", "replace": "x"}], "non-empty"),
        ([], "at least one edit"),
    ):
        with pytest.raises(ValueError, match=match):
            apply_edits(source, edits)
    with pytest.raises(ValueError, match="unterminated"):
        fenced_spans("# EDITABLE-START\nx = 1\n")
    with pytest.raises(ValueError, match="does not compile"):
        check_compiles("def (:\n", "kernels.py")


def test_edits_keep_the_fences_and_match_only_inside_them():
    from torch_tensorrt.autotune._code_edits import apply_edits

    source = "x = 1\n# EDITABLE-START\nx = 1\nPAD = (1, 1, 1)\n# EDITABLE-END\n"
    # The fenced occurrence is edited even though the text also occurs outside.
    edited = apply_edits(source, [{"search": "x = 1", "replace": "x = 2"}])
    assert edited.startswith("x = 1\n# EDITABLE-START\nx = 2\n")
    for edits, match in (
        # Code merged onto the END fence line would leave the block empty.
        ([{"search": "PAD = (1, 1, 1)\n", "replace": "PAD = 0"}], "fence line"),
        ([{"search": "x = 1\nPAD = (1, 1, 1)\n", "replace": ""}], "empty"),
        # Overlapping occurrences are ambiguous too.
        ([{"search": "1, 1", "replace": "2, 2"}], "occurs 2 times inside"),
        ([{"search": "y = 3", "replace": "y = 4"}], "does not occur"),
    ):
        with pytest.raises(ValueError, match=match):
            apply_edits(source, edits)


def test_editable_files_resolve_their_path_and_record_its_digest(tmp_path, monkeypatch):
    import hashlib

    from torch_tensorrt.autotune import EditableFile

    (tmp_path / "kernels.py").write_text("# EDITABLE-START\nN = 1\n# EDITABLE-END\n")
    monkeypatch.chdir(tmp_path)
    editable = EditableFile("kernels", "kernels.py", "launch settings")
    assert editable.path == str(tmp_path / "kernels.py")
    recorded = editable.to_json()
    assert (
        recorded["sha256"]
        == hashlib.sha256((tmp_path / "kernels.py").read_bytes()).hexdigest()
    )
    assert EditableFile.from_json(recorded) == editable


def test_noise_control_rejects_missing_or_infinite_numbers():
    from torch_tensorrt.autotune import NoiseControl

    for kwargs in (
        {"max_spread": None},
        {"noise_factor": None},
        {"max_spread": float("inf")},
        {"noise_factor": 0},
        {"quiet_gpu_s": -1},
        {"head_to_head_pairs": 1.5},
    ):
        with pytest.raises(ValueError, match="NoiseControl"):
            NoiseControl(**kwargs)


def test_knobs_declare_and_configs_respect_requirements():
    graphs = Knob("runtime.cudagraphs", (False, True), False, "graphs")
    blocks = Knob(
        "kernels.blocks", ("", "fused"), "", "fused", (), {"runtime.cudagraphs": True}
    )
    assert blocks.to_json()["requires"] == {"runtime.cudagraphs": True}
    knobs = knobs_by_name([graphs, blocks])
    with pytest.raises(ValueError, match="requires runtime.cudagraphs=True"):
        apply_changes({}, {"kernels.blocks": "fused"}, knobs)
    both = {"kernels.blocks": "fused", "runtime.cudagraphs": True}
    assert apply_changes({}, both, knobs) == both
    # Turning the requirement off again is refused while the option is set.
    with pytest.raises(ValueError, match="requires"):
        apply_changes(both, {"runtime.cudagraphs": False}, knobs)
    with pytest.raises(ValueError, match="no knob offers"):
        knobs_by_name([blocks])
    with pytest.raises(ValueError, match="cannot require itself"):
        Knob("kernels.x", (0, 1), 0, "x", (), {"kernels.x": 1})


def test_configs_carry_edit_digests_beside_knobs():
    from ._workloads import FAKE_KNOBS

    knobs = knobs_by_name(FAKE_KNOBS)
    sha = "a" * 64
    config = apply_changes({}, {"edit.speed": sha, "runtime.cudagraphs": True}, knobs)
    assert config == {"edit.speed": sha, "runtime.cudagraphs": True}
    assert resolve(config, knobs)["edit.speed"] == sha
    assert difference({}, config, knobs) == {
        "runtime.cudagraphs": True,
        "edit.speed": sha,
    }
    assert apply_changes(config, {"edit.speed": ""}, knobs) == {
        "runtime.cudagraphs": True
    }
    with pytest.raises(ValueError, match="digest of an edited file"):
        apply_changes({}, {"edit.speed": "not-a-digest"}, knobs)


# ---------------------------------------------------- per-value requirements

POOL = Knob(
    "runtime.graph_pool", (False, True), False, "", (), {"runtime.cudagraphs": True}
)
GRAPHS = Knob("runtime.cudagraphs", (False, True), False, "graphs")


def mixed_knob():
    # Values of several JSON types, so the JSON round trip must keep them.
    return Knob(
        "kernels.mixed",
        ("none", "fused", "tiled", 3, True),
        "none",
        "mixed",
        (),
        [
            (True, {"runtime.graph_pool": True}),
            ("fused", {"runtime.cudagraphs": True}),
            ("tiled", {"runtime.graph_pool": True, "runtime.cudagraphs": True}),
            (3, {"runtime.cudagraphs": True}),
        ],
    )


def test_knobs_may_require_settings_per_value():
    from torch_tensorrt.autotune._knobs import (
        chosen_changes,
        unmet_requirements,
        with_requirements,
    )

    mixed = mixed_knob()
    assert mixed.requires_by_value and not POOL.requires_by_value
    # Kept in the order of the knob's values.
    assert [value for value, _ in mixed.requires] == ["fused", "tiled", 3, True]
    assert mixed.requirements("fused") == {"runtime.cudagraphs": True}
    assert mixed.requirements(True) == {"runtime.graph_pool": True}
    assert mixed.requirements("none") == {} and mixed.requirements(1) == {}
    assert POOL.requirements(True) == {"runtime.cudagraphs": True}
    assert POOL.requirements(False) == {}
    knobs = knobs_by_name([GRAPHS, POOL, mixed])
    with pytest.raises(
        ValueError, match="kernels.mixed='fused' requires runtime.cudagraphs=True"
    ):
        apply_changes({}, {"kernels.mixed": "fused"}, knobs)
    with pytest.raises(ValueError, match="requires runtime.graph_pool=True"):
        apply_changes({}, {"kernels.mixed": "tiled", "runtime.cudagraphs": True}, knobs)
    full = {
        "kernels.mixed": "tiled",
        "runtime.cudagraphs": True,
        "runtime.graph_pool": True,
    }
    assert apply_changes({}, full, knobs) == full
    assert unmet_requirements(
        {"runtime.cudagraphs": True}, "kernels.mixed", knobs, "tiled"
    ) == {"runtime.graph_pool": True}
    assert unmet_requirements({"kernels.mixed": 3}, "kernels.mixed", knobs, 3) == {
        "runtime.cudagraphs": True
    }
    assert with_requirements({}, {"kernels.mixed": "tiled"}, knobs) == full
    assert chosen_changes(full, knobs) == {"kernels.mixed": "tiled"}
    # JSON round trips: per-value entries keep their values' types.
    data = json.loads(json.dumps(mixed.to_json()))
    assert data["requires"][2:] == [
        {"value": 3, "requires": {"runtime.cudagraphs": True}},
        {"value": True, "requires": {"runtime.graph_pool": True}},
    ]
    assert Knob.from_json(data) == mixed
    assert Knob.from_json(json.loads(json.dumps(POOL.to_json()))) == POOL
    assert dataclasses.replace(mixed, description="other").requires == mixed.requires
    # A mapping from value to requirements works too.
    mapped = Knob(
        "kernels.k", ("", "on"), "", "", (), {"on": {"runtime.cudagraphs": True}}
    )
    assert mapped.to_json()["requires"] == [
        {"value": "on", "requires": {"runtime.cudagraphs": True}}
    ]


@pytest.mark.parametrize(
    "requires,match",
    [
        ({"z": {"runtime.cudagraphs": True}}, "not one of its values"),
        ({"a": {"runtime.cudagraphs": True}}, "its default"),
        ({"b": {"runtime.cudagraphs": True}, "runtime.x": True}, "mixes"),
        ([("b", {"engine.x": 1}), ("b", {"engine.x": 2})], "twice"),
        ({"b": {"engine.k": "a"}}, "cannot require itself"),
        ({"b": {"runtime.cudagraphs": [True]}}, "JSON scalars"),
        ({"b": {3: True}}, "must name other knobs"),
        (["runtime.cudagraphs"], "must map other knobs"),
    ],
)
def test_per_value_requirements_are_validated(requires, match):
    with pytest.raises(ValueError, match=match):
        Knob("engine.k", ("a", "b"), "a", "", (), requires)


def test_knobs_refuse_requirements_no_config_can_meet():
    from torch_tensorrt.autotune._knobs import chosen_changes

    unoffered = Knob(
        "kernels.k", ("", "on"), "", "", (), {"on": {"runtime.cudagraphs": 2}}
    )
    with pytest.raises(
        ValueError, match="kernels.k='on' requires runtime.cudagraphs=2"
    ):
        knobs_by_name([GRAPHS, unoffered])
    # Debug mode needs CUDA graphs off; the kernel needs them on and debug on.
    debug = Knob(
        "runtime.debug", (False, True), False, "", (), {"runtime.cudagraphs": False}
    )
    contradiction = Knob(
        "kernels.k",
        ("", "on"),
        "",
        "",
        (),
        {"runtime.cudagraphs": True, "runtime.debug": True},
    )
    with pytest.raises(
        ValueError,
        match=r"kernels.k='on' can never be used: runtime.debug=True requires "
        r"runtime.cudagraphs=False, which conflicts with runtime.cudagraphs=True",
    ):
        knobs_by_name([GRAPHS, debug, contradiction])
    # Options that require each other are chosen together.
    a = Knob("engine.a", (0, 1), 0, "", (), {"engine.b": 1})
    b = Knob("engine.b", (0, 1), 0, "", (), {"engine.a": 1})
    both = {"engine.a": 1, "engine.b": 1}
    assert chosen_changes(both, knobs_by_name([a, b])) == both


# ---------------------------------------------------- compute roofline hint


def test_the_compute_floor_comes_from_the_candidate_or_its_structure():
    from torch_tensorrt.autotune._measure import compute_roofline

    declared = SimpleNamespace(flops_per_unit=2e12, peak_tflops=200.0)
    line = compute_roofline(declared, {"compute_floor_ms": 1.0}, 20.0)
    assert line["floor_ms"] == pytest.approx(10.0)
    assert line["fraction"] == pytest.approx(0.5)
    # A describe() that reports its own floor stands for it, whatever its rate.
    structure = {
        "gflop_per_unit": 3000.0,
        "peak_tflops": 250.0,
        "compute_floor_ms": 12.5,
    }
    line = compute_roofline(object(), structure, 25.0)
    assert line["floor_ms"] == 12.5 and line["fraction"] == pytest.approx(0.5)
    assert line["flops_per_unit"] == pytest.approx(3e12)
    line = compute_roofline(
        object(), {"gflop_per_unit": 1000.0, "peak_tflops": 100.0}, 20.0
    )
    assert line["floor_ms"] == pytest.approx(10.0)
    assert compute_roofline(object(), {"gflop_per_unit": 1000.0}, 20.0) is None
    assert compute_roofline(object(), {"compute_floor_ms": "fast"}, 20.0) is None


def test_the_compute_gap_hint_points_at_compute_bound_work():
    from torch_tensorrt.autotune._profile import bottleneck_hints

    compute = {
        "flops_per_unit": 2.5e12,
        "peak_tflops": 250.0,
        "floor_ms": 10.0,
        "fraction": 0.2,
    }
    found = bottleneck_hints(None, {}, {}, None, compute)
    assert [hint["name"] for hint in found] == ["compute_gap"]
    assert found[0]["strength"] == pytest.approx(0.8)
    assert found[0]["detail"].startswith("it runs at 20% of compute-bound speed")
    assert "2500 GFLOP per unit at 250 TFLOP/s" in found[0]["detail"]
    assert bottleneck_hints(None, {}, {}, None, {**compute, "fraction": 0.9}) == []
    # With both floors known, only the higher one binds.
    memory = {"bytes_per_unit": 1e9, "peak_read_gbps": 1000.0, "floor_ms": 1.0}
    names = [
        hint["name"]
        for hint in bottleneck_hints(None, {}, {}, {**memory, "fraction": 0.1}, compute)
    ]
    assert names == ["compute_gap"]
    memory.update(floor_ms=20.0, fraction=0.4)
    assert [
        hint["name"] for hint in bottleneck_hints(None, {}, {}, memory, compute)
    ] == ["roofline_gap"]
    # Knobs can name it as a bottleneck they relieve.
    assert Knob(
        "engine.tactic", ("a", "b"), "a", "", ("compute_gap",)
    ).bottleneck_hints == ("compute_gap",)


def test_the_worker_reports_the_compute_floor():
    from torch_tensorrt.autotune._measure import measure_candidate

    from ._workloads import FakeWorkload

    # 1e12 FLOPs at 1000 TFLOP/s: a 1 ms floor; the fake takes 2.5 ms a unit.
    workload = FakeWorkload(milliseconds=5.0, flops_per_unit=1e12, peak_tflops=1000.0)
    timing = Timing(warmup=0, rounds=3, calls=1, profile_calls=0)
    result = measure_candidate(
        workload,
        resolve({}, knobs_by_name(workload.knobs())),
        workload.reference(),
        timing,
    )
    assert result["compute_roofline"]["floor_ms"] == pytest.approx(1.0)
    assert 0 < result["compute_roofline"]["fraction"] < 0.8
    assert "compute_gap" in [hint["name"] for hint in result["bottleneck_hints"]]


# ------------------------------------------------------- profiles and exports


def test_profile_summaries_skip_annotation_spans():
    from torch_tensorrt.autotune._profile import summarize_events

    cuda = torch.autograd.DeviceType.CUDA

    def event(name, us, annotation=False, device=cuda):
        return SimpleNamespace(
            device_type=device,
            name=name,
            device_time_total=us,
            is_user_annotation=annotation,
        )

    summary = summarize_events(
        [
            event("gemm", 300.0),
            event("Memcpy DtoH (Device -> Pinned)", 50.0),
            # Inductor's "## Call CompiledFxGraph" span covers kernels counted above.
            event("## Call CompiledFxGraph f ##", 900.0, annotation=True),
            event("aten::mm", 0.0, device=torch.autograd.DeviceType.CPU),
        ],
        per_unit=1,
        wall_ms=1.0,
    )
    assert summary["kernels"] == 1 and summary["host_syncs"] == 1
    assert summary["device_ms"] == pytest.approx(0.35)
    assert summary["copy_ms"] == pytest.approx(0.05)
    assert [kernel["name"] for kernel in summary["top_kernels"]] == [
        "gemm",
        "Memcpy DtoH (Device -> Pinned)",
    ]


def test_unsupported_exports_are_told_apart_from_failed_ones():
    from torch_tensorrt.autotune._measure import export_unsupported
    from torch_tensorrt.region import RegionError

    assert export_unsupported(NotImplementedError("cannot save"))
    assert export_unsupported(
        RegionError(
            "Saving execute_in_torch region 'r' with backend='inductor' is not "
            "supported: the exporter cannot preserve its execution backend."
        )
    )
    assert not export_unsupported(
        RegionError("Saving fused_kernel region 'r' requires a built TensorRT engine.")
    )
    assert not export_unsupported(RuntimeError("the disk is full"))


# --------------------------------------------------------------- provenance


def test_workload_files_hash_the_factory_and_its_siblings(tmp_path):
    from torch_tensorrt.autotune._worker import workload_files

    (tmp_path / "workloads.py").write_text("def create():\n    pass\n")
    (tmp_path / "helper.py").write_text("X = 1\n")
    (tmp_path / "notes.txt").write_text("not a module\n")
    (tmp_path / "package").mkdir()
    (tmp_path / "package" / "inner.py").write_text("")
    files = workload_files(f"{tmp_path / 'workloads.py'}:create")
    assert sorted(files) == [
        str(tmp_path / "helper.py"),
        str(tmp_path / "workloads.py"),
    ]
    assert files[str(tmp_path / "helper.py")] == hashlib.sha256(b"X = 1\n").hexdigest()
    # A module factory: the imported module's file and the files beside it.
    assert str(Path(json.__file__).resolve()) in workload_files("json:loads")
    assert workload_files("not_imported_anywhere:create") == {}


def make_checkout(repo, commit=True):
    """A git repository with ``package/__init__.py``; return its HEAD (or None)."""
    package = repo / "package"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("A = 1\n")
    (repo / ".gitignore").write_text("site/\n")

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
            + ["-c", "commit.gpgsign=false", *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q")
    if not commit:
        return None
    git("add", "-A")
    git("commit", "-q", "-m", "first")
    return git("rev-parse", "HEAD")


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_source_provenance_names_the_commit_and_uncommitted_changes(
    tmp_path, monkeypatch
):
    from torch_tensorrt.autotune._worker import source_provenance

    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    repo = tmp_path / "repo"
    head = make_checkout(repo)
    package = repo / "package"
    assert source_provenance(package) == {"commit": head, "uncommitted_sha256": None}
    (package / "__init__.py").write_text("A = 2\n")
    edited = source_provenance(package)
    assert edited["commit"] == head and edited["uncommitted_sha256"]
    (package / "new.py").write_text("B = 1\n")
    untracked = source_provenance(package)["uncommitted_sha256"]
    assert untracked not in (None, edited["uncommitted_sha256"])
    # Files no checkout tracks, as of a wheel installed inside a repository.
    site = repo / "site" / "package"
    site.mkdir(parents=True)
    (site / "__init__.py").write_text("")
    assert source_provenance(site) is None
    (tmp_path / "plain").mkdir()
    assert source_provenance(tmp_path / "plain") is None
    # A git that sees no repository, as with GIT_DIR pointing elsewhere.
    monkeypatch.setenv("GIT_DIR", os.devnull)
    assert source_provenance(package) is None


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() != 0,
    reason="needs root to hand the checkout to another user",
)
def test_source_provenance_reads_a_checkout_another_user_owns(tmp_path, monkeypatch):
    from torch_tensorrt.autotune._worker import source_provenance

    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    # Only the session's own settings: no safe.directory from the machine.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    repo = tmp_path / "repo"
    head = make_checkout(repo)
    package = repo / "package"
    for path in [repo, *repo.rglob("*")]:
        os.chown(path, 65534, 65534, follow_symlinks=False)
    refused = subprocess.run(
        ["git", "-C", str(package), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    assert refused.returncode and "dubious ownership" in refused.stderr
    # The worker runs this checkout's code, so its provenance is read anyway.
    assert source_provenance(package) == {"commit": head, "uncommitted_sha256": None}
    (package / "__init__.py").write_text("A = 2\n")
    assert source_provenance(package)["uncommitted_sha256"]


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_source_provenance_says_why_a_checkout_cannot_be_read(tmp_path, monkeypatch):
    from torch_tensorrt.autotune._worker import source_provenance

    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    make_checkout(tmp_path / "repo", commit=False)
    package = tmp_path / "repo" / "package"
    # A checkout without a commit has no HEAD to pin.
    unread = source_provenance(package)
    assert list(unread) == ["unavailable"]
    assert unread["unavailable"].startswith("git rev-parse failed: fatal: ")
    # Without git a checkout cannot be read; a package outside any checkout
    # is an installed one, which its version identifies.
    monkeypatch.setenv("PATH", str(tmp_path / "no-git"))
    assert source_provenance(package) == {"unavailable": "git is not installed"}
    (tmp_path / "plain").mkdir()
    if not any((parent / ".git").exists() for parent in tmp_path.parents):
        assert source_provenance(tmp_path / "plain") is None
