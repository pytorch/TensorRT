# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""bench_compare: specs, the run order, the GPU gate, summaries and the driver.

Run with ``python -m pytest tools/llm/tests/test_bench_compare.py``; no GPU or
model download is needed (one test checks the reproduction kit's recipes
against the workloads' knobs when the models' configs are cached). The
driver tests start real processes, with a tiny CPU workload, and stop and
kill drivers (Linux).
"""

import fcntl
import hashlib
import itertools
import json
import os
import re
import signal
import subprocess
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import bench_compare  # noqa: E402
from bench_compare import SpecError, load_spec  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
KIT = HERE / "agentic_bench"
DECODE = "tools/llm/agentic_decode.py:create_case"
ARGS = {"model": "Qwen/Qwen3-0.6B", "isl": 8192, "osl": 64}


def write_spec(path, **entries):
    spec = {
        "workload": DECODE,
        "args": ARGS,
        "reference": "reference.pt",
        "configs": {"trt-baseline": {"recipe": {}}},
        **entries,
    }
    path.write_text(json.dumps(spec))
    return path


# ------------------------------------------------------------------- specs


def test_paths_resolve_against_the_repository_and_the_output_directory(
    tmp_path, monkeypatch
):
    spec_path = write_spec(tmp_path / "spec.json")
    spec = load_spec(spec_path, out_dir=tmp_path / "out")
    factory = HERE / "agentic_decode.py"
    assert factory.is_file()
    assert spec["workload"] == f"{factory}:create_case"
    assert spec["reference"] == str(tmp_path / "out" / "reference.pt")
    # Defaults are filled in.
    assert spec["timeout_s"] == 3600.0
    assert spec["noise_control"] == {
        "quiet_gpu_s": 60.0,
        "retries": 2,
        "max_spread": 0.03,
    }
    assert spec["accuracy"] == {}
    assert spec["configs"] == {"trt-baseline": {"recipe": {}, "bench": {}}}
    # --reference overrides the spec, relative to the working directory.
    monkeypatch.chdir(tmp_path)
    spec = load_spec(spec_path, reference="shared/reference.pt")
    assert spec["reference"] == str(tmp_path / "shared" / "reference.pt")
    # Absolute paths and package factories are kept.
    absolute = write_spec(
        tmp_path / "absolute.json",
        workload="package.module:create_case",
        reference=str(tmp_path / "ref.pt"),
    )
    spec = load_spec(absolute)
    assert spec["workload"] == "package.module:create_case"
    assert spec["reference"] == str(tmp_path / "ref.pt")
    # A resolved spec resolves to itself (``all`` hands it to every run).
    resolved = tmp_path / "resolved.json"
    resolved.write_text(json.dumps(load_spec(spec_path, out_dir=tmp_path)))
    assert load_spec(resolved) == load_spec(spec_path, out_dir=tmp_path)


@pytest.mark.parametrize(
    "entries, match",
    [
        ({"reference": "reference.pt"}, "relative to an output directory"),
        ({"workload": "tools/llm/no_such_file.py:create_case"}, "does not exist"),
        ({"workload": "create_case"}, "path/to/file.py:function"),
        ({"config": {}}, r"unknown keys \['config'\]"),
        ({"configs": {}}, "non-empty"),
        ({"configs": {"../up": {}}}, "names files"),
        ({"configs": {"a": {"recipes": {}}}}, r"unknown keys \['recipes'\]"),
        ({"configs": {"a": {"recipe": []}}}, "a.recipe must be an object"),
        ({"configs": {"a": {"extends": "b"}}}, "extends 'b', which the spec"),
        (
            {"configs": {"a": {"extends": "b"}, "b": {"extends": "a"}}},
            "cycle: a -> b -> a",
        ),
        ({"timeout_s": 0}, "timeout_s must be a positive number"),
        ({"noise_control": {"quiet_s": 1}}, "unknown noise_control settings"),
        ({"noise_control": {"retries": -1}}, "retries must be a non-negative"),
        ({"accuracy": {"mean_kl": ["=", 1]}}, "accuracy limit of 'mean_kl'"),
        (
            {"head_to_head": ["trt-baseline", "trt-session"]},
            r"lacks: \['trt-session'\]",
        ),
    ],
)
def test_specs_that_cannot_run_are_refused_with_the_reason(tmp_path, entries, match):
    out_dir = None if entries.get("reference") else tmp_path
    with pytest.raises(SpecError, match=match):
        load_spec(write_spec(tmp_path / "spec.json", **entries), out_dir=out_dir)


def test_configs_extend_other_configs(tmp_path):
    configs = {
        "base": {"recipe": {"a.x": 1, "a.y": 2}, "bench": {"backend": "tensorrt"}},
        "more": {"extends": "base", "recipe": {"a.y": 3, "a.z": 4}},
        "most": {"extends": "more", "bench": {"mode": "default"}},
        "alone": {"bench": {"backend": "eager"}},
    }
    spec = load_spec(write_spec(tmp_path / "s.json", configs=configs), out_dir=tmp_path)
    assert spec["configs"] == {
        "base": {"recipe": {"a.x": 1, "a.y": 2}, "bench": {"backend": "tensorrt"}},
        "more": {
            "recipe": {"a.x": 1, "a.y": 3, "a.z": 4},
            "bench": {"backend": "tensorrt"},
        },
        "most": {
            "recipe": {"a.x": 1, "a.y": 3, "a.z": 4},
            "bench": {"backend": "tensorrt", "mode": "default"},
        },
        "alone": {"recipe": {}, "bench": {"backend": "eager"}},
    }


def make_session(directory, config, kwargs=ARGS, factory=DECODE):
    directory.mkdir(parents=True)
    (directory / "session.json").write_text(
        json.dumps(
            {
                "workload_factory": {
                    "factory": f"/elsewhere/{factory}",
                    "kwargs": kwargs,
                }
            }
        )
    )
    (directory / "report.json").write_text(
        json.dumps(
            {
                "winner": {"trial": "0003", "config": config},
                "accuracy_bounds": {
                    "top1_agreement": [">=", 0.97],
                    "mean_kl": ["<=", 0.005],
                },
                "decision": "accepted",
                "stop_reason": "the planner stopped",
            }
        )
    )
    return directory


def test_a_session_supplies_trt_session_its_limits_and_its_edited_files(tmp_path):
    configs = {
        "trt-session": {"recipe": {"workload.cache": "static_v3"}},
        "trt-session-graphs": {
            "extends": "trt-session",
            "recipe": {"runtime.cudagraphs": True},
        },
    }
    spec_path = write_spec(
        tmp_path / "s.json", configs=configs, accuracy={"mean_kl": ["<=", 0.01]}
    )
    config = {"kernels.decode_blocks": "all", "edit.decode_launch": "a" * 64}
    session = make_session(tmp_path / "session", config)
    spec = load_spec(spec_path, out_dir=tmp_path, session=session)
    assert spec["configs"]["trt-session"] == {"recipe": config, "bench": {}}
    # Configs that extend trt-session follow it.
    assert spec["configs"]["trt-session-graphs"]["recipe"] == {
        **config,
        "runtime.cudagraphs": True,
    }
    # The session's limits apply; the spec's own win.
    assert spec["accuracy"] == {
        "top1_agreement": [">=", 0.97],
        "mean_kl": ["<=", 0.01],
    }
    assert spec["edits_dir"] == str(session)
    assert spec["session"]["winner_config"] == config
    assert spec["session"]["decision"] == "accepted"
    assert bench_compare.edit_digests(spec) == {"a" * 64}
    # Without --session the spec's own recipe stays.
    assert load_spec(spec_path, out_dir=tmp_path)["configs"]["trt-session"] == {
        "recipe": {"workload.cache": "static_v3"},
        "bench": {},
    }


def test_a_session_of_another_workload_is_refused(tmp_path):
    spec_path = write_spec(tmp_path / "s.json")
    other = make_session(tmp_path / "other", {}, kwargs={**ARGS, "isl": 128})
    with pytest.raises(SpecError, match="not of this spec's"):
        load_spec(spec_path, out_dir=tmp_path, session=other)
    prefill = make_session(
        tmp_path / "prefill", {}, factory="tools/llm/agentic_prefill.py:create_case"
    )
    with pytest.raises(SpecError, match="not of this spec's"):
        load_spec(spec_path, out_dir=tmp_path, session=prefill)
    (tmp_path / "unfinished").mkdir()
    with pytest.raises(SpecError, match="not a finished Auto-Tuner session"):
        load_spec(spec_path, out_dir=tmp_path, session=tmp_path / "unfinished")


# ------------------------------------------------------------ the kit


def kit_specs():
    return sorted((KIT / "specs").glob("*.json"))


def kit_benches():
    briefs = (KIT / "briefs").glob("*.md")
    return sorted(path.stem for path in briefs if not path.stem.endswith("-hybrid"))


def kernels_specs():
    """The specs that compare where a region's kernels come from (``*-kernels``)."""
    return [path for path in kit_specs() if path.stem.endswith("-kernels")]


def test_the_kit_has_two_briefs_and_two_specs_per_benchmark():
    benches = set(kit_benches())
    assert len(benches) == 8
    hybrid = {f"{bench}-hybrid" for bench in benches}
    assert {path.stem for path in (KIT / "briefs").glob("*.md")} == benches | hybrid
    # Besides them, kernels head-to-heads, which no session brief belongs to.
    kernels = {path.stem for path in kernels_specs()}
    assert {path.stem for path in kit_specs()} == benches | hybrid | kernels
    readme = (KIT / "README.md").read_text()
    for name in sorted(kernels):
        assert f"specs/{name}.json" in readme, name
    for bench in sorted(benches):
        for name in ("briefs/{}.md", "briefs/{}-hybrid.md", "specs/{}.json"):
            assert name.format(bench) in readme, bench
        assert f"specs/{bench}-hybrid.json" in readme, bench
        # A hybrid brief is the session's brief, unchanged, and a section on
        # the hybrid knobs, which the session's brief does not mention.
        brief = (KIT / "briefs" / f"{bench}.md").read_bytes()
        extended = (KIT / "briefs" / f"{bench}-hybrid.md").read_bytes()
        assert extended.startswith(brief) and len(extended) > len(brief), bench
        assert b"kernels.inductor" in extended[len(brief) :], bench
        assert b"kernels.inductor" not in brief, bench


def test_the_session_briefs_are_the_ones_the_published_sessions_read():
    """``briefs/SHA256SUMS`` holds the ``brief_sha256`` each session recorded."""
    listed = {}
    for line in (KIT / "briefs" / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ")
        listed[name] = digest
    assert sorted(listed) == [f"{bench}.md" for bench in kit_benches()]
    for name, digest in listed.items():
        actual = hashlib.sha256((KIT / "briefs" / name).read_bytes()).hexdigest()
        assert actual == digest, name


@pytest.mark.parametrize("path", kit_specs(), ids=lambda path: path.stem)
def test_the_kit_specs_are_templates(tmp_path, path):
    text = path.read_text()
    raw = json.loads(text)
    # No machine's paths: workloads relative to the repository, the
    # reference relative to the output directory.
    assert not re.search(r"/(dev/shm|workspace|tmp|root|home)/", text)
    assert raw["workload"].startswith("tools/llm/")
    assert raw["reference"] == "reference.pt"
    spec = load_spec(path, out_dir=tmp_path)
    assert Path(spec["workload"].rpartition(":")[0]).is_file()
    assert spec["reference"] == str(tmp_path / "reference.pt")
    assert spec["head_to_head"]
    if path.stem.endswith("-kernels"):
        # A kernels head-to-head runs at least one generated configuration
        # (test_the_kernels_specs_compare_kernel_sources has the rest).
        assert any(
            generated_values(spec["configs"][name]["recipe"])
            for name in spec["head_to_head"]
        )
    else:
        # The configurations the published head-to-head ran, starting with the
        # session's recipe (which --session replaces).
        assert "trt-session" in spec["head_to_head"]
    if path.stem.endswith("-hybrid"):
        assert any(
            raw["configs"][name].get("extends") == "trt-session"
            for name in raw["head_to_head"]
        )
    elif not path.stem.endswith("-kernels"):
        assert spec["head_to_head"][:2] == ["trt-baseline", "trt-session"]
    # Descriptions and notes explain configurations, not measured results.
    prose = json.dumps([raw.get("description"), raw.get("notes")])
    assert not re.search(r"\d\s*(ms|us)\b|%|\bx faster", prose)


def generated_values(recipe):
    """The knob values of ``recipe`` that run kernels a KernelGenerator writes."""
    return {
        key: value
        for key, value in recipe.items()
        if (key == "kernels.generated" and value != "none")
        or (key.startswith("kernels.") and value == "generated")
    }


# The kernels head-to-heads of the kit's benchmarks: the TensorRT-alone base
# (a configuration of the bench's hybrid spec), its best hybrid and its
# strongest Inductor configuration, and the generated value on that base.
KERNELS_SPECS = {
    "gemma3-prefill-2k": (
        "trt-session-cudagraphs",
        "trt-hybrid-rope-plugin",
        {"kernels.generated": "qk-norm-rope"},
        "inductor-fp16-fp16acc-max-autotune-torchflash-cdt",
    ),
    "gemma3-prefill-8k": (
        "trt-session-cudagraphs",
        "trt-hybrid-attention-rope-inductor",
        {"kernels.generated": "qk-norm-rope"},
        "inductor-fp16-fp16acc-max-autotune-torchflash",
    ),
    "modernbert-512": (
        "trt-decompose-cudagraphs",
        "trt-decompose-hybrid-rope-kernel",
        {"kernels.generated": "split-rope"},
        "inductor-max-autotune-flex-fp16acc",
    ),
    "modernbert-8k": (
        "trt-decompose-cudagraphs",
        "trt-decompose-hybrid-rope-kernel",
        {"kernels.generated": "split-rope"},
        "inductor-max-autotune-flex-cdt-fp16acc",
    ),
    "qwen35-prefill-2k": (
        "trt-session",
        "trt-hybrid-attention+prologue-decay",
        {"kernels.gdn_scan": "generated"},
        "inductor-functional-causal-tf32-fp16-fp16acc-default",
    ),
    "nemotron-decode": (
        "trt-session",
        "trt-hybrid-lm-head",
        {"kernels.generated": "mlp"},
        "inductor-max-autotune-flex",
    ),
}


@pytest.mark.parametrize("bench", sorted(KERNELS_SPECS))
def test_the_kernels_specs_compare_kernel_sources(tmp_path, bench):
    """``<bench>-kernels.json``: TensorRT alone, the best hybrid, the generated
    kernels on the same base and the strongest Inductor configuration."""
    base, best, generated, inductor = KERNELS_SPECS[bench]
    spec = load_spec(KIT / "specs" / f"{bench}-kernels.json", out_dir=tmp_path)
    hybrid = load_spec(KIT / "specs" / f"{bench}-hybrid.json", out_dir=tmp_path)
    session = load_spec(KIT / "specs" / f"{bench}.json", out_dir=tmp_path)
    # The bench's workload and arguments.
    assert (spec["workload"], spec["args"]) == (hybrid["workload"], hybrid["args"])
    assert (spec["workload"], spec["args"]) == (session["workload"], session["args"])
    names = spec["head_to_head"]
    assert names[0] == "trt-alone" and names[1] == best and names[-1] == inductor
    assert len(names) == 4
    configs = spec["configs"]
    # TensorRT alone: the hybrid spec's base with CUDA graphs.
    alone = configs["trt-alone"]
    assert alone == hybrid["configs"][base]
    assert alone["recipe"]["runtime.cudagraphs"] is True and not alone["bench"]
    assert not generated_values(alone["recipe"])
    # The best hybrid and the Inductor configuration as the hybrid spec has them.
    assert configs[best] == hybrid["configs"][best]
    assert configs[inductor] == hybrid["configs"][inductor]
    # The generated configuration: TensorRT alone plus the generated value.
    (name,) = [name for name in names if generated_values(configs[name]["recipe"])]
    assert configs[name] == {
        "recipe": {**alone["recipe"], **generated},
        "bench": {},
    }
    # Configurations off the head-to-head extend the same base or the
    # session recipe; --session replaces that recipe (not ModernBERT's base,
    # which is not the session's).
    raw = json.loads((KIT / "specs" / f"{bench}-kernels.json").read_text())
    assert configs["trt-session"] == hybrid["configs"]["trt-session"]
    if base != "trt-decompose-cudagraphs":
        assert raw["configs"]["trt-alone"]["extends"] == "trt-session"
    assert set(raw["notes"]) == set(raw["configs"])


def test_the_kit_briefs_hold_no_machine_paths():
    for brief in (KIT / "briefs").glob("*.md"):
        assert not re.search(r"/(dev/shm|workspace|tmp)/", brief.read_text()), brief


def test_the_kit_recipes_fit_the_workloads_knobs(monkeypatch):
    """Every kit recipe is a valid config of its workload at this commit.

    Needs the models' Hugging Face configs in the local cache (the knobs
    depend on them); skipped otherwise.
    """
    import torch_tensorrt.dynamo
    from torch_tensorrt.autotune import WorkloadFactory
    from torch_tensorrt.autotune._knobs import apply_changes, knobs_by_name

    # Most workloads probe TensorRT's build routes with a builder, which needs
    # a GPU; offer the routes anyway.
    monkeypatch.setattr(
        torch_tensorrt.dynamo, "is_global_perf_tuning_available", lambda: True
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    knobs = {}
    for path in kit_specs():
        spec = load_spec(path, out_dir=path.parent)
        key = (spec["workload"], json.dumps(spec["args"], sort_keys=True))
        if key not in knobs:
            try:
                workload = WorkloadFactory(spec["workload"], spec["args"]).load()
                knobs[key] = knobs_by_name(workload.knobs())
            except OSError as error:  # the model's config is not cached
                pytest.skip(f"{spec['args'].get('model')}: {error}")
        for name, config in spec["configs"].items():
            try:
                apply_changes({}, config["recipe"], knobs[key])
            except ValueError as error:
                pytest.fail(f"{path.name} {name}: {error}")


# ------------------------------------------------------------- the order


@pytest.mark.parametrize("count", range(1, 8))
def test_rounds_follow_a_balanced_latin_square(count):
    configs = [chr(ord("a") + index) for index in range(count)]
    period = count if count % 2 == 0 else 2 * count
    rows = [bench_compare.round_order(configs, index) for index in range(period)]
    for row in rows:
        assert sorted(row) == configs
    # Every config runs equally often in every position ...
    for position in range(count):
        assert Counter(row[position] for row in rows) == Counter(
            dict.fromkeys(configs, period // count)
        )
    # ... and right after every other config equally often.
    pairs = Counter(pair for row in rows for pair in zip(row, row[1:]))
    expected = dict.fromkeys(itertools.permutations(configs, 2), period // count)
    assert pairs == Counter(expected)
    # The rows repeat after a period.
    assert bench_compare.round_order(configs, period) == rows[0]


def test_two_configs_alternate_and_four_take_a_williams_square():
    order = bench_compare.round_order
    assert [order(["a", "b"], index) for index in range(4)] == [
        ["a", "b"],
        ["b", "a"],
        ["a", "b"],
        ["b", "a"],
    ]
    assert [order("abcd", index) for index in range(4)] == [
        list("abdc"),
        list("bcad"),
        list("cdba"),
        list("dacb"),
    ]
    assert order([], 3) == []


# ---------------------------------------------------- lock and idle wait


def lock_is_free(path):
    with open(path) as other:
        try:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(other, fcntl.LOCK_UN)
        return True


def wait_for(condition, timeout_s=30.0):
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def test_the_gpu_lock_is_the_lock_flock_takes(tmp_path):
    path = tmp_path / "locks" / "gpu.lock"
    with bench_compare.gpu_lock(path) as handle:
        assert isinstance(handle, int)
        assert not lock_is_free(path)
    assert lock_is_free(path)
    with bench_compare.gpu_lock(None) as handle:
        assert handle is None


def test_a_run_inherits_the_gpu_lock(tmp_path):
    path = tmp_path / "gpu.lock"
    # A flock(2) lock belongs to the open file: a child that inherited the
    # descriptor holds it after the driver closed its own ...
    with bench_compare.gpu_lock(path) as handle:
        holder = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"], pass_fds=(handle,)
        )
    try:
        assert not lock_is_free(path)
    finally:
        holder.kill()
        holder.wait()
    assert lock_is_free(path)
    # ... and every run gets the descriptor that Gate.hold yields.
    script = (
        "import os, sys; "
        "print(os.path.samestat(os.fstat(int(sys.argv[1])), os.stat(sys.argv[2])))"
    )
    log = tmp_path / "run.log"
    with bench_compare.Gate(path, quiet_s=0).hold() as held:
        assert held.quiet == "idle wait off" and len(held.fds) == 1
        code = bench_compare._child(
            [sys.executable, "-c", script, str(held.fds[0]), str(path)],
            log,
            dict(os.environ),
            30.0,
            pass_fds=held.fds,
        )
    assert code == 0 and log.read_text().strip() == "True"
    with bench_compare.Gate(None, quiet_s=0).hold() as held:
        assert held.fds == ()


def test_the_lock_and_the_quiet_command_come_from_options_or_the_environment(
    monkeypatch,
):
    monkeypatch.delenv(bench_compare.LOCK_ENV, raising=False)
    # One path for the whole machine, whatever TMPDIR is.
    assert bench_compare.DEFAULT_LOCK == Path("/tmp/torch_tensorrt_gpu.lock")
    assert bench_compare._lock_path(None) == bench_compare.DEFAULT_LOCK
    assert bench_compare._lock_path("none") is None
    monkeypatch.setenv(bench_compare.LOCK_ENV, "/var/lock/gpu.lock")
    assert bench_compare._lock_path(None) == Path("/var/lock/gpu.lock")
    assert bench_compare._lock_path("/other.lock") == Path("/other.lock")
    monkeypatch.setenv(bench_compare.LOCK_ENV, "none")
    assert bench_compare._lock_path(None) is None
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert bench_compare.visible_gpu() == "0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-1234, 0")
    assert bench_compare.visible_gpu() == "GPU-1234"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert bench_compare.visible_gpu() is None


def scripted(readings):
    readings = iter(readings)
    return lambda: next(readings)


def test_the_idle_wait_needs_consecutive_idle_samples():
    sleeps = []
    wait = bench_compare.wait_for_idle_gpu
    busy, idle = (40.0, 30000.0), (1.0, 30000.0)
    ok, message = wait(
        3,
        60,
        read=scripted([idle, busy, idle, idle, busy, idle, idle, idle]),
        sleep=sleeps.append,
    )
    assert ok and message == "GPU idle for 3 s (waited 7 s, 30000 MB free)"
    assert len(sleeps) == 7
    # Too little free memory is not idle.
    low = (0.0, 1000.0)
    ok, _ = wait(
        2, 60, 12000, read=scripted([low, low, idle, idle]), sleep=sleeps.append
    )
    assert ok
    ok, message = wait(2, 3, 12000, read=scripted([low] * 10), sleep=sleeps.append)
    assert not ok
    assert message == (
        "GPU still busy after 3 s (0% busy, 1000 MB free); running anyway"
    )
    assert wait(0, 60, read=scripted([])) == (True, "idle wait off")
    ok, message = wait(3, 60, read=scripted([None]))
    assert ok and "cannot read" in message


def test_a_quiet_command_replaces_the_idle_wait(tmp_path):
    ok, message = bench_compare.run_quiet_command("sh -c 'echo waiting; echo idle'")
    assert ok and message == "idle"
    ok, message = bench_compare.run_quiet_command("sh -c 'echo busy; exit 3'")
    assert not ok and message == "busy (exit status 3); running anyway"
    ok, message = bench_compare.run_quiet_command(str(tmp_path / "missing"))
    assert not ok and "did not start" in message
    gate = bench_compare.Gate(tmp_path / "gpu.lock", "sh -c 'echo idle'")
    with gate.hold() as held:
        assert held.quiet == "idle"
    assert (tmp_path / "gpu.lock").exists()


def test_a_run_past_its_timeout_is_killed_with_its_process_group(tmp_path, monkeypatch):
    monkeypatch.setattr(bench_compare, "STACKS_GRACE_S", 1.0)
    marker, log = tmp_path / "grandchild.pid", tmp_path / "run.log"
    script = textwrap.dedent(f"""
        import faulthandler, signal, subprocess, sys, time
        faulthandler.register(signal.SIGUSR1, all_threads=True)
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        open({str(marker)!r}, "w").write(str(child.pid))
        time.sleep(60)
        """)
    code = bench_compare._child([sys.executable, "-c", script], log, os.environ, 2.0)
    assert code is None
    # Before the kill, SIGUSR1 had the run write its stacks to its log.
    assert 'File "<string>"' in log.read_text()
    pid = int(marker.read_text())
    # The grandchild went with the group (gone, or a zombie not yet reaped).
    assert wait_for(lambda: _state(pid) in ("gone", "Z"), 5.0)
    assert (
        bench_compare._child(
            [sys.executable, "-c", "raise SystemExit(3)"], log, os.environ, 30.0
        )
        == 3
    )
    # A run that exits leaves nothing running: what it started dies with it.
    script = textwrap.dedent(f"""
        import subprocess, sys
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        open({str(marker)!r}, "w").write(str(child.pid))
        """)
    marker.unlink()
    assert (
        bench_compare._child([sys.executable, "-c", script], log, os.environ, 30.0) == 0
    )
    pid = int(marker.read_text())
    assert wait_for(lambda: _state(pid) in ("gone", "Z"), 5.0)
    # The run learns the driver's process ID.
    script = f"import os; print(os.environ[{bench_compare.DRIVER_ENV!r}])"
    assert (
        bench_compare._child([sys.executable, "-c", script], log, os.environ, 30.0) == 0
    )
    assert log.read_text().strip() == str(os.getpid())
    assert bench_compare._exit_note(-11) == "SIGSEGV"
    assert bench_compare._exit_note(None) == "timeout"


def _state(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0]
    except OSError:
        return "gone"


# --------------------------------------------------------------- summary


def result(median, *, stable=True, violations=(), sha="r" * 64, **extra):
    return {
        "status": "ok",
        "latency_ms": {"median": median, "p25": median * 0.99, "p75": median * 1.01},
        "timing_stability": {"stable": stable},
        "accuracy": {"mean_kl": 1e-4, "top1_agreement": 1.0},
        "accuracy_violations": list(violations),
        "reference": {"sha256": sha},
        "build_s": 10.0,
        **extra,
    }


def write_runs(out_dir, runs):
    for (round_name, config), value in runs.items():
        directory = out_dir / round_name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{config}.log").write_text("log\n")
        if value is not None:
            (directory / f"{config}.json").write_text(json.dumps(value))


def test_the_summary_takes_the_median_over_rounds(tmp_path):
    (tmp_path / "spec.json").write_text(
        json.dumps({"configs": {"trt-baseline": {}, "fast": {}, "broken": {}}})
    )
    write_runs(
        tmp_path,
        {
            ("r1", "trt-baseline"): result(2.0),
            ("r2", "trt-baseline"): result(2.4),
            ("r3", "trt-baseline"): result(2.2),
            ("r1", "fast"): result(1.0, stable=False),
            ("r2", "fast"): result(
                1.2, violations=["after timing: mean_kl=0.01 (needs <= 0.005)"]
            ),
            ("r3", "fast"): result(1.1),
            ("r1", "broken"): {
                "status": "failed",
                "stage": "build",
                "error": "E\nmore",
            },
            ("r2", "broken"): None,
            ("r3", "broken"): result(9.0),
        },
    )
    summary = bench_compare.summarize(tmp_path)
    assert list(summary["configs"]) == ["fast", "trt-baseline", "broken"]
    assert summary["rounds"] == 3
    # No trt-session ran: the spec's first configuration is the base.
    assert summary["base"] == "trt-baseline"
    fast = summary["configs"]["fast"]
    assert fast["median_ms"] == 1.1 and (fast["min_ms"], fast["max_ms"]) == (1.0, 1.2)
    assert fast["round_medians_ms"] == {"r1": 1.0, "r2": 1.2, "r3": 1.1}
    assert fast["vs_base"] == pytest.approx(0.5)
    assert fast["unstable_runs"] == ["r1"]
    assert fast["accuracy_violations"] == [
        "r2: after timing: mean_kl=0.01 (needs <= 0.005)"
    ]
    broken = summary["configs"]["broken"]
    assert (broken["runs"], broken["ok"], broken["median_ms"]) == (3, 1, 9.0)
    assert broken["failures"] == ["r1: build: E", "r2: no result (see the log)"]
    table = bench_compare.format_summary(summary)
    lines = table.splitlines()
    assert lines[0].split()[:3] == ["config", "median", "ms"]
    assert "vs trt-baseline" in lines[0]
    assert lines[1].startswith("fast") and "0.500x" in lines[1]
    assert "MISSES r2: after timing" in lines[1] and "unstable in r1" in lines[1]
    assert "2 failed" in lines[3]
    # trt-session is the default base once it ran; --base overrides.
    write_runs(tmp_path, {("r1", "trt-session"): result(1.0)})
    assert bench_compare.summarize(tmp_path)["base"] == "trt-session"
    assert bench_compare.summarize(tmp_path, base="fast")["base"] == "fast"
    table = bench_compare.write_summary(tmp_path, base="fast", write=False)
    assert "vs fast" in table and not (tmp_path / "summary.json").exists()
    assert bench_compare.write_summary(tmp_path, base="fast") == table
    assert json.loads((tmp_path / "summary.json").read_text())["base"] == "fast"
    with pytest.raises(SpecError, match="no runs of 'missing'"):
        bench_compare.write_summary(tmp_path, base="missing")


def code(commit="c" * 40, files=None, torch="2.16.0"):
    return {
        "environment": {
            "torch_tensorrt": "2.11.0.dev0",
            "torch_tensorrt_source": {"commit": commit, "uncommitted_sha256": None},
            "gpu": "any GPU",
        },
        "torch": torch,
        "workload": {"factory": "w.py:f", "files": files or {"w.py": "1" * 64}},
    }


def test_the_summary_warns_when_runs_ran_different_code(tmp_path):
    write_runs(
        tmp_path,
        {
            ("r1", "a"): result(1.0, **code()),
            ("r1", "b"): result(2.0, **code()),
            ("r2", "a"): result(1.0, **code()),
            # A failure that reached the measurement records its code too.
            ("r2", "b"): {"status": "failed", "stage": "build", **code()},
        },
    )
    summary = bench_compare.summarize(tmp_path)
    (version,) = summary["code"]
    assert summary["code"][version] == ["r1/a", "r1/b", "r2/a", "r2/b"]
    assert "WARNING" not in bench_compare.format_summary(summary)
    # Another commit, an edited workload directory or another PyTorch: the
    # runs are not comparable, and the table says so.
    write_runs(
        tmp_path,
        {
            ("r3", "a"): result(1.0, **code(commit="d" * 40)),
            ("r3", "b"): result(2.0, **code(files={"w.py": "2" * 64})),
            ("r4", "a"): result(1.0, **code(torch="2.17.0")),
            # Results the driver wrote, and older ones, say nothing.
            ("r4", "b"): {"status": "failed", "stage": "crash", "error": "x"},
        },
    )
    summary = bench_compare.summarize(tmp_path)
    assert len(summary["code"]) == 4
    assert sum(len(runs) for runs in summary["code"].values()) == 7
    table = bench_compare.format_summary(summary)
    assert "WARNING: the runs ran different code" in table
    assert f"{version[:12]} in r1/a, r1/b, r2/a, r2/b" in table
    assert bench_compare.code_version(code()) == version
    assert bench_compare.code_version({"status": "failed"}) is None
    assert bench_compare.code_version(None) is None


def test_the_summary_flags_unchecked_accuracy_and_mixed_references(tmp_path):
    legacy = result(1.0)
    del legacy["accuracy_violations"]
    write_runs(
        tmp_path,
        {
            ("r1", "old"): legacy,
            ("r1", "new"): result(2.0, sha="s" * 64),
            ("r2", "failed"): {"status": "failed", "stage": "recipe", "error": "bad"},
        },
    )
    summary = bench_compare.summarize(tmp_path)
    assert summary["configs"]["old"]["accuracy_violations"] is None
    assert summary["configs"]["new"]["accuracy_violations"] == []
    assert summary["references"] == ["r" * 64, "s" * 64]
    table = bench_compare.format_summary(summary)
    assert "not checked" in table
    assert "WARNING: the runs used different reference outputs" in table
    assert re.search(r"failed\s+FAILED\s+0/1\s+r2: recipe: bad", table)
    assert bench_compare.format_summary(bench_compare.summarize(tmp_path / "x")) == (
        f"{tmp_path / 'x'}: no runs yet"
    )


def test_run_lines_say_what_a_run_measured_or_why_it_failed():
    line = bench_compare.run_line
    assert line(result(1.5)) == (
        "1.5000 ms/unit [1.4850, 1.5150] mean_kl=0.0001 top1_agreement=1"
    )
    assert "UNSTABLE" in line(result(1.5, stable=False))
    # Results written before the Auto-Tuner rename call the record measurement.
    earlier = {**result(1.5), "measurement": {"stable": False}}
    del earlier["timing_stability"]
    assert "UNSTABLE" in line(earlier)
    assert line(result(1.5, violations=["first call: x"])).endswith(
        "MISSES first call: x"
    )
    assert line(None) == "FAILED at no result (see the log)"
    assert line({"status": "failed", "stage": "time", "error": "Boom\nTrace"}) == (
        "FAILED at time: Boom"
    )


# ------------------------------------------------------ end to end (CPU)

TINY = """
import os
import time

import torch
from torch_tensorrt.autotune import Candidate, EditableFile, Knob, Timing, Workload

# The editable file's fenced block (an edited version replaces this file).
# EDITABLE-START
SCALE = None
# EDITABLE-END

NAMES = ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "TMPDIR", "HF_HUB_OFFLINE")


class TinyCandidate(Candidate):
    def __init__(self, scale, cache):
        self.scale = scale
        self.cache = cache

    def __call__(self):
        return torch.ones(4) * self.scale

    def describe(self):
        return {
            **{name: os.environ.get(name) for name in NAMES},
            "inductor_cache_at_build": self.cache,
        }


class TinyWorkload(Workload):
    name = "tiny"

    def knobs(self):
        return (Knob("workload.scale", (1, 2), 1, "output scale"),)

    def editable_files(self):
        return (EditableFile("tiny", __file__, "this file"),)

    def build(self, values):
        # What Inductor's cache held when the build started.
        inductor = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        cache = sorted(os.listdir(inductor)) if inductor and os.path.isdir(inductor) else []
        # bench.crash and bench.hang name a file; they act while it exists.
        crash = values.get("bench.crash")
        if crash and os.path.exists(crash):
            os._exit(3)
        hang = values.get("bench.hang")
        if hang and os.path.exists(hang):
            # Leave a cache entry, say which process hangs, and hang.
            os.makedirs(inductor, exist_ok=True)
            open(os.path.join(inductor, "warm"), "w").close()
            with open(hang + ".tmp", "w") as stream:
                stream.write(str(os.getpid()))
            os.replace(hang + ".tmp", hang + ".pid")
            time.sleep(600)
        scale = values["workload.scale"]
        if values.get("edit.tiny"):
            scale = float(open(values["edit.tiny"]).read())
        return TinyCandidate(scale, cache)

    def reference(self):
        return torch.ones(4)

    def compare(self, outputs, reference):
        return {"max_abs_error": float((outputs - reference).abs().max())}

    def accuracy_defaults(self):
        return {"max_abs_error": ("<=", 0.5)}

    def timing(self):
        return Timing(warmup=1, rounds=3, calls=1, profile_calls=0)


def create_workload():
    return TinyWorkload()
"""


@pytest.fixture
def tiny_spec(tmp_path):
    (tmp_path / "tiny_bench_workload.py").write_text(TINY)
    (tmp_path / "crash.flag").touch()
    spec = {
        "workload": f"{tmp_path / 'tiny_bench_workload.py'}:create_workload",
        "reference": "reference.pt",
        "timeout_s": 300,
        "noise_control": {"quiet_gpu_s": 0, "retries": 0, "max_spread": 10.0},
        "head_to_head": ["plain", "wrong", "invalid", "crash"],
        "configs": {
            "plain": {"recipe": {}},
            "wrong": {"recipe": {"workload.scale": 2}},
            "invalid": {"recipe": {"workload.missing": 1}},
            "crash": {"bench": {"crash": str(tmp_path / "crash.flag")}},
            "hang": {"bench": {"hang": str(tmp_path / "hang.flag")}},
        },
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return path


def test_one_run_measures_in_this_process(tiny_spec, tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    reference = tmp_path / "out" / "reference.pt"
    reference.parent.mkdir()
    with pytest.raises(SpecError, match="does not exist"):
        bench_compare.run(
            tiny_spec, "plain", tmp_path / "plain.json", reference=reference
        )
    bench_compare.make_reference(tiny_spec, tmp_path / "out")
    assert reference.is_file()
    plain = bench_compare.run(
        tiny_spec, "plain", tmp_path / "plain.json", reference=reference
    )
    assert plain["status"] == "ok" and plain["accuracy_violations"] == []
    assert plain["accuracy_bounds"] == {"max_abs_error": ["<=", 0.5]}
    assert (
        plain["values"]["workload.scale"] == 1 and plain["values"]["edit.tiny"] is None
    )
    assert plain["reference"]["sha256"] == bench_compare._sha256(reference)
    assert json.loads((tmp_path / "plain.json").read_text())["config"] == "plain"
    wrong = bench_compare.run(
        tiny_spec, "wrong", tmp_path / "wrong.json", reference=reference
    )
    assert wrong["accuracy_violations"] == [
        "first call: max_abs_error=1 (needs <= 0.5)",
        "after timing: max_abs_error=1 (needs <= 0.5)",
    ]
    invalid = bench_compare.run(
        tiny_spec, "invalid", tmp_path / "invalid.json", reference=reference
    )
    assert (invalid["status"], invalid["stage"]) == ("failed", "recipe")
    assert "workload.missing" in invalid["error"]


def test_an_edited_session_config_runs_its_edited_file(
    tiny_spec, tmp_path, monkeypatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    digest = "b" * 64
    spec = json.loads(tiny_spec.read_text())
    session = make_session(
        tmp_path / "session",
        {"edit.tiny": digest},
        kwargs={},
        factory=f"{tmp_path.name}/tiny_bench_workload.py:create_workload",
    )
    reference = tmp_path / "reference.pt"
    bench_compare.make_reference(tiny_spec, reference=reference)
    spec["configs"]["trt-session"] = {"recipe": {}}
    tiny_spec.write_text(json.dumps(spec))
    # The edited file is missing until the session stores it.
    missing = bench_compare.run(
        tiny_spec,
        "trt-session",
        tmp_path / "s.json",
        reference=reference,
        session=session,
    )
    assert missing["stage"] == "recipe" and "does not exist" in missing["error"]
    (session / "edited_files").mkdir()
    (session / "edited_files" / f"{digest}.py").write_text("1.0")
    edited = bench_compare.run(
        tiny_spec,
        "trt-session",
        tmp_path / "s.json",
        reference=reference,
        session=session,
    )
    assert edited["status"] == "ok", edited
    assert edited["values"]["edit.tiny"] == str(
        session / "edited_files" / f"{digest}.py"
    )
    # An editable file the workload does not declare, or an edited file
    # without its session, is refused.
    spec["configs"]["other"] = {"recipe": {"edit.other": digest}}
    spec["configs"]["unsourced"] = {"recipe": {"edit.tiny": digest}}
    tiny_spec.write_text(json.dumps(spec))
    other = bench_compare.run(
        tiny_spec, "other", tmp_path / "o.json", reference=reference
    )
    assert "does not declare" in other["error"]
    unsourced = bench_compare.run(
        tiny_spec, "unsourced", tmp_path / "u.json", reference=reference
    )
    assert "pass --session" in unsourced["error"]


def test_all_runs_every_config_in_a_fresh_process(
    tiny_spec, tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    out = tmp_path / "out"
    gate = bench_compare.Gate(tmp_path / "gpu.lock", quiet_s=0)
    summary = bench_compare.run_all(tiny_spec, out, rounds=1, gate=gate)
    printed = capsys.readouterr().out
    # The reference first, then each run, then the table.
    assert printed.index("reference ") < printed.index(
        "r1 order: plain wrong crash invalid"
    )
    assert (out / "reference.pt").is_file() and (tmp_path / "gpu.lock").exists()
    resolved = json.loads((out / "spec.json").read_text())
    assert resolved["reference"] == str(out / "reference.pt")
    assert resolved["source"]["path"] == str(tiny_spec)
    plain = json.loads((out / "r1" / "plain.json").read_text())
    assert plain["status"] == "ok" and plain["accuracy_violations"] == []
    # A cold run: compiler caches and TMPDIR of its own, removed afterwards;
    # offline, since computing the reference fetched the model.
    scratch = out / "r1" / "cache" / "plain"
    assert plain["structure"] == {
        "TORCHINDUCTOR_CACHE_DIR": str(scratch / "inductor"),
        "TRITON_CACHE_DIR": str(scratch / "triton"),
        "TMPDIR": str(scratch / "tmp"),
        "HF_HUB_OFFLINE": "1",
        "inductor_cache_at_build": [],
    }
    assert not (out / "r1" / "cache").exists()
    # The run recorded the code it measured.
    assert plain["workload"]["files"][str(tiny_spec.parent / "tiny_bench_workload.py")]
    assert bench_compare.code_version(plain) in summary["code"]
    crash = json.loads((out / "r1" / "crash.json").read_text())
    assert (crash["stage"], crash["error"], crash["written_by"]) == (
        "crash",
        "exited with status 3 and no result",
        "driver",
    )
    assert "log_tail" in crash and (out / "r1" / "crash.log").is_file()
    configs = summary["configs"]
    # Fastest first (plain and wrong take about the same time), failures last.
    assert set(list(configs)[:2]) == {"plain", "wrong"}
    assert list(configs)[2:] == ["crash", "invalid"]
    assert configs["wrong"]["accuracy_violations"] == [
        "r1: after timing: max_abs_error=1 (needs <= 0.5)",
        "r1: first call: max_abs_error=1 (needs <= 0.5)",
    ]
    assert configs["invalid"]["failures"][0].startswith("r1: recipe: ")
    assert json.loads((out / "summary.json").read_text())["base"] == "plain"
    assert "FAILED at crash: exited with status 3" in printed
    # The same directory again: refused unless resumed. Resumed, the runs
    # that left a result stay and the new round runs.
    with pytest.raises(SpecError, match="pass --resume"):
        bench_compare.run_all(tiny_spec, out, rounds=1, gate=gate)
    summary = bench_compare.run_all(
        tiny_spec, out, ["plain"], rounds=2, gate=gate, resume=True
    )
    printed = capsys.readouterr().out
    assert re.search(r"r1 plain\s+kept", printed)
    assert summary["configs"]["plain"]["runs"] == 2
    assert (out / "r2" / "plain.json").is_file() and not (
        out / "r2" / "wrong.json"
    ).exists()
    # A run that crashed (a result the driver wrote) runs again; a failure
    # the run reported itself is a result and stays.
    (tmp_path / "crash.flag").unlink()
    summary = bench_compare.run_all(
        tiny_spec, out, ["crash", "invalid"], rounds=1, gate=gate, resume=True
    )
    printed = capsys.readouterr().out
    assert re.search(r"r1 invalid\s+kept\s+FAILED at recipe", printed)
    assert re.search(r"r1 crash\s+exit=0", printed)
    assert summary["configs"]["crash"]["ok"] == 1
    assert summary["configs"]["invalid"]["ok"] == 0
    # A changed spec cannot continue the directory.
    spec = json.loads(tiny_spec.read_text())
    spec["configs"]["plain"]["recipe"] = {"workload.scale": 2}
    tiny_spec.write_text(json.dumps(spec))
    with pytest.raises(SpecError, match="differs in config plain"):
        bench_compare.run_all(tiny_spec, out, rounds=1, gate=gate, resume=True)


def test_a_run_stops_with_its_driver_and_runs_again_cold(
    tiny_spec, tmp_path, monkeypatch
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    out, lock = tmp_path / "out", tmp_path / "gpu.lock"
    flag = tmp_path / "hang.flag"
    marker = Path(f"{flag}.pid")
    flag.touch()
    command = [
        *bench_compare._python("all", str(tiny_spec), str(out), "hang"),
        *("--rounds", "1", "--lock", str(lock), "--quiet-s", "0", "--resume"),
    ]
    # A driver stopped by SIGTERM kills its run; one killed outright takes
    # its run along (the parent-death signal). Either way the lock is free
    # once the run is gone, and the run left no result.
    outputs = {}
    for stop, status in ((signal.SIGTERM, 143), (signal.SIGKILL, -9)):
        marker.unlink(missing_ok=True)
        driver = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            assert wait_for(marker.exists, 120.0), "the run never started"
            run = int(marker.read_text())
            # The run hangs under the lock.
            assert not lock_is_free(lock)
            os.kill(driver.pid, stop)
            outputs[stop] = driver.communicate(timeout=60)[0]
        finally:
            if driver.poll() is None:
                driver.kill()
                driver.wait()
        assert driver.returncode == status, outputs.get(stop)
        assert wait_for(lambda: _state(run) in ("gone", "Z"), 30.0), stop
        assert wait_for(lambda: lock_is_free(lock), 30.0), stop
        assert not (out / "r1" / "hang.json").exists()
    assert "bench_compare.py: stopped by SIGTERM" in outputs[signal.SIGTERM]
    # The killed attempt left its caches behind ...
    assert (out / "r1" / "cache" / "hang" / "inductor" / "warm").is_file()
    # ... which the next attempt does not see: it builds cold.
    flag.unlink()
    gate = bench_compare.Gate(lock, quiet_s=0)
    summary = bench_compare.run_all(
        tiny_spec, out, ["hang"], rounds=1, gate=gate, resume=True
    )
    result = json.loads((out / "r1" / "hang.json").read_text())
    assert result["status"] == "ok"
    assert result["structure"]["inductor_cache_at_build"] == []
    assert summary["configs"]["hang"]["ok"] == 1
    assert not (out / "r1" / "cache").exists()


def test_the_command_line(tiny_spec, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(bench_compare.QUIET_ENV, raising=False)
    main = bench_compare.main
    (tmp_path / "empty").mkdir()
    assert main(["summary", str(tmp_path / "empty")]) == 0
    assert "no runs yet" in capsys.readouterr().out
    # summary only reads the directory unless told to write.
    assert not (tmp_path / "empty" / "summary.json").exists()
    assert main(["summary", str(tmp_path / "empty"), "--write"]) == 0
    assert (tmp_path / "empty" / "summary.json").is_file()
    assert main(["summary", str(tmp_path / "missing")]) == 2
    assert "is not a directory" in capsys.readouterr().err
    assert not (tmp_path / "missing").exists()
    # Configs before or after the options.
    head = ["all", str(tiny_spec), str(tmp_path / "o")]
    for tail in (
        ["plain", "nope", "--lock", "none"],
        ["--lock", "none", "--rounds", "2", "plain", "nope"],
        ["plain", "--lock", "none", "nope"],
    ):
        assert main(head + tail) == 2
        assert "has no configs ['nope']" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(head + ["--lock", "none", "plain", "--nope"])
    assert "unrecognized arguments: --nope" in capsys.readouterr().err
    # A template's reference is relative to an output directory, which run
    # does not take.
    assert main(["run", str(tiny_spec), "plain", str(tmp_path / "p.json")]) == 2
    assert "pass --reference PATH" in capsys.readouterr().err
    assert main(["quiet", "--quiet-command", "sh -c 'echo idle'"]) == 0
    assert capsys.readouterr().out == "idle\n"
    assert main(["quiet", "--quiet-command", "sh -c 'exit 1'"]) == 1
    assert main(["quiet", "--quiet-s", "0"]) == 0
    with pytest.raises(SystemExit):
        main([])
