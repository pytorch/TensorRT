# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""The records of the kernels generation registers or misses, and how builds
pinned to cached kernels revalidate them, without agent or GPU access."""

import json
import logging
import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch_tensorrt.dynamo.region_compilation._generation import _factory
from torch_tensorrt.region import (
    GeneratedKernel,
    MissingKernel,
    RegionError,
    generated_kernels,
    missing_kernels,
    record_generated_kernel,
    record_missing_kernel,
)

from ._generation_utils import _candidate, _generate, _passed

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Kernel generation requires a POSIX host"
)


def new_records(before):
    """The kernels recorded since ``generated_kernels()`` had ``before`` entries."""
    return generated_kernels()[before:]


def generate(factory, region_id, memo, samples=None):
    return _factory.generate_registration(
        factory.graph,
        factory.config,
        region_id=region_id,
        memo=memo,
        samples=samples,
    )


def test_a_search_and_a_cache_hit_record_the_kernel_with_its_measurement(factory):
    factory.author.return_value = _candidate("fresh")
    factory.evaluate.return_value = _passed(0.2)
    before = len(generated_kernels())
    assert _generate(factory) == _candidate("fresh")
    (new,) = new_records(before)
    published = next(factory.root.glob("*/accepted.json"))
    accepted = json.loads(published.read_text())
    assert (new.region, new.region_id) == ("activation", "activation")
    assert new.task_sha256 == accepted["task_sha256"] == published.parent.name
    assert new.kernel_sha256 == accepted["candidate_sha256"]
    assert (new.origin, new.backend, new.provider, new.model) == (
        "new",
        "triton",
        "claude",
        None,
    )
    # From the final validation that accepted it: kernel 0.2 ms, eager 1.0 ms.
    assert (new.latency_ms, new.baseline, new.baseline_latency_ms) == (
        0.2,
        "eager",
        1.0,
    )
    assert new.speedup == pytest.approx(5.0)
    assert new.generation_s >= 0
    # The run that accepted the kernel keeps its own copy of the manifest.
    assert Path(new.manifest).parent.name.startswith("run-")

    # A later compilation reuses the cached kernel after revalidating it; the
    # provider is the one that wrote it.
    before = len(generated_kernels())
    assert _generate(factory, provider="codex") == _candidate("fresh")
    (cached,) = new_records(before)
    assert cached.origin == "cached" and cached.provider == "claude"
    assert cached.kernel_sha256 == new.kernel_sha256
    assert cached.task_sha256 == new.task_sha256
    assert cached.manifest == str(published) and cached.speedup == new.speedup
    assert json.loads(json.dumps(cached.to_json())) == {
        **new.to_json(),
        "origin": "cached",
        "generation_s": cached.generation_s,
        "manifest": str(published),
    }


def test_the_tensorrt_objective_measures_against_the_tensorrt_engine(factory):
    factory.author.return_value = _candidate("fast")
    # The eager region takes 1.0 ms, the TensorRT engine of it 0.5 ms.
    measured = {**_passed(0.2), "objective": "tensorrt", "baseline_latency_ms": 0.5}
    factory.evaluate.return_value = measured
    factory.finalize.return_value = measured
    before = len(generated_kernels())
    _generate(factory, objective="tensorrt")
    (kernel,) = new_records(before)
    assert (kernel.baseline, kernel.baseline_latency_ms) == ("tensorrt", 0.5)
    assert kernel.speedup == pytest.approx(2.5)


@pytest.mark.parametrize("use_cache", [True, False])
def test_identical_regions_record_the_kernel_they_share(factory, use_cache):
    factory.author.return_value = _candidate("shared")
    factory.evaluate.return_value = _passed(0.2)
    factory.config = replace(factory.config, use_cache=use_cache)
    before = len(generated_kernels())
    memo = {}
    for index in range(3):
        generate(factory, f"block#{index}", memo)
    first, *others = new_records(before)
    assert first.origin == "new" and first.region == "block"
    assert [kernel.region_id for kernel in others] == ["block#1", "block#2"]
    assert all(kernel.origin == "shared" for kernel in others)
    assert all(
        (kernel.kernel_sha256, kernel.speedup, kernel.manifest)
        == (first.kernel_sha256, first.speedup, first.manifest)
        for kernel in others
    )


def replacing(factory):
    """Script a search that finds kernel "a", then "b" for a region "a" fails.

    "a" fails the data of a region whose samples hold 2.0, so that region
    searches again, on its identical regions' data too, and finds "b".
    """
    factory.author.side_effect = [_candidate("a"), _candidate("b")]
    factory.evaluate.return_value = _passed(0.2)

    def finalize(request, directory, timeout):
        samples = torch.load(request["samples_path"], weights_only=True)
        values = [float(sample[0][0]) for sample in samples]
        if request["candidate"]["kernel_name"] == "a" and 2.0 in values:
            return {"status": "error", "stage": "correctness", "error": "mismatch"}
        return _passed(0.1 if request["candidate"]["kernel_name"] == "b" else 0.2)

    factory.finalize.side_effect = finalize


def sample(value):
    return [(torch.full((7,), value),)]


def test_a_region_records_the_kernel_a_later_identical_region_replaced(factory):
    replacing(factory)
    before = len(generated_kernels())
    memo = {}
    generate(factory, "block#0", memo, sample(1.0))
    generate(factory, "block#1", memo, sample(2.0))
    first, second = new_records(before)
    assert first.kernel_sha256 != second.kernel_sha256
    assert (first.origin, second.origin) == ("new", "new")
    # Every identical region then uses "b", and its record says so.
    for region_id in ("block#0", "block#1"):
        _factory.final_registration(
            factory.graph, factory.config, region_id=region_id, memo=memo
        )
    first, second = new_records(before)
    assert first.region_id == "block#0" and first.origin == "shared"
    assert first.kernel_sha256 == second.kernel_sha256
    assert first.speedup == second.speedup == pytest.approx(10.0)


def test_a_compilation_updates_only_its_own_records(factory):
    replacing(factory)
    before = len(generated_kernels())
    # An earlier compilation of the same region kept "a".
    earlier = {}
    generate(factory, "block#0", earlier, sample(1.0))
    # A later one found "b" for its second region; its first region shares it.
    later = {}
    generate(factory, "block#0", later, sample(1.0))
    generate(factory, "block#1", later, sample(2.0))
    for region_id in ("block#0", "block#1"):
        _factory.final_registration(
            factory.graph, factory.config, region_id=region_id, memo=later
        )
    # The earlier compilation finishing afterwards changes neither its own
    # record nor the later compilation's records of the same region.
    _factory.final_registration(
        factory.graph, factory.config, region_id="block#0", memo=earlier
    )
    old, first, second = new_records(before)
    assert old.region_id == first.region_id == "block#0"
    assert old.kernel_sha256 != second.kernel_sha256
    assert first.kernel_sha256 == second.kernel_sha256 and first.origin == "shared"


def test_a_record_that_cannot_be_updated_never_fails_a_compilation(
    factory, monkeypatch, caplog
):
    replacing(factory)
    memo = {}
    generate(factory, "block#0", memo, sample(1.0))
    generate(factory, "block#1", memo, sample(2.0))

    def broken(old, source):
        raise RuntimeError("the records are gone")

    monkeypatch.setattr(_factory, "_share_generated_kernel", broken)
    with caplog.at_level(logging.WARNING, logger=_factory.__name__):
        registration = _factory.final_registration(
            factory.graph, factory.config, region_id="block#0", memo=memo
        )
    assert registration == _candidate("b")
    assert "Could not update the kernel record of region 'block#0'" in caplog.text


def test_a_build_without_generation_records_the_kernel_it_misses(factory, monkeypatch):
    factory.root.mkdir()
    before = len(missing_kernels())
    with pytest.raises(RegionError, match="no accepted kernel is cached"):
        _generate(factory, allow_generation=False)
    (missing,) = missing_kernels()[before:]
    assert (missing.region, missing.region_id) == ("activation", "activation")
    assert missing.reason.startswith("no accepted kernel is cached for this task")
    # It names the task whose kernel a search then generates.
    factory.author.return_value = _candidate("fresh")
    factory.evaluate.return_value = _passed(0.2)
    registered = len(generated_kernels())
    _generate(factory)
    (kernel,) = generated_kernels()[registered:]
    assert kernel.task_sha256 == missing.task_sha256
    # A build that finds its kernel misses none.
    monkeypatch.setenv("TORCH_TENSORRT_REGION_GENERATION", "0")
    assert _generate(factory) == _candidate("fresh")
    assert missing_kernels()[before:] == (missing,)


def test_the_generation_switch_revalidates_cached_kernels_for_correctness_only(
    factory, monkeypatch
):
    factory.author.return_value = _candidate("published")
    accepted = {**_passed(1.0), "baseline_latency_ms": 1.2}
    factory.evaluate.return_value = accepted
    factory.finalize.return_value = accepted
    objective = {"objective": "eager", "min_speedup": 1.1}
    _generate(factory, **objective)
    # Remeasured on a busy GPU, the kernel misses min_speedup: with
    # allow_generation=False its full revalidation refuses it.
    factory.finalize.return_value = {**_passed(1.0), "baseline_latency_ms": 1.0}
    with pytest.raises(RegionError, match="remeasured speedup"):
        _generate(factory, allow_generation=False, **objective)
    assert factory.finalize.call_args.args[0]["objective"] == "eager"
    # A build pinned through the environment switch rechecks correctness only:
    # the kernel's speed was measured when it was accepted.
    monkeypatch.setenv("TORCH_TENSORRT_REGION_GENERATION", "0")
    assert _generate(factory, **objective) == _candidate("published")
    assert factory.finalize.call_args.args[0]["objective"] == "candidate"
    # A wrong kernel still fails it.
    factory.finalize.return_value = {
        "status": "error",
        "stage": "correctness",
        "error": "Fresh inputs disagree",
    }
    with pytest.raises(RegionError, match="failed revalidation"):
        _generate(factory, **objective)
    factory.author.assert_called_once()


def test_a_failed_generation_records_nothing(factory):
    factory.author.return_value = _candidate("wrong")
    factory.evaluate.return_value = {
        "status": "error",
        "stage": "correctness",
        "error": "Output 0 is wrong",
    }
    before = len(generated_kernels())
    with pytest.raises(RegionError):
        _generate(factory)
    assert new_records(before) == ()


def test_recording_never_fails_a_compilation(factory, monkeypatch, caplog):
    factory.author.return_value = _candidate("fresh")
    factory.evaluate.return_value = _passed(0.2)

    def broken(kernel):
        raise OSError("no space left")

    monkeypatch.setattr(_factory, "record_generated_kernel", broken)
    with caplog.at_level(logging.WARNING, logger=_factory.__name__):
        assert _generate(factory) == _candidate("fresh")
    assert "Could not record the kernel of region 'activation'" in caplog.text


def test_records_are_checked_and_others_may_add_them():
    kernel = GeneratedKernel(
        region="norm",
        region_id="norm#0",
        task_sha256="t" * 64,
        kernel_sha256="k" * 64,
        backend="cuda_cpp",
        origin="cached",
    )
    before = len(generated_kernels())
    record_generated_kernel(kernel)
    assert new_records(before) == (kernel,)
    with pytest.raises(TypeError, match="Expected a GeneratedKernel"):
        record_generated_kernel(kernel.to_json())
    with pytest.raises(ValueError, match="origin must be one of"):
        replace(kernel, origin="downloaded")
    with pytest.raises(ValueError, match="kernel_sha256 must be a nonempty string"):
        replace(kernel, kernel_sha256="")


def test_missing_kernel_records_are_checked_and_others_may_add_them():
    missing = MissingKernel(
        region="norm",
        region_id="norm#0",
        task_sha256="t" * 64,
        reason="no accepted kernel is cached for this task",
    )
    before = len(missing_kernels())
    record_missing_kernel(missing)
    assert missing_kernels()[before:] == (missing,)
    assert missing.to_json()["task_sha256"] == "t" * 64
    with pytest.raises(TypeError, match="Expected a MissingKernel"):
        record_missing_kernel(missing.to_json())
    with pytest.raises(ValueError, match="task_sha256 must be a nonempty string"):
        replace(missing, task_sha256="")
