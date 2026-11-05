# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Real model batches become region validation cases, saved and checked as data."""

import json
import logging
import os
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from filelock import FileLock

from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.region_compilation import (
    _plugins,
    materialize_regions,
    normalize_region_scopes,
)
from torch_tensorrt.dynamo.region_compilation._generation import (
    _factory,
    _samples,
    _worker,
)
from torch_tensorrt.region import KernelGenerator, RegionError, fused_kernel
from torch_tensorrt.region._session import RegionCompilationSession

from ._generation_utils import _candidate, _passed, _proposal

# The real controller, for tests that generate through materialize_regions.
_GENERATE = _factory.generate_registration

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Kernel generation requires a POSIX host"
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


class Batches:
    """A reusable iterable that counts the batches drawn from it."""

    def __init__(self, batches):
        self.batches = batches
        self.drawn = 0

    def __iter__(self):
        for batch in self.batches:
            self.drawn += 1
            yield batch


class Projected(torch.nn.Module):
    """A region fed by a weighted layer and by a model input."""

    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x, y):
        x = self.linear(x)
        with fused_kernel(generator=self.generator, name="mixed"):
            return torch.relu(x + y)


class Stacked(torch.nn.Module):
    """Two generated regions that share one generator."""

    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, x):
        with fused_kernel(generator=self.generator, name="first"):
            x = torch.relu(x + 1)
        x = x * 0.5
        with fused_kernel(generator=self.generator, name="second"):
            return torch.relu(x - 1)


class Twin(torch.nn.Module):
    """Two identical generated regions, as repeated layers have."""

    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, x):
        with fused_kernel(generator=self.generator, name="first"):
            x = torch.relu(x + 1)
        x = x * 0.5 - 3.0
        with fused_kernel(generator=self.generator, name="second"):
            return torch.relu(x + 1)


class Dense(torch.nn.Module):
    """A region fed by a weighted layer."""

    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x):
        x = self.linear(x)
        with fused_kernel(generator=self.generator, name="dense"):
            return torch.relu(x + 1)


class Noisy(torch.nn.Module):
    """A region whose inputs change in every run, as after atomics upstream."""

    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, x):
        x = x + torch.rand_like(x)
        with fused_kernel(generator=self.generator, name="noisy"):
            return torch.relu(x + 1)


class Keyword(torch.nn.Module):
    """A region fed by a positional input and by nested keyword inputs."""

    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, x, *, mask, pair):
        with fused_kernel(generator=self.generator, name="masked"):
            return torch.relu(x + mask + pair["a"] - pair["b"])


class Scaled(torch.nn.Module):
    """A region that reads a parameter."""

    def __init__(self, generator):
        super().__init__()
        self.generator = generator
        self.scale = torch.nn.Parameter(torch.arange(4.0))

    def forward(self, x):
        with fused_kernel(generator=self.generator, name="scaled"):
            return torch.relu(x * self.scale)


def _keywords():
    return {"mask": torch.zeros(4), "pair": {"a": torch.zeros(4), "b": torch.zeros(4)}}


def _materialize(
    model, *inputs, kwargs=None, settings=None, generate=None, register=None
):
    """Capture regions and materialize them with generation stubbed out."""
    generate = generate or Mock(return_value="registration")
    register = register or Mock(return_value=torch.ops.aten.relu.default)
    session = RegionCompilationSession(strict=True)
    with session:
        program = torch.export.export(model.eval(), inputs, kwargs, strict=True)
        program = normalize_region_scopes(program)
        captured = program.graph_module
        graph = program.run_decompositions().module()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_factory, "generate_registration", generate)
        patch.setattr(_plugins, "register_region_plugin", register)
        with session:
            materialize_regions(graph, captured, settings or CompilationSettings())
    return generate


@pytest.mark.parametrize(
    "batches",
    [
        torch.ones(2, 4),
        iter([torch.ones(4)]),
        (torch.ones(4) for _ in range(2)),
        {"x": torch.ones(4)},
        "batches",
        4,
    ],
    ids=["tensor", "iterator", "generator", "mapping", "string", "number"],
)
def test_validation_inputs_must_be_reusable_batches(batches):
    with pytest.raises(TypeError, match="reusable iterable of batches"):
        KernelGenerator(validation_inputs=batches)


def test_validation_batches_leave_generator_identity_and_repr_alone():
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.ones(8)))
    for batches in ([torch.ones(4)], (torch.ones(4),), loader):
        generator = KernelGenerator(validation_inputs=batches)
        assert generator.validation_inputs is batches
        assert generator == KernelGenerator()
        assert hash(generator) == hash(KernelGenerator())
        assert "validation_inputs" not in repr(generator)


@requires_cuda
def test_region_inputs_are_captured_from_real_batches_and_weights():
    torch.manual_seed(0)
    batches = [
        (torch.randn(4, 8, device="cuda"), torch.randn(4, 8, device="cuda")),
        # DataLoaders yield lists of CPU tensors; they move to the inputs' device.
        [torch.randn(4, 8), torch.randn(4, 8)],
    ]
    model = Projected(KernelGenerator(validation_inputs=batches)).cuda()
    generate = _materialize(
        model, torch.zeros(4, 8, device="cuda"), torch.zeros(4, 8, device="cuda")
    )
    samples = generate.call_args.kwargs["samples"]
    assert len(samples) == 2
    with torch.no_grad():
        for sample, (x, y) in zip(samples, batches):
            expected = (model.linear(x.cuda()), y)
            for value, wanted in zip(sample, expected):
                assert value.device.type == "cpu" and value.is_contiguous()
                torch.testing.assert_close(value, wanted.cpu())


def test_regions_sharing_batches_share_model_runs_and_use_at_most_eight():
    batches = Batches([(torch.full((4,), float(index)),) for index in range(10)])
    generate = _materialize(
        Stacked(KernelGenerator(validation_inputs=batches)), torch.zeros(4)
    )
    assert batches.drawn == _samples.MAX_SAMPLES == 8
    first, second = (call.kwargs["samples"] for call in generate.call_args_list)
    assert [float(sample[0][0]) for sample in first] == [float(i) for i in range(8)]
    assert [float(sample[0][0]) for sample in second] == [
        (max(i + 1, 0) * 0.5) for i in range(8)
    ]


def test_materialize_drops_its_samples_after_each_region():
    # Only materialize_regions' own references are checked: the controller's
    # per-compilation memo keeps samples that identical regions must also pass.
    previous = []
    released = []

    def generate(child, config, *, samples, **kwargs):
        released.append(all(ref() is None for ref in previous))
        previous[:] = [weakref.ref(value) for sample in samples for value in sample]
        return "registration"

    batches = [torch.full((4,), 1.0)]
    _materialize(
        Stacked(KernelGenerator(validation_inputs=batches)),
        torch.zeros(4),
        generate=generate,
    )
    assert released == [True, True]


def test_regions_without_validation_batches_get_no_samples():
    generate = _materialize(Stacked(KernelGenerator()), torch.zeros(4))
    assert [call.kwargs["samples"] for call in generate.call_args_list] == [None] * 2


def test_dry_runs_do_not_run_validation_batches():
    batches = Batches([torch.zeros(4)])
    generate = _materialize(
        Stacked(KernelGenerator(validation_inputs=batches)),
        torch.zeros(4),
        settings=CompilationSettings(dryrun=True),
    )
    generate.assert_not_called()
    assert batches.drawn == 0


@pytest.mark.parametrize(
    "batches, message",
    [
        (
            [torch.zeros(2)],
            r"batch 1 input 0 \(x\) is float32 \[2\]; .* float32 \[4\]",
        ),
        (
            [torch.zeros(4), torch.zeros(4).half()],
            r"batch 2 input 0 \(x\) is float16 \[4\]",
        ),
        ([(torch.zeros(4), torch.zeros(4))], "batch 1 has 2 positional inputs; .* 1"),
        ([{"x": torch.zeros(4)}], "batch 1 has 0 positional inputs; .* takes 1"),
        ([None], "batch 1 must be a tensor, a tuple of tensors, or a dict"),
        ([], "produced no batches"),
    ],
)
def test_batches_that_do_not_fit_the_compiled_inputs_fail_before_generation(
    batches, message
):
    generate = Mock()
    with pytest.raises(RegionError, match=message):
        _materialize(
            Stacked(KernelGenerator(validation_inputs=batches)),
            torch.ones(4),
            generate=generate,
        )
    generate.assert_not_called()


@pytest.mark.parametrize("container", [tuple, list])
def test_keyword_inputs_follow_the_exported_input_spec(container):
    x, mask, a, b = (torch.full((4,), value) for value in (1.0, 10.0, 1000.0, 100.0))
    # Keyword inputs follow the positional ones as a dict, in any key order.
    batch = container([x, {"pair": {"b": b, "a": a}, "mask": mask}])
    generate = _materialize(
        Keyword(KernelGenerator(validation_inputs=[batch])),
        torch.zeros(4),
        kwargs=_keywords(),
    )
    region = generate.call_args.args[0]
    (sample,) = generate.call_args.kwargs["samples"]
    torch.testing.assert_close(region(*sample)[0], torch.relu(x + mask + a - b))


@pytest.mark.parametrize(
    "batch, message",
    [
        (
            (torch.ones(4),),
            r"keyword inputs \[\]; the compiled model takes \['mask', 'pair'\]",
        ),
        (
            (torch.ones(4), {"mask": torch.ones(4), "pair": (torch.ones(4),) * 2}),
            r"structure of the compiled inputs: .* expected <class 'dict'>",
        ),
        (_keywords(), "has 0 positional inputs; the compiled model takes 1"),
    ],
    ids=["missing", "structure", "positional"],
)
def test_batches_must_match_the_compiled_keyword_inputs(batch, message):
    generate = Mock()
    with pytest.raises(RegionError, match=message):
        _materialize(
            Keyword(KernelGenerator(validation_inputs=[batch])),
            torch.zeros(4),
            kwargs=_keywords(),
            generate=generate,
        )
    generate.assert_not_called()


def test_a_dict_batch_holds_keyword_inputs_only():
    class Keywords(torch.nn.Module):
        def __init__(self, generator):
            super().__init__()
            self.generator = generator

        def forward(self, *, x):
            with fused_kernel(generator=self.generator, name="keywords"):
                return torch.relu(x + 1)

    batches = [{"x": torch.full((4,), 2.0)}]
    generate = _materialize(
        Keywords(KernelGenerator(validation_inputs=batches)),
        kwargs={"x": torch.zeros(4)},
    )
    ((value,),) = generate.call_args.kwargs["samples"]
    assert torch.equal(value, batches[0]["x"])


def test_parameter_operands_are_copied_once_for_all_samples(tmp_path):
    model = Scaled(
        KernelGenerator(validation_inputs=[torch.full((4,), 5.0 + i) for i in range(3)])
    )
    samples = _materialize(model, torch.zeros(4)).call_args.kwargs["samples"]
    index = next(
        index
        for index, value in enumerate(samples[0])
        if torch.equal(value, model.scale.detach())
    )
    assert all(sample[index] is samples[0][index] for sample in samples)
    # torch.save then stores the parameter once.
    _samples.save_samples(samples, tmp_path / "samples.pt")
    loaded = torch.load(tmp_path / "samples.pt", weights_only=True)
    assert len({sample[index].untyped_storage().data_ptr() for sample in loaded}) == 1


# Stacked keeps two float32 [4] region inputs, 32 bytes, from every batch.
@pytest.mark.parametrize("limit, kept", [(3 * 32, 3), (1, 1)])
def test_validation_batches_stop_at_the_byte_limit_but_keep_the_first(
    monkeypatch, caplog, limit, kept
):
    monkeypatch.setattr(_samples, "MAX_SAMPLE_BYTES", limit)
    batches = Batches([torch.full((4,), float(index)) for index in range(8)])
    with caplog.at_level(logging.WARNING, logger=_samples.__name__):
        generate = _materialize(
            Stacked(KernelGenerator(validation_inputs=batches)), torch.zeros(4)
        )
    assert batches.drawn == kept
    assert [len(call.kwargs["samples"]) for call in generate.call_args_list] == [
        kept
    ] * 2
    assert ("above the" in caplog.text) == (limit < 32)


@requires_cuda
def test_capture_returns_cached_activation_memory_before_generation():
    class Wide(torch.nn.Module):
        def __init__(self, generator):
            super().__init__()
            self.generator = generator

        def forward(self, x):
            x = x.exp().sin().cos()
            with fused_kernel(generator=self.generator, name="wide"):
                return torch.relu(x + 1)

    size = 1 << 23  # 32 MiB of float32 per activation
    batches = [torch.randn(size, device="cuda")]
    model = Wide(KernelGenerator(validation_inputs=batches))
    reserved = []

    def generate(*args, **kwargs):
        reserved.append(torch.cuda.memory_reserved())
        return "registration"

    inputs = torch.zeros(size, device="cuda")
    torch.cuda.empty_cache()
    before = torch.cuda.memory_reserved()
    _materialize(model, inputs, generate=generate)
    # The forward pass cached several activations of 32 MiB each.
    assert reserved[0] < before + (16 << 20)


def _samples_of(*values):
    return [(torch.full((7,), value),) for value in values]


def _generate_with(factory, samples, memo=None, **changes):
    return _factory.generate_registration(
        factory.graph,
        replace(factory.config, **changes),
        region_id="activation",
        samples=samples,
        memo=memo,
    )


def _checked(factory, failing=lambda name, values: False):
    """Record the kernel and first sample values of every holdout check."""
    checks = []

    def finalize(request, directory, timeout):
        path = request["samples_path"]
        values = (
            None
            if path is None
            else [float(sample[0][0]) for sample in _samples.load_samples(path)]
        )
        name = request["candidate"]["kernel_name"]
        checks.append((name, values))
        if failing(name, values):
            return {"status": "error", "stage": "correctness", "error": "mismatch"}
        return _passed(0.2)

    factory.finalize.side_effect = finalize
    return checks


def test_samples_are_saved_as_data_and_checked_by_every_evaluation(factory):
    factory.author.return_value = _candidate("sampled")
    factory.evaluate.return_value = _passed(0.2)
    samples = _samples_of(1.0, float("-inf"))
    assert _generate_with(factory, samples) == _candidate("sampled")
    run = next(factory.root.glob("*/run-*"))
    task = json.loads((run / "task.json").read_text())
    assert task["samples"] == {
        "count": 2,
        "sha256": _samples.samples_digest(samples),
    }
    saved = torch.load(run / "samples.pt", weights_only=True)
    assert len(saved) == 2 and torch.equal(saved[1][0], samples[1][0])
    requests = [
        call.args[0]
        for mock in (factory.evaluate, factory.finalize)
        for call in mock.call_args_list
    ]
    assert len(requests) == 2
    assert all(
        request["samples_path"] == str(run / "samples.pt") for request in requests
    )


def test_cached_kernels_are_reused_after_a_check_on_new_samples(factory):
    factory.author.return_value = _candidate("a")
    factory.evaluate.return_value = _passed(0.2)
    checks = _checked(factory)
    assert _generate_with(factory, _samples_of(1.0, -2.0)) == _candidate("a")
    # New weights, nondeterministic upstream ops, or other batches change the
    # samples but not the task, so each reuse checks the kernel on its own.
    for samples in (_samples_of(1.0, -2.0), _samples_of(5.0, -float("inf")), None):
        assert _generate_with(factory, samples) == _candidate("a")
    factory.author.assert_called_once()
    assert checks == [
        ("a", [1.0, -2.0]),
        ("a", [1.0, -2.0]),
        ("a", [5.0, -float("inf")]),
        ("a", None),
    ]
    assert len(list(factory.root.glob("*/accepted.json"))) == 1


def test_cache_only_builds_reuse_a_kernel_that_passes_their_samples(factory):
    factory.author.return_value = _candidate("a")
    factory.evaluate.return_value = _passed(0.2)
    checks = _checked(factory, failing=lambda name, values: values == [9.0])
    _generate_with(factory, _samples_of(1.0))
    assert _generate_with(
        factory, _samples_of(2.0), allow_generation=False
    ) == _candidate("a")
    with pytest.raises(RegionError, match="failed revalidation"):
        _generate_with(factory, _samples_of(9.0), allow_generation=False)
    factory.author.assert_called_once()
    assert checks == [("a", [1.0]), ("a", [2.0]), ("a", [9.0])]
    # Runs pile up with every compilation, so one that reused a kernel drops
    # its samples after the check; the search and the refused run keep theirs.
    runs = list(factory.root.glob("*/run-*"))
    assert len(runs) == 3
    for run in runs:
        check = run / "cache_check.json"
        reused = check.is_file() and json.loads(check.read_text())["status"] == "passed"
        assert (run / "samples.pt").is_file() is not reused


@pytest.mark.parametrize("use_cache", [True, False])
def test_identical_regions_share_one_search_and_check_their_own_samples(
    factory, use_cache
):
    factory.author.return_value = _candidate("shared")
    factory.evaluate.return_value = _passed(0.2)
    checks = _checked(factory)
    memo = {}
    for samples in (_samples_of(1.0), _samples_of(2.0), _samples_of(1.0), None):
        assert _generate_with(
            factory, samples, memo, use_cache=use_cache
        ) == _candidate("shared")
    factory.author.assert_called_once()
    factory.evaluate.assert_called_once()
    # Only new samples need a check: the third region has the first's data.
    assert checks == [("shared", [1.0]), ("shared", [2.0])]
    # Every region loads the copy of the kernel in the run that accepted it.
    directories = [call.args[-1] for call in factory.register.call_args_list]
    assert len(directories) == 4 and len(set(directories)) == 1


def test_a_region_failing_the_shared_kernel_searches_on_every_identical_regions_samples(
    factory,
):
    factory.author.side_effect = [_candidate("a"), _candidate("b")]
    factory.evaluate.return_value = _passed(0.2)
    checks = _checked(
        factory, failing=lambda name, values: name == "a" and 2.0 in values
    )
    memo = {}
    assert _generate_with(factory, _samples_of(1.0), memo) == _candidate("a")
    assert _generate_with(factory, _samples_of(2.0), memo) == _candidate("b")
    assert _generate_with(factory, _samples_of(3.0), memo) == _candidate("b")
    # "b" was searched and validated on both regions' samples, so it can
    # replace "a" in the cache without failing the first region next time.
    assert checks == [("a", [1.0]), ("a", [2.0]), ("b", [1.0, 2.0]), ("b", [3.0])]
    searched = factory.evaluate.call_args.args[0]
    assert [
        float(sample[0][0])
        for sample in _samples.load_samples(searched["samples_path"])
    ] == [1.0, 2.0]
    task = json.loads((Path(searched["samples_path"]).parent / "task.json").read_text())
    assert task["samples"] == {
        "count": 2,
        "sha256": _samples.samples_digest(_samples_of(1.0, 2.0)),
    }
    memo = {}
    for value in (1.0, 2.0):
        assert _generate_with(factory, _samples_of(value), memo) == _candidate("b")
    assert factory.author.call_count == 2


@pytest.mark.parametrize(
    "samples",
    [
        [(torch.ones(8),)],
        [(torch.ones(7, dtype=torch.float16),)],
        [(torch.ones(7), torch.ones(7))],
    ],
    ids=["shape", "dtype", "arity"],
)
def test_samples_outside_the_region_contract_fail_before_authoring(factory, samples):
    with pytest.raises(RegionError, match="do not fit region 'activation'"):
        _generate_with(factory, samples)
    factory.author.assert_not_called()
    assert not factory.root.exists()


@pytest.fixture
def scripted(monkeypatch):
    """Generate through materialize_regions with a scripted author and evaluator."""
    checks = []

    def evaluate(request, directory, timeout):
        path = request["samples_path"]
        samples = [] if path is None else _samples.load_samples(path)
        checks.append((request["validation_phase"], _samples.samples_digest(samples)))
        return _passed(0.2)

    author = Mock(side_effect=lambda **kwargs: _proposal(_candidate("generated")))
    monkeypatch.setattr(_factory._providers, "propose", author)
    monkeypatch.setattr(_factory, "_evaluate", evaluate)
    monkeypatch.setattr(
        _factory, "make_registration", lambda candidate, *args: candidate
    )
    monkeypatch.setattr(
        _factory, "_target", lambda *args: {"name": "test-gpu", "capability": [9, 0]}
    )
    monkeypatch.setattr(
        _factory, "gpu_lock", lambda root, device: FileLock(str(root / "gpu.lock"))
    )
    return SimpleNamespace(author=author, checks=checks)


def _timed(latency, baseline):
    return {
        **_passed(latency),
        "baseline_latency_ms": baseline,
        "benchmark": {"baseline": {"mad_ms": 0.0}, "candidate": {"mad_ms": 0.0}},
    }


@pytest.mark.parametrize("allow", [True, False], ids=["generating", "cache_only"])
def test_identical_regions_recheck_only_the_shared_kernels_correctness(factory, allow):
    # The shared kernel met the speedup objective in this compilation. Timing
    # noise in an identical region's check must neither refuse a cache-only
    # build nor start a search that replaces the cached kernel.
    factory.author.side_effect = [_candidate("a"), _candidate("b")]
    factory.evaluate.return_value = {**_timed(1.0, 1.3), "execution": "launch"}
    objectives = []

    def finalize(request, directory, timeout):
        objectives.append(request["objective"])
        values = _samples.load_samples(request["samples_path"])
        # Region 2's check meets a busy GPU.
        return _timed(1.0, 1.05 if [float(v[0][0]) for v in values] == [2.0] else 1.3)

    factory.finalize.side_effect = finalize
    objective = dict(objective="eager", min_speedup=1.1)
    assert _generate_with(factory, _samples_of(1.0), {}, **objective) == _candidate("a")
    memo = {}
    for value in (1.0, 2.0):
        registration = _generate_with(
            factory, _samples_of(value), memo, allow_generation=allow, **objective
        )
        assert registration == _candidate("a")
    factory.author.assert_called_once()
    # Region 1's cache check remeasures; region 2's check is correctness only.
    assert objectives[-2:] == ["eager", "candidate"]
    accepted = json.loads(next(factory.root.glob("*/accepted.json")).read_text())
    assert accepted["candidate"] == _candidate("a")


@pytest.mark.parametrize("cause", ["timeout", "out_of_memory"])
def test_an_inconclusive_identical_region_check_is_retried_before_a_search(
    factory, cause
):
    factory.author.side_effect = [_candidate("a"), _candidate("b")]
    factory.evaluate.return_value = _passed(0.2)
    failure = {"status": "error", "stage": "timeout", "error": "Evaluator exceeded"}
    if cause == "out_of_memory":
        # The report of a worker whose correctness check found the GPU full.
        def allocate():
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")

        with pytest.raises(_worker._EvaluationError) as error:
            _worker._run_stage("correctness", allocate)
        failure = _worker._error_report(error.value, factory.root, "holdout")
    outcomes = iter([_passed(0.2), failure, _passed(0.2)])
    factory.finalize.side_effect = lambda *args: next(outcomes)
    memo = {}
    _generate_with(factory, _samples_of(1.0), memo)
    assert _generate_with(factory, _samples_of(2.0), memo) == _candidate("a")
    factory.author.assert_called_once()


def test_a_kernel_that_times_out_on_an_identical_regions_samples_is_replaced(
    factory,
):
    # A timeout can come from the kernel itself, here one that hangs on the
    # second region's samples: after the retry, the search must still run.
    factory.author.side_effect = [_candidate("a"), _candidate("b")]
    factory.evaluate.return_value = _passed(0.2)

    def finalize(request, directory, timeout):
        samples = _samples.load_samples(request["samples_path"])
        values = [float(sample[0][0]) for sample in samples]
        if request["candidate"]["kernel_name"] == "a" and 2.0 in values:
            return {"status": "error", "stage": "timeout", "error": "Exceeded"}
        return _passed(0.2)

    factory.finalize.side_effect = finalize
    memo = {}
    assert _generate_with(factory, _samples_of(1.0), memo) == _candidate("a")
    assert _generate_with(factory, _samples_of(2.0), memo) == _candidate("b")
    assert factory.author.call_count == 2


def _digests(generate):
    return [
        _samples.samples_digest(call.kwargs["samples"])
        for call in generate.call_args_list
    ]


@requires_cuda
def test_identical_regions_with_validation_inputs_make_one_author_call(
    scripted, tmp_path
):
    batches = [torch.randn(16, device="cuda") for _ in range(2)]
    generator = KernelGenerator(
        cache_dir=tmp_path, max_attempts=1, validation_inputs=batches
    )
    generate = Mock(wraps=_GENERATE)
    _materialize(
        Twin(generator).cuda(), torch.zeros(16, device="cuda"), generate=generate
    )
    first, second = _digests(generate)
    assert first != second
    scripted.author.assert_called_once()
    assert scripted.checks == [
        ("search", first),
        ("holdout", first),
        ("holdout", second),
    ]
    assert len([path for path in tmp_path.iterdir() if path.is_dir()]) == 1
    # Engine building no longer needs the samples that the regions shared.
    assert generate.call_args.kwargs["memo"] == {}


@requires_cuda
def test_identical_regions_all_register_the_kernel_that_passed_their_samples(
    scripted, tmp_path, monkeypatch
):
    # The second region's samples reject the first kernel, so its search finds
    # one that passes both regions' samples. The first region, materialized
    # before that search, must not keep the kernel this compilation rejected.
    author = Mock(side_effect=[_proposal(_candidate("a")), _proposal(_candidate("b"))])
    monkeypatch.setattr(_factory._providers, "propose", author)
    holdouts = []

    def evaluate(request, directory, timeout):
        name = request["candidate"]["kernel_name"]
        if request["validation_phase"] == "holdout":
            holdouts.append(name)
            if name == "a" and holdouts.count("a") > 1:
                return {"status": "error", "stage": "correctness", "error": "nan"}
        return _passed(0.2)

    monkeypatch.setattr(_factory, "_evaluate", evaluate)
    registered = Mock(return_value=torch.ops.aten.relu.default)
    batches = [torch.randn(16, device="cuda") for _ in range(2)]
    generator = KernelGenerator(
        cache_dir=tmp_path, max_attempts=1, validation_inputs=batches
    )
    _materialize(
        Twin(generator).cuda(),
        torch.zeros(16, device="cuda"),
        generate=Mock(wraps=_GENERATE),
        register=registered,
    )
    assert author.call_count == 2
    kernels = [call.args[1]["kernel_name"] for call in registered.call_args_list]
    assert kernels == ["b", "b"]


@requires_cuda
@pytest.mark.parametrize("model", [Dense, Noisy], ids=["weights", "upstream"])
def test_cache_only_builds_reuse_a_kernel_after_its_check_on_their_samples(
    scripted, tmp_path, model
):
    """A new checkpoint, or nondeterministic values upstream, changes the samples."""
    batches = [torch.randn(4, 8, device="cuda")]
    generate = Mock(wraps=_GENERATE)
    for seed, allow in ((1, True), (2, False)):
        torch.manual_seed(seed)
        generator = KernelGenerator(
            cache_dir=tmp_path,
            max_attempts=1,
            validation_inputs=batches,
            allow_generation=allow,
        )
        _materialize(
            model(generator).cuda(),
            torch.zeros(4, 8, device="cuda"),
            generate=generate,
        )
    generated, checked = _digests(generate)
    assert generated != checked
    scripted.author.assert_called_once()
    assert scripted.checks == [
        ("search", generated),
        ("holdout", generated),
        ("holdout", checked),
    ]


@requires_cuda
def test_identical_regions_keep_kernels_of_the_tasks_they_were_generated_for(
    scripted, tmp_path, monkeypatch
):
    # Instruction files are read again when each search starts, so an edit
    # during a compilation gives the next identical region another task. Each
    # region keeps a kernel checked on its own samples, and registering the
    # final kernels never starts another search.
    author = Mock(side_effect=[_proposal(_candidate("a")), _proposal(_candidate("b"))])
    monkeypatch.setattr(_factory._providers, "propose", author)
    guide = tmp_path / "guide.md"
    guide.write_text("Before.\n")
    batches = [torch.randn(16, device="cuda") for _ in range(2)]
    generator = KernelGenerator(
        cache_dir=tmp_path / "cache",
        max_attempts=1,
        validation_inputs=batches,
        instruction_files=(guide,),
    )

    def generate(*args, **kwargs):
        registration = _GENERATE(*args, **kwargs)
        guide.write_text("After.\n")
        return registration

    registered = Mock(return_value=torch.ops.aten.relu.default)
    _materialize(
        Twin(generator).cuda(),
        torch.zeros(16, device="cuda"),
        generate=Mock(side_effect=generate),
        register=registered,
    )
    assert author.call_count == 2
    kernels = [call.args[1]["kernel_name"] for call in registered.call_args_list]
    assert kernels == ["a", "b"]
