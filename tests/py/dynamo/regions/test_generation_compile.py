# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""How compilation configures and invokes kernel generation, without an agent."""

import logging
import os
import re
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._ConverterRegistry import DYNAMO_CONVERTERS
from torch_tensorrt.dynamo.region_compilation import (
    _capture,
    _plugins,
    audit_region_placement,
    get_region_records,
    materialize_regions,
    normalize_region_scopes,
)
from torch_tensorrt.dynamo.region_compilation._generation import _factory, _providers
from torch_tensorrt.region import (
    KernelGenerator,
    RegionError,
    SearchPolicy,
    fused_kernel,
)
from torch_tensorrt.region._session import RegionCompilationSession

from ._generation_utils import _candidate, _generate, _passed
from ._utils import compile_regions


class TwoRegions(torch.nn.Module):
    """Two identical generated regions that share one generator."""

    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, x):
        with fused_kernel(generator=self.generator, name="first"):
            x = torch.relu(x + 1)
        x = x * 0.5
        with fused_kernel(generator=self.generator, name="second"):
            return torch.relu(x + 1)


def _captured(model, *inputs):
    """Capture regions in an owned session, leaving materialization to the test."""
    session = RegionCompilationSession(strict=True)
    with session:
        program = torch.export.export(model.eval(), inputs, strict=True)
        program = normalize_region_scopes(program)
        captured = program.graph_module
        graph = program.run_decompositions().module()
    return session, graph, captured


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("options", "conflict"),
    [
        ({}, False),
        ({"offload_module_to_cpu": True}, True),
        # Generated regions belong in TensorRT, so this is not a conflict.
        ({"require_full_compilation": True}, False),
    ],
)
def test_dry_run_runs_reference_bodies_without_generating(
    monkeypatch, tmp_path, options, conflict
):
    generate = Mock(side_effect=AssertionError("A dry run invoked generation"))
    monkeypatch.setattr(_factory, "generate_registration", generate)
    monkeypatch.setattr(_providers, "propose", generate)
    warning = Mock(wraps=_capture.logger.warning)
    monkeypatch.setattr(_capture.logger, "warning", warning)
    model = TwoRegions(KernelGenerator(cache_dir=tmp_path / "cache")).cuda().eval()
    x = torch.randn(4, 8, device="cuda")

    compiled = compile_regions(model, [x], dryrun=True, **options)

    generate.assert_not_called()
    assert not (tmp_path / "cache").exists()
    # Each region gets one stand-in warning and no other, except on a host that
    # is not POSIX, where the dry run also warns that generation requires one.
    stand_ins = [
        call.args[1]
        for call in warning.call_args_list
        if "requires a POSIX host" not in str(call.args[1])
    ]
    assert stand_ins == ["first#0", "second#1"]
    records = get_region_records(compiled)
    # Stand-ins that run the reference bodies take the kernels' plugin route.
    assert [(record.route, record.backend) for record in records] == [
        ("plugin", "eager"),
        ("plugin", "eager"),
    ]
    audit_region_placement(compiled, records)
    if not conflict:
        # Partitioning places them where the kernels' plugins would run.
        assert [record.owner for record in records] == ["_run_on_acc_0"] * 2
    assert bool(compiled.meta.get("torch_tensorrt_region_conflicts")) == conflict
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), model(x))
    converters, _ = DYNAMO_CONVERTERS.get_all_converters_with_target(
        records[0].plugin_op
    )
    with pytest.raises(RegionError, match="'first#0' has a dry run's stand-in"):
        converters[0](None, records[0].plugin_op, (x,), {}, "stand_in")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_conflicting_settings_are_rejected_before_generation(monkeypatch, tmp_path):
    generate = Mock(side_effect=AssertionError("Generation preceded the check"))
    monkeypatch.setattr(_factory, "generate_registration", generate)
    model = TwoRegions(KernelGenerator(cache_dir=tmp_path)).cuda().eval()
    with pytest.raises(RegionError, match="offload_module_to_cpu=True"):
        compile_regions(
            model, [torch.randn(4, 8, device="cuda")], offload_module_to_cpu=True
        )
    generate.assert_not_called()


def test_hardware_compatible_engines_reject_generation_before_any_agent_call(
    monkeypatch,
):
    propose = Mock(side_effect=AssertionError("The provider was called"))
    monkeypatch.setattr(_providers, "propose", propose)
    with pytest.raises(RegionError, match="hardware_compatible=True"):
        _factory.generate_registration(
            torch.fx.symbolic_trace(lambda x: x + 1),
            KernelGenerator(),
            region_id="portable#0",
            settings=CompilationSettings(hardware_compatible=True),
        )
    propose.assert_not_called()


@pytest.mark.skipif(os.name != "posix", reason="Kernel generation requires POSIX")
def test_generation_receives_settings_and_the_session_memo(monkeypatch):
    calls = Mock()
    check, generate = calls.check, calls.generate
    generate.return_value = "registration"
    register = Mock(return_value=torch.ops.aten.relu.default)
    monkeypatch.setattr(_factory, "check_generation", check)
    monkeypatch.setattr(_factory, "generate_registration", generate)
    monkeypatch.setattr(_plugins, "register_region_plugin", register)
    generator = KernelGenerator()
    settings = CompilationSettings()
    session, graph, captured = _captured(TwoRegions(generator), torch.ones(7))

    with session:
        materialize_regions(graph, captured, settings)

    # Every region is checked before the first search.
    assert [call[0] for call in calls.mock_calls] == ["check"] * 2 + ["generate"] * 2
    for region_calls in (check.call_args_list, generate.call_args_list):
        assert [call.kwargs["region_id"] for call in region_calls] == [
            "first#0",
            "second#1",
        ]
    assert all(call.kwargs["settings"] is settings for call in check.call_args_list)
    for call in generate.call_args_list:
        assert call.args[1] is generator
        assert call.kwargs["settings"] is settings
        assert call.kwargs["memo"] is session.generated_kernels
    assert [call.args[1] for call in register.call_args_list] == ["registration"] * 2
    assert [record.route for record in get_region_records(graph)] == ["plugin"] * 2


def test_generated_region_requires_its_compilation_session(monkeypatch):
    generate = Mock()
    monkeypatch.setattr(_factory, "generate_registration", generate)
    _, graph, captured = _captured(TwoRegions(KernelGenerator()), torch.ones(7))
    with pytest.raises(RegionError, match="owning compilation session"):
        materialize_regions(graph, captured, CompilationSettings())
    generate.assert_not_called()


def test_generation_rejects_a_non_posix_host(monkeypatch):
    generate = Mock()
    monkeypatch.setattr(_factory, "generate_registration", generate)
    monkeypatch.setattr(_capture, "os", SimpleNamespace(name="nt"))
    session, graph, captured = _captured(TwoRegions(KernelGenerator()), torch.ones(7))
    with session, pytest.raises(RegionError, match="'first#0' requires a POSIX host"):
        materialize_regions(graph, captured, CompilationSettings())
    generate.assert_not_called()


@pytest.mark.skipif(os.name != "posix", reason="Kernel generation requires POSIX")
def test_generator_checks_files_and_builtin_tuning_when_constructed(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    guidance = tmp_path / "guidance.md"
    guidance.write_text("Prefer coalesced loads.")
    generator = KernelGenerator(instruction_files=["~/guidance.md"])
    assert generator.instruction_files == (guidance.resolve(),)
    assert hash(generator) == hash(KernelGenerator(instruction_files=(guidance,)))
    for missing in ("~/missing.md", tmp_path):
        with pytest.raises(ValueError, match="not an existing file"):
            KernelGenerator(instruction_files=(missing,))
    with pytest.raises(ValueError, match="num_warps"):
        KernelGenerator(
            backend="cutile", search=SearchPolicy(tuning={"num_warps": (4,)})
        )
    # Extension backends declare their tuning schema only once imported.
    extension = KernelGenerator(
        backend="missing_package:backend", search=SearchPolicy(tuning={"knob": (1,)})
    )
    assert extension.search.tuning == {"knob": (1,)}


class InvalidSecond(torch.nn.Module):
    """A generated region, then one whose int64 boundary generation rejects."""

    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, x, index):
        with fused_kernel(generator=self.generator, name="first"):
            x = torch.relu(x + 1)
        with fused_kernel(generator=self.generator, name="second"):
            index = index * 2 + 1
        return x, index


def _invalid_second(tmp_path):
    model = InvalidSecond(KernelGenerator(cache_dir=tmp_path / "cache"))
    inputs = (torch.randn(4, 8, device="cuda"), torch.arange(8, device="cuda"))
    return model.cuda().eval(), inputs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(os.name != "posix", reason="Kernel generation requires POSIX")
def test_an_invalid_region_fails_before_any_region_is_searched(monkeypatch, tmp_path):
    search = Mock(side_effect=AssertionError("A search started"))
    monkeypatch.setattr(_factory, "generate_registration", search)
    monkeypatch.setattr(_providers, "propose", search)
    model, inputs = _invalid_second(tmp_path)
    with pytest.raises(
        RegionError,
        match=r"^Kernel generation for 'second#1': .* bfloat16 boundaries$",
    ):
        compile_regions(model, inputs)
    search.assert_not_called()
    assert not (tmp_path / "cache").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(os.name != "posix", reason="Kernel generation requires POSIX")
def test_dry_run_warns_of_regions_that_generation_would_reject(
    monkeypatch, tmp_path, caplog
):
    search = Mock(side_effect=AssertionError("A dry run started a search"))
    monkeypatch.setattr(_factory, "generate_registration", search)
    monkeypatch.setattr(_providers, "propose", search)
    model, inputs = _invalid_second(tmp_path)
    with caplog.at_level(logging.WARNING, logger=_capture.__name__):
        compile_regions(model, inputs, dryrun=True)
    failures = [
        record.getMessage()
        for record in caplog.records
        if "Kernel generation for" in record.getMessage()
    ]
    assert failures == [
        "Region dry-run: Kernel generation for 'second#1': Agent-generated "
        "regions support float16, float32, and bfloat16 boundaries"
    ]
    search.assert_not_called()
    assert not (tmp_path / "cache").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_dry_run_reports_generated_kernels_that_would_run_outside_tensorrt(
    monkeypatch, tmp_path, caplog
):
    generate = Mock(side_effect=AssertionError("A dry run invoked generation"))
    monkeypatch.setattr(_factory, "generate_registration", generate)
    model = TwoRegions(KernelGenerator(cache_dir=tmp_path / "cache")).cuda().eval()
    report = tmp_path / "report.txt"
    with caplog.at_level(logging.WARNING, logger="torch_tensorrt"):
        compile_regions(
            model,
            [torch.randn(4, 8, device="cuda")],
            dryrun=str(report),
            min_block_size=10,
        )
    # Compiling without dryrun raises this error once the kernels exist.
    messages = [record.getMessage() for record in caplog.records]
    for region_id in ("first#0", "second#1"):
        message = f"fused_kernel region {region_id!r} must run in a TensorRT plugin"
        assert sum(message in text for text in messages) == 1
        assert (
            f"Region: {region_id}, fused_kernel region outside TensorRT"
            in report.read_text()
        )
    assert "execute_in_torch" not in report.read_text()
    generate.assert_not_called()


class Transposed(torch.nn.Module):
    """A generated region whose non-contiguous output no plugin can have."""

    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, x):
        with fused_kernel(generator=self.generator, name="transposed"):
            x = torch.relu(x).t()
        return x * 2.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(os.name != "posix", reason="Kernel generation requires POSIX")
def test_dry_run_report_lists_the_errors_that_compiling_raises(monkeypatch, tmp_path):
    search = Mock(side_effect=AssertionError("A dry run started a search"))
    monkeypatch.setattr(_factory, "generate_registration", search)
    monkeypatch.setattr(_providers, "propose", search)
    model, inputs = _invalid_second(tmp_path)
    report = tmp_path / "invalid.txt"
    compile_regions(model, inputs, dryrun=str(report))
    assert (
        "Region: second#1, Kernel generation for 'second#1': Agent-generated "
        "regions support float16, float32, and bfloat16 boundaries"
    ) in report.read_text()
    # The dry run shows a region whose boundaries no plugin can have as its
    # reference body, which is no execute_in_torch region.
    model = Transposed(KernelGenerator(cache_dir=tmp_path / "cache")).cuda().eval()
    report = tmp_path / "transposed.txt"
    compiled = compile_regions(
        model, [torch.randn(16, 32, device="cuda")], dryrun=str(report)
    )
    assert [record.route for record in get_region_records(compiled)] == ["torch"]
    text = report.read_text()
    # The error names the operation that returns the transposed view.
    found = re.search(
        r"Plugin region 'transposed#0' output 0 requires contiguous strided "
        r"tensors, but aten\.\w+\.\w+ returns a non-contiguous tensor; call "
        r"\.contiguous\(\) on the output inside the region",
        text,
    )
    assert found is not None, text
    error = found[0]
    assert f"Region: transposed#0, {error}" in text
    assert (
        "Region: transposed#0, generated fused_kernel region shown as its "
        f"PyTorch reference because no plugin can have its boundaries ({error})"
    ) in text
    assert "execute_in_torch" not in text
    search.assert_not_called()
    assert not (tmp_path / "cache").exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(os.name != "posix", reason="Kernel generation requires POSIX")
def test_a_dry_run_stopped_by_settings_reports_the_region_errors(monkeypatch, tmp_path):
    search = Mock(side_effect=AssertionError("A dry run started a search"))
    monkeypatch.setattr(_factory, "generate_registration", search)
    model, inputs = _invalid_second(tmp_path)
    report = tmp_path / "report.txt"
    compile_regions(model, inputs, dryrun=str(report), offload_module_to_cpu=True)
    text = report.read_text()
    assert "Dry run stopped before partitioning: Regions are incompatible" in text
    assert (
        "Region: second#1, Kernel generation for 'second#1': Agent-generated "
        "regions support float16, float32, and bfloat16 boundaries"
    ) in text
    search.assert_not_called()


def _without_qdp_plugins(monkeypatch):
    # As on TensorRT-RTX and TensorRT 10.14 builds.
    monkeypatch.setattr(
        _factory,
        "ENABLED_FEATURES",
        _factory.ENABLED_FEATURES._replace(qdp_plugin=False),
    )


def test_generation_requires_tensorrt_qdp_plugins(factory, monkeypatch):
    _without_qdp_plugins(monkeypatch)
    message = (
        r"^Kernel generation for 'activation' embeds its kernel in the TensorRT "
        r"engine as a Quick Deployable Plugin \(QDP\), which this TensorRT build "
        r"does not support"
    )
    with pytest.raises(RegionError, match=message):
        _factory.check_generation(factory.graph, factory.config, region_id="activation")
    with pytest.raises(RegionError, match=message):
        _generate(factory)
    factory.author.assert_not_called()
    assert not factory.root.exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(os.name != "posix", reason="Kernel generation requires POSIX")
def test_without_qdp_plugins_compilation_fails_before_any_search(
    monkeypatch, tmp_path, caplog
):
    search = Mock(side_effect=AssertionError("A search started"))
    monkeypatch.setattr(_factory, "generate_registration", search)
    monkeypatch.setattr(_providers, "propose", search)
    _without_qdp_plugins(monkeypatch)
    model = TwoRegions(KernelGenerator(cache_dir=tmp_path / "cache")).cuda().eval()
    inputs = [torch.randn(4, 8, device="cuda")]
    report = tmp_path / "report.txt"
    with caplog.at_level(logging.WARNING, logger=_capture.__name__):
        compile_regions(model, inputs, dryrun=str(report))
    for region_id in ("first#0", "second#1"):
        error = f"Kernel generation for {region_id!r} embeds its kernel"
        assert any(error in record.getMessage() for record in caplog.records)
        assert f"Region: {region_id}, {error}" in report.read_text()
    with pytest.raises(RegionError, match="'first#0' embeds its kernel"):
        compile_regions(model, inputs)
    search.assert_not_called()
    assert not (tmp_path / "cache").exists()


def test_generation_checks_instructions_and_extension_tuning(factory, monkeypatch):
    # KernelGenerator checks only that instruction files exist, and the tuning
    # choices of built-in backends; the check before any search reads both.
    guidance = factory.root.parent / "guidance.md"
    guidance.write_text(" \n")
    config = replace(factory.config, instruction_files=(guidance,))
    with pytest.raises(
        RegionError,
        match=r"^Kernel generation for 'activation' failed: Instruction resource "
        r"'instruction-file-1/guidance.md' is empty$",
    ):
        _factory.check_generation(factory.graph, config, region_id="activation")
    backend = SimpleNamespace(
        name="custom",
        version="1",
        instruction_resources=(),
        proposal_schema={"type": "object", "properties": {"artifact": {}}},
        target_metadata=lambda device: {},
        validate_candidate=lambda candidate, inputs, outputs: dict(candidate),
        make_registration=factory.register,
    )
    monkeypatch.setitem(sys.modules, "test_backend", SimpleNamespace(adapter=backend))
    config = replace(
        factory.config,
        backend="test_backend:adapter",
        search=SearchPolicy(tuning={"knob": (1, 2)}),
    )
    with pytest.raises(
        RegionError,
        match=r"^Kernel generation for 'activation' failed: Backend "
        r"'test_backend:adapter' has no tunable field 'knob'$",
    ):
        _factory.check_generation(factory.graph, config, region_id="activation")
    factory.author.assert_not_called()
    assert not factory.root.exists()


class Guided(torch.nn.Module):
    """A generated region, then one whose generator has blank guidance."""

    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, x):
        with fused_kernel(generator=self.first, name="first"):
            x = torch.relu(x + 1)
        with fused_kernel(generator=self.second, name="second"):
            return torch.sigmoid(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(os.name != "posix", reason="Kernel generation requires POSIX")
def test_a_blank_instruction_file_fails_before_any_region_is_searched(
    monkeypatch, tmp_path, caplog
):
    search = Mock(side_effect=AssertionError("A search started"))
    monkeypatch.setattr(_factory, "generate_registration", search)
    monkeypatch.setattr(_providers, "propose", search)
    guidance = tmp_path / "guidance.md"
    guidance.write_text(" \n")
    model = Guided(
        KernelGenerator(cache_dir=tmp_path / "cache"),
        KernelGenerator(cache_dir=tmp_path / "cache", instruction_files=(guidance,)),
    )
    inputs = [torch.randn(4, 8, device="cuda")]
    error = (
        "Kernel generation for 'second#1' failed: Instruction resource "
        "'instruction-file-1/guidance.md' is empty"
    )
    report = tmp_path / "report.txt"
    with caplog.at_level(logging.WARNING, logger=_capture.__name__):
        compile_regions(model.cuda().eval(), inputs, dryrun=str(report))
    assert f"Region dry-run: {error}" in [r.getMessage() for r in caplog.records]
    assert f"Region: second#1, {error}" in report.read_text()
    with pytest.raises(RegionError, match=f"^{re.escape(error)}$"):
        compile_regions(model.cuda().eval(), inputs)
    search.assert_not_called()
    assert not (tmp_path / "cache").exists()


def test_generation_checks_keep_the_cause_of_a_toolchain_error(factory, monkeypatch):
    failure = RuntimeError("The probe kernel failed MLIR verification")

    def probe(backend, device):
        try:
            raise failure
        except RuntimeError as error:
            raise RegionError(
                "CuTe DSL generation cannot compile and load a probe kernel"
            ) from error

    monkeypatch.setattr(_factory, "_target", probe)
    with pytest.raises(RegionError) as error:
        _factory.check_generation(factory.graph, factory.config, region_id="activation")
    assert str(error.value) == (
        "Kernel generation for 'activation': CuTe DSL generation cannot compile "
        "and load a probe kernel"
    )
    # As when the search raises it, the compiler's exception is the cause.
    assert error.value.__cause__ is failure
    assert failure.__traceback__ is not None


def test_generation_checks_name_the_region_once(factory, monkeypatch):
    class WithConstant(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("bias", torch.ones(7))

        def forward(self, x):
            return x + self.bias

    def check(graph=factory.graph, config=factory.config):
        _factory.check_generation(graph, config, region_id="activation")

    with pytest.raises(
        RegionError, match=r"^Kernel generation for 'activation': .*tensor constants"
    ):
        check(torch.fx.symbolic_trace(WithConstant()))
    toolchain = RegionError("Kernel generation backend requires cuda-tile")
    monkeypatch.setattr(_factory, "_target", Mock(side_effect=toolchain))
    with pytest.raises(RegionError) as error:
        check()
    assert str(error.value) == f"Kernel generation for 'activation': {toolchain}"
    failure = RuntimeError("cuTile toolchain is unavailable")
    monkeypatch.setattr(_factory, "_target", Mock(side_effect=failure))
    with pytest.raises(RegionError, match="'activation' failed: cuTile") as error:
        check()
    assert error.value.__cause__ is failure
    # An error that already names the region keeps its message.
    monkeypatch.setenv("TORCH_TENSORRT_REGION_GENERATION", "0")
    with pytest.raises(RegionError, match="use_cache=False") as error:
        check(config=replace(factory.config, use_cache=False))
    assert str(error.value).count("'activation'") == 1
    factory.author.assert_not_called()
    assert not factory.root.exists()


@pytest.mark.skipif(os.name != "posix", reason="Kernel generation requires POSIX")
def test_a_search_that_accepts_no_kernel_says_how_to_proceed(factory):
    advice = (
        "Compilation has no fallback for the region: compile again with "
        "SearchPolicy(resume=True) to continue this search, or with a larger "
        "budget (max_attempts, timeout_s)"
    )
    escapes = (
        "supply a kernel through fused_kernel(kernel=...); or remove the annotation"
    )
    factory.author.return_value = _candidate("wrong")
    factory.evaluate.return_value = {
        "status": "error",
        "stage": "correctness",
        "error": "Numerical mismatch",
    }
    with pytest.raises(RegionError) as error:
        _generate(factory)
    assert re.search(
        r"Numerical mismatch\. Artifacts: \S+\. Compilation", str(error.value)
    )
    assert str(error.value).endswith(f"{advice}; {escapes}")
    # A finalist can also fail its final validation.
    factory.evaluate.return_value = _passed(0.2)
    factory.finalize.return_value = {
        "status": "error",
        "stage": "correctness",
        "error": "Fresh input mismatch",
    }
    with pytest.raises(RegionError, match="final validation: Fresh") as error:
        _generate(factory)
    assert str(error.value).endswith(f"{advice}; {escapes}")
    # Correct kernels that are too slow can meet a lower bar.
    with pytest.raises(RegionError, match="met the performance objective") as error:
        _generate(factory, objective="eager", min_speedup=1.2)
    assert str(error.value).endswith(
        f"{advice}, a lower min_speedup or objective='candidate'; {escapes}"
    )
