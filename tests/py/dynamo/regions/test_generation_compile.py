# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""How compilation configures and invokes kernel generation, without an agent."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from torch_tensorrt.dynamo._settings import CompilationSettings
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
    assert [call.args[1] for call in warning.call_args_list] == ["first#0", "second#1"]
    records = get_region_records(compiled)
    assert [(record.route, record.backend) for record in records] == [
        ("torch", "eager"),
        ("torch", "eager"),
    ]
    audit_region_placement(compiled, records)
    assert bool(compiled.meta.get("torch_tensorrt_region_conflicts")) == conflict
    with torch.no_grad():
        torch.testing.assert_close(compiled(x), model(x))


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
    generate = Mock(return_value="registration")
    register = Mock(return_value=torch.ops.aten.relu.default)
    monkeypatch.setattr(_factory, "generate_registration", generate)
    monkeypatch.setattr(_plugins, "register_region_plugin", register)
    generator = KernelGenerator()
    settings = CompilationSettings()
    session, graph, captured = _captured(TwoRegions(generator), torch.ones(7))

    with session:
        materialize_regions(graph, captured, settings)

    assert [call.kwargs["region_id"] for call in generate.call_args_list] == [
        "first#0",
        "second#1",
    ]
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
