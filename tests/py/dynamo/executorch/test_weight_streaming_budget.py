"""CPU-only tests for the ExecuTorch weight streaming budget option.

These exercise the export-time plumbing: budget validation and the compile spec
carried into the delegate. The automatic default is applied by the C++ delegate
at load time, so it is not covered here.
"""

import importlib
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("executorch.exir")

import torch  # noqa: E402
from executorch.exir.backend.compile_spec_schema import CompileSpec  # noqa: E402
from torch_tensorrt._compile import save  # noqa: E402
from torch_tensorrt.executorch.partitioner import (  # noqa: E402
    normalize_weight_streaming_budget_per_engine,
    WEIGHT_STREAMING_BUDGET_COMPILE_SPEC_KEY,
)

_KEY = WEIGHT_STREAMING_BUDGET_COMPILE_SPEC_KEY


class FakeExportedProgram:
    """A program stand-in carrying the graph module and signature ``export()`` reads.

    ``export()`` runs the mutation-declaration pass over every program it is handed, and
    that pass reads both members. Nothing here is mutated, so the pass finds nothing to
    declare and hands back the same object.
    """

    def __init__(self):
        graph = torch.fx.Graph()
        graph.output((graph.placeholder("x"),))
        self.graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)
        self.graph_signature = SimpleNamespace(inputs_to_buffers={}, output_specs=[])


class FakeTensorRTPartitioner:
    def __init__(self, compile_specs):
        self.compile_specs = compile_specs


def _budget_spec(specs):
    for spec in specs:
        if spec.key == _KEY:
            return spec
    return None


def _patch_lowering(monkeypatch, engine_counts=None):
    """Stub out everything export() needs except the compile-spec resolution itself.

    Engines and ExecuTorch lowering are irrelevant to where the budget spec lands, and
    stubbing them keeps these tests CPU-only.
    """
    import executorch.exir
    import torch_tensorrt._features as features
    import torch_tensorrt.executorch as executorch_api
    import torch_tensorrt.executorch._export_utils as export_utils

    monkeypatch.setattr(
        features,
        "ENABLED_FEATURES",
        features.ENABLED_FEATURES._replace(torch_tensorrt_runtime=True),
    )
    export_module = importlib.import_module("torch_tensorrt.executorch._export")
    engine_counts = engine_counts or {}
    lower = MagicMock(return_value=object())
    monkeypatch.setattr(executorch.exir, "to_edge_transform_and_lower", lower)
    monkeypatch.setattr(executorch_api, "TensorRTPartitioner", FakeTensorRTPartitioner)
    monkeypatch.setattr(executorch_api, "get_edge_compile_config", lambda: "default")
    monkeypatch.setattr(export_module, "ExportedProgram", FakeExportedProgram)
    monkeypatch.setattr(
        export_utils,
        "validate_engine_program",
        lambda program, resolved=None: engine_counts.get(program, 1),
    )
    monkeypatch.setattr(export_utils, "stage_exported_program", lambda program: program)
    monkeypatch.setattr(
        export_utils,
        "replace_execute_engine",
        lambda program, resolved=None: program,
    )
    return export_module, lower


def _specs_by_method(lower):
    """The TensorRT partitioner's compile specs, per method, from the lowering call."""
    partitioner = lower.call_args.kwargs["partitioner"]
    if not isinstance(partitioner, dict):
        partitioner = {"forward": partitioner}
    return {name: chain[0].compile_specs for name, chain in partitioner.items()}


# ---------------------------------------------------------------------------
# normalize_weight_streaming_budget_per_engine
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        (0, b"0"),
        (8589934592, b"8589934592"),
    ],
)
def test_normalize_valid(value, expected):
    assert normalize_weight_streaming_budget_per_engine(value) == expected


@pytest.mark.unit
@pytest.mark.parametrize("value", [-1, -(2**63), 2**63, 2**63 + 5])
def test_normalize_out_of_range_raises(value):
    with pytest.raises(ValueError):
        normalize_weight_streaming_budget_per_engine(value)


@pytest.mark.unit
@pytest.mark.parametrize("value", ["auto", "disabled", "1024"])
def test_normalize_string_raises(value):
    # Strings are not accepted; the budget is a non-negative int (or None).
    with pytest.raises(TypeError):
        normalize_weight_streaming_budget_per_engine(value)


@pytest.mark.unit
@pytest.mark.parametrize("value", [True, False])
def test_normalize_bool_raises(value):
    with pytest.raises(TypeError):
        normalize_weight_streaming_budget_per_engine(value)


@pytest.mark.unit
def test_normalize_float_raises():
    with pytest.raises(TypeError):
        normalize_weight_streaming_budget_per_engine(1.5)


# ---------------------------------------------------------------------------
# The budget reaching the TensorRT partitioner through export()
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("budget,expected", [(0, b"0"), (8589934592, b"8589934592")])
def test_kwarg_injects_compile_spec(monkeypatch, budget, expected):
    export_module, lower = _patch_lowering(monkeypatch)
    export_module.export(
        FakeExportedProgram(), weight_streaming_budget_per_engine=budget
    )
    spec = _budget_spec(_specs_by_method(lower)["forward"])
    assert spec is not None
    assert spec.value == expected


@pytest.mark.unit
def test_kwarg_injects_compile_spec_into_every_method(monkeypatch):
    export_module, lower = _patch_lowering(monkeypatch)
    export_module.export(
        {"forward": FakeExportedProgram(), "decode": FakeExportedProgram()},
        weight_streaming_budget_per_engine=4096,
    )
    per_method = _specs_by_method(lower)
    assert set(per_method) == {"forward", "decode"}
    for name, specs in per_method.items():
        spec = _budget_spec(specs)
        assert spec is not None, name
        assert spec.value == b"4096"


@pytest.mark.unit
def test_no_spec_injected_without_budget(monkeypatch):
    # No budget: nothing is injected. The delegate applies the automatic budget
    # itself for streaming-built engines.
    export_module, lower = _patch_lowering(monkeypatch)
    export_module.export(FakeExportedProgram())
    assert _budget_spec(_specs_by_method(lower)["forward"]) is None


@pytest.mark.unit
def test_caller_compile_specs_passed_through(monkeypatch):
    # Non-budget caller compile_specs are forwarded alongside the budget.
    export_module, lower = _patch_lowering(monkeypatch)
    caller = CompileSpec("target_device", b"cuda:1")
    export_module.export(
        FakeExportedProgram(),
        compile_specs=[caller],
        weight_streaming_budget_per_engine=4096,
    )
    specs = _specs_by_method(lower)["forward"]
    assert caller in specs
    assert _budget_spec(specs).value == b"4096"


@pytest.mark.unit
@pytest.mark.parametrize("budget", [None, 4096])
def test_caller_budget_spec_in_compile_specs_raises(monkeypatch, budget):
    # The budget must come from the kwarg, not a manually-pinned compile spec.
    export_module, _ = _patch_lowering(monkeypatch)
    with pytest.raises(ValueError, match=_KEY):
        export_module.export(
            FakeExportedProgram(),
            compile_specs=[CompileSpec(_KEY, b"4096")],
            weight_streaming_budget_per_engine=budget,
        )


@pytest.mark.unit
def test_invalid_budget_raises_from_export(monkeypatch):
    export_module, _ = _patch_lowering(monkeypatch)
    with pytest.raises(TypeError):
        export_module.export(
            FakeExportedProgram(), weight_streaming_budget_per_engine="auto"
        )


# ---------------------------------------------------------------------------
# Multi-engine warning
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_multi_engine_explicit_warns(monkeypatch, caplog):
    program = FakeExportedProgram()
    export_module, _ = _patch_lowering(monkeypatch, engine_counts={program: 2})
    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.executorch._export"):
        export_module.export(program, weight_streaming_budget_per_engine=4096)
    assert "weight_streaming_budget_per_engine applies to each" in caplog.text


@pytest.mark.unit
def test_multi_engine_none_does_not_warn(monkeypatch, caplog):
    program = FakeExportedProgram()
    export_module, lower = _patch_lowering(monkeypatch, engine_counts={program: 2})
    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.executorch._export"):
        export_module.export(program)
    assert _budget_spec(_specs_by_method(lower)["forward"]) is None
    assert "weight_streaming_budget_per_engine applies to each" not in caplog.text


# ---------------------------------------------------------------------------
# save() entry-point guards
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_save_rejects_bool_budget(tmp_path):
    with pytest.raises(TypeError):
        save(
            torch.nn.Linear(1, 1),
            str(tmp_path / "model.pte"),
            output_format="executorch",
            weight_streaming_budget_per_engine=True,
        )


@pytest.mark.unit
def test_save_rejects_string_budget(tmp_path):
    with pytest.raises(TypeError):
        save(
            torch.nn.Linear(1, 1),
            str(tmp_path / "model.pte"),
            output_format="executorch",
            weight_streaming_budget_per_engine="auto",
        )


@pytest.mark.unit
def test_save_rejects_negative_budget(tmp_path):
    with pytest.raises(ValueError):
        save(
            torch.nn.Linear(1, 1),
            str(tmp_path / "model.pte"),
            output_format="executorch",
            weight_streaming_budget_per_engine=-1,
        )


@pytest.mark.unit
def test_save_rejects_unknown_executorch_kwarg(tmp_path):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        save(
            torch.nn.Linear(1, 1),
            str(tmp_path / "model.pte"),
            output_format="executorch",
            weight_streaming_budget_per_enginet=4096,
        )


@pytest.mark.unit
def test_save_warns_when_budget_used_with_non_executorch_format(tmp_path, caplog):
    """The docstring says the executorch-only kwargs are ignored with a warning for
    other formats. compile_specs and backend_config warn; the budget must too.

    The save is expected to fail afterwards (a plain nn.Module is not an
    ExportedProgram), so the warning is asserted independently of the outcome.
    """
    with caplog.at_level(logging.WARNING):
        try:
            save(
                torch.nn.Linear(1, 1),
                str(tmp_path / "model.ep"),
                output_format="exported_program",
                weight_streaming_budget_per_engine=4096,
            )
        except Exception:
            pass
    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "weight_streaming_budget_per_engine=" in m and "will be ignored" in m
        for m in messages
    ), messages


@pytest.mark.unit
def test_save_forwards_the_budget_to_export(monkeypatch, tmp_path):
    """save() must actually pass the budget on; a dropped kwarg is silent otherwise."""
    import torch_tensorrt._compile as compile_module

    seen = {}

    def fake_export(program, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after the forward")

    monkeypatch.setattr("torch_tensorrt.executorch.export", fake_export)
    # _compile.py binds ENABLED_FEATURES at import, so patch the name it holds.
    monkeypatch.setattr(
        compile_module,
        "ENABLED_FEATURES",
        compile_module.ENABLED_FEATURES._replace(torch_tensorrt_runtime=True),
    )
    with pytest.raises(RuntimeError, match="stop after the forward"):
        compile_module._save_as_executorch(
            object(),
            str(tmp_path / "model.pte"),
            weight_streaming_budget_per_engine=4096,
        )
    assert seen["weight_streaming_budget_per_engine"] == 4096
