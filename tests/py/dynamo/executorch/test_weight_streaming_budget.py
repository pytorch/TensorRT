"""CPU-only tests for the ExecuTorch weight streaming budget option.

These exercise the export-time plumbing: budget validation and the compile spec
carried into the delegate. The automatic default is applied by the C++ delegate
at load time, so it is not covered here.
"""

import logging
from types import SimpleNamespace

import pytest

executorch = pytest.importorskip("executorch.exir")

import torch  # noqa: E402
from torch_tensorrt._compile import (  # noqa: E402
    _normalize_weight_streaming_budget,
    _resolve_executorch_compile_specs,
    save,
)
from torch_tensorrt.executorch.partitioner import (  # noqa: E402
    WEIGHT_STREAMING_BUDGET_COMPILE_SPEC_KEY,
)

_KEY = WEIGHT_STREAMING_BUDGET_COMPILE_SPEC_KEY


def _budget_spec(specs):
    for spec in specs:
        if spec.key == _KEY:
            return spec
    return None


@pytest.fixture
def patch_engine_count(monkeypatch):
    """Patch the engine-node count so the resolver runs without a real program."""

    def _apply(count=1):
        monkeypatch.setattr(
            "torch_tensorrt._compile._count_executorch_engine_nodes",
            lambda exp_program: count,
        )

    return _apply


# ---------------------------------------------------------------------------
# _normalize_weight_streaming_budget
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
    assert _normalize_weight_streaming_budget(value) == expected


@pytest.mark.unit
@pytest.mark.parametrize("value", [-1, -(2**63), 2**63, 2**63 + 5])
def test_normalize_out_of_range_raises(value):
    with pytest.raises(ValueError):
        _normalize_weight_streaming_budget(value)


@pytest.mark.unit
@pytest.mark.parametrize("value", ["auto", "disabled", "1024"])
def test_normalize_string_raises(value):
    # Strings are not accepted; the budget is a non-negative int (or None).
    with pytest.raises(TypeError):
        _normalize_weight_streaming_budget(value)


@pytest.mark.unit
@pytest.mark.parametrize("value", [True, False])
def test_normalize_bool_raises(value):
    with pytest.raises(TypeError):
        _normalize_weight_streaming_budget(value)


@pytest.mark.unit
def test_normalize_float_raises():
    with pytest.raises(TypeError):
        _normalize_weight_streaming_budget(1.5)


# ---------------------------------------------------------------------------
# _resolve_executorch_compile_specs
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize("budget,expected", [(0, b"0"), (8589934592, b"8589934592")])
def test_kwarg_injects_compile_spec(patch_engine_count, budget, expected):
    patch_engine_count(1)
    specs = _resolve_executorch_compile_specs(SimpleNamespace(), [], budget)
    spec = _budget_spec(specs)
    assert spec is not None
    assert spec.value == expected


@pytest.mark.unit
def test_kwarg_spec_lands_on_delegation_spec(patch_engine_count):
    from torch_tensorrt.executorch.partitioner import TensorRTPartitioner

    patch_engine_count(1)
    specs = _resolve_executorch_compile_specs(SimpleNamespace(), [], 8589934592)
    partitioner = TensorRTPartitioner(compile_specs=specs)
    spec = _budget_spec(partitioner.delegation_spec.compile_specs)
    assert spec is not None
    assert spec.value == b"8589934592"


@pytest.mark.unit
def test_no_spec_injected_without_budget():
    # No budget: nothing is injected. The delegate applies the automatic budget
    # itself for streaming-built engines.
    specs = _resolve_executorch_compile_specs(SimpleNamespace(), [], None)
    assert _budget_spec(specs) is None


@pytest.mark.unit
def test_caller_compile_specs_passed_through():
    # Non-budget caller compile_specs are forwarded unchanged.
    sentinel = SimpleNamespace(key="target_device", value=b"cuda:1")
    specs = _resolve_executorch_compile_specs(SimpleNamespace(), [sentinel], None)
    assert sentinel in specs
    assert _budget_spec(specs) is None


@pytest.mark.unit
def test_caller_budget_spec_in_compile_specs_raises():
    # The budget must come from the kwarg, not a manually-pinned compile spec.
    spec = SimpleNamespace(key=_KEY, value=b"4096")
    with pytest.raises(ValueError):
        _resolve_executorch_compile_specs(SimpleNamespace(), [spec], None)


# ---------------------------------------------------------------------------
# Multi-engine warning
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_multi_engine_explicit_warns(patch_engine_count, caplog):
    patch_engine_count(2)
    with caplog.at_level(logging.WARNING, logger="torch_tensorrt._compile"):
        _resolve_executorch_compile_specs(SimpleNamespace(), [], 4096)
    assert "multiple TensorRT engines" in caplog.text


@pytest.mark.unit
def test_multi_engine_none_does_not_warn(patch_engine_count, caplog):
    patch_engine_count(2)
    with caplog.at_level(logging.WARNING, logger="torch_tensorrt._compile"):
        specs = _resolve_executorch_compile_specs(SimpleNamespace(), [], None)
    assert _budget_spec(specs) is None
    assert "multiple TensorRT engines" not in caplog.text


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
            weight_streaming_budget=True,
        )


@pytest.mark.unit
def test_save_rejects_string_budget(tmp_path):
    with pytest.raises(TypeError):
        save(
            torch.nn.Linear(1, 1),
            str(tmp_path / "model.pte"),
            output_format="executorch",
            weight_streaming_budget="auto",
        )


@pytest.mark.unit
def test_save_rejects_negative_budget(tmp_path):
    with pytest.raises(ValueError):
        save(
            torch.nn.Linear(1, 1),
            str(tmp_path / "model.pte"),
            output_format="executorch",
            weight_streaming_budget=-1,
        )


@pytest.mark.unit
def test_save_rejects_unknown_executorch_kwarg(tmp_path):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        save(
            torch.nn.Linear(1, 1),
            str(tmp_path / "model.pte"),
            output_format="executorch",
            weight_streaming_budgett=4096,
        )
