# type: ignore
"""Tests for ``user_symbol_bounds`` plumbed from ``compile_module`` through
the partitioner into ``extract_var_range_info``."""

import os
import unittest

import pytest
import sympy
import torch
import torch_tensorrt as torchtrt
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo._compiler import _build_user_symbol_bounds
from torch_tensorrt.dynamo.utils import extract_var_range_info

assertions = unittest.TestCase()


def _first_sym_placeholder(ep: torch.export.ExportedProgram, sym_dim: int = 0):
    """First placeholder with a ``SymInt`` at ``sym_dim``; skips lifted
    param/buffer placeholders that ``strict=False`` may prepend."""
    for node in ep.graph.nodes:
        if node.op != "placeholder":
            continue
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor):
            continue
        if sym_dim < len(val.size()) and isinstance(val.size()[sym_dim], torch.SymInt):
            return val, val.size()[sym_dim]
    return None, None


class _SmallLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.linear(x).relu()


@pytest.mark.unit
def test_extract_var_range_info_fills_unbounded_max_from_user():
    """User bounds must fill the gap when the exporter's upper is ``int_oo``,
    and successive calls must not leak state across invocations."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(2, 8, device="cuda")

    dyn_batch = torch.export.Dim.DYNAMIC
    ep = torch.export.export(
        model, (sample,), dynamic_shapes=({0: dyn_batch},), strict=False
    )

    fake_val, sym_dim = _first_sym_placeholder(ep, sym_dim=0)
    assert sym_dim is not None
    assert isinstance(sym_dim, torch.SymInt)
    expr = sym_dim.node.expr
    assert isinstance(expr, sympy.Symbol)

    assert extract_var_range_info(sym_dim)["max"] is None

    info_with_user = extract_var_range_info(sym_dim, user_symbol_bounds={expr: (1, 8)})
    assert info_with_user["max"] == 8
    assert info_with_user["min"] == 1

    # Different map on the same SymInt must return the new bounds (no caching).
    info_wider = extract_var_range_info(sym_dim, user_symbol_bounds={expr: (1, 16)})
    assert info_wider["max"] == 16
    assert info_wider["min"] == 1

    # Dropping the map must revert (no ShapeEnv mutation).
    assert extract_var_range_info(sym_dim)["max"] is None


@pytest.mark.unit
def test_extract_var_range_info_does_not_widen_lower_bound():
    """Lower bound is intersected so the 0/1 specialization survives even
    when the user passes ``min_shape=0``."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(2, 8, device="cuda")
    ep = torch.export.export(
        model,
        (sample,),
        dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
        strict=False,
    )
    _, sym_dim = _first_sym_placeholder(ep, sym_dim=0)
    assert sym_dim is not None
    expr = sym_dim.node.expr

    info = extract_var_range_info(sym_dim, user_symbol_bounds={expr: (0, 8)})
    assert info["min"] == 1, "exporter lower must not be widened to user's 0"
    assert info["max"] == 8


@pytest.mark.unit
def test_extract_var_range_info_does_not_override_finite_max():
    """A finite exporter max must win over ``user_symbol_bounds``."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(2, 8, device="cuda")
    ep = torch.export.export(
        model,
        (sample,),
        dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=4)},),
        strict=False,
    )
    _, sym_dim = _first_sym_placeholder(ep, sym_dim=0)
    assert sym_dim is not None
    expr = sym_dim.node.expr

    info = extract_var_range_info(sym_dim, user_symbol_bounds={expr: (1, 16)})
    assert info["max"] == 4


@pytest.mark.unit
def test_extract_var_range_info_handles_sympy_oo_from_bound_sympy():
    """Composite exprs (e.g. ``s0 + s1``) hit ``bound_sympy``, which returns
    ``sympy.oo`` (not ``int_oo``) for unbounded operands. A naive
    ``!= int_oo`` guard misses it and crashes on ``int(sympy.oo)``."""

    class _ConcatBatch(torch.nn.Module):
        def forward(self, a, b):
            return torch.cat([a, b], dim=0)

    model = _ConcatBatch().eval().cuda()
    a = torch.randn(2, 8, device="cuda")
    b = torch.randn(3, 8, device="cuda")
    ep = torch.export.export(
        model,
        (a, b),
        dynamic_shapes=(
            {0: torch.export.Dim.DYNAMIC},
            {0: torch.export.Dim.DYNAMIC},
        ),
        strict=False,
    )

    cat_node = next(
        (
            n
            for n in ep.graph.nodes
            if n.op == "call_function" and "cat" in str(n.target)
        ),
        None,
    )
    if cat_node is None or "val" not in cat_node.meta:
        pytest.skip("no cat node with composite SymInt on this PyTorch")

    composite_dim = cat_node.meta["val"].size()[0]
    if not isinstance(composite_dim, torch.SymInt):
        pytest.skip("composite dim specialized to a concrete int")
    if isinstance(composite_dim.node.expr, sympy.Symbol):
        pytest.skip("expr collapsed to a plain symbol on this PyTorch")

    # Pre-fix this raised ``AttributeError`` from ``int(sympy.oo)``.
    info = extract_var_range_info(composite_dim)
    assert info["max"] is None
    assert info["min"] is not None


@pytest.mark.unit
def test_build_user_symbol_bounds_uses_dynamic_inputs_only():
    """Static ``Input``s contribute nothing to the user-bounds map."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(2, 8, device="cuda")
    ep = torch.export.export(
        model,
        (sample,),
        dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
        strict=False,
    )

    dynamic_input = Input(
        min_shape=(1, 8),
        opt_shape=(4, 8),
        max_shape=(8, 8),
        dtype=torch.float32,
    )
    bounds = _build_user_symbol_bounds(ep.module(), [dynamic_input], {})
    assert len(bounds) == 1
    ((sym, (lo, hi)),) = bounds.items()
    assert isinstance(sym, sympy.Symbol)
    assert (lo, hi) == (1, 8)

    # Static input -> empty map.
    static_input = Input(shape=(2, 8), dtype=torch.float32)
    assert _build_user_symbol_bounds(ep.module(), [static_input], {}) == {}


@pytest.mark.unit
def test_build_user_symbol_bounds_warns_on_01_specialization(caplog):
    """``user_min=1, exp_min=2`` is the 0/1 specialization artifact -- warn,
    don't raise."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(2, 8, device="cuda")
    ep = torch.export.export(
        model,
        (sample,),
        dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
        strict=False,
    )
    input_min1 = Input(
        min_shape=(1, 8),
        opt_shape=(4, 8),
        max_shape=(8, 8),
        dtype=torch.float32,
    )

    import logging

    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.dynamo._compiler"):
        _build_user_symbol_bounds(ep.module(), [input_min1], {})

    msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    if msgs:
        assert "0/1 specialization" in "\n".join(msgs)


@pytest.mark.unit
def test_build_user_symbol_bounds_raises_when_input_min_genuinely_below_export():
    """``user_min < exp_min`` (not the 1->2 case) must raise."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(10, 8, device="cuda")
    ep = torch.export.export(
        model,
        (sample,),
        dynamic_shapes=({0: torch.export.Dim("batch", min=10, max=20)},),
        strict=False,
    )
    input_too_low = Input(
        min_shape=(2, 8),
        opt_shape=(5, 8),
        max_shape=(10, 8),
        dtype=torch.float32,
    )
    with pytest.raises(ValueError) as exc_info:
        _build_user_symbol_bounds(ep.module(), [input_too_low], {})
    msg = str(exc_info.value)
    assert "Input.min_shape" in msg and "exported program" in msg
    assert "re-export" in msg.lower() or "Input.min_shape >=" in msg


@pytest.mark.unit
def test_build_user_symbol_bounds_raises_when_input_max_above_export():
    """``user_max > exp_max`` must raise (TRT would reject at runtime)."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(10, 8, device="cuda")
    ep = torch.export.export(
        model,
        (sample,),
        dynamic_shapes=({0: torch.export.Dim("batch", min=10, max=20)},),
        strict=False,
    )
    too_wide_input = Input(
        min_shape=(10, 8),
        opt_shape=(15, 8),
        max_shape=(25, 8),
        dtype=torch.float32,
    )
    with pytest.raises(ValueError) as exc_info:
        _build_user_symbol_bounds(ep.module(), [too_wide_input], {})
    msg = str(exc_info.value)
    assert "Input.max_shape" in msg and "exported program" in msg
    assert "re-export" in msg.lower() or "Input.max_shape <=" in msg


@pytest.mark.unit
def test_build_user_symbol_bounds_no_warning_on_matching_bounds(caplog):
    """Matching bounds must not warn or raise."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(10, 8, device="cuda")
    ep = torch.export.export(
        model,
        (sample,),
        dynamic_shapes=({0: torch.export.Dim("batch", min=10, max=20)},),
        strict=False,
    )
    matching_input = Input(
        min_shape=(10, 8),
        opt_shape=(15, 8),
        max_shape=(20, 8),
        dtype=torch.float32,
    )

    import logging

    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.dynamo._compiler"):
        _build_user_symbol_bounds(ep.module(), [matching_input], {})

    warning_messages = [
        rec.message for rec in caplog.records if rec.levelno >= logging.WARNING
    ]
    assert not warning_messages, f"unexpected warnings: {warning_messages}"


@pytest.mark.unit
def test_build_user_symbol_bounds_narrows_to_user_on_subset(caplog):
    """Strict-subset Input narrows the engine profile to the user's bounds.
    No warning -- the user got exactly what they asked for; just an info log."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(10, 8, device="cuda")
    ep = torch.export.export(
        model,
        (sample,),
        dynamic_shapes=({0: torch.export.Dim("batch", min=10, max=20)},),
        strict=False,
    )
    subset_input = Input(
        min_shape=(12, 8),
        opt_shape=(15, 8),
        max_shape=(18, 8),
        dtype=torch.float32,
    )

    import logging

    with caplog.at_level(logging.INFO, logger="torch_tensorrt.dynamo._compiler"):
        bounds = _build_user_symbol_bounds(ep.module(), [subset_input], {})

    # No warning -- subset is a valid, intended narrowing.
    warning_messages = [
        rec.message for rec in caplog.records if rec.levelno >= logging.WARNING
    ]
    assert not warning_messages, f"unexpected warnings: {warning_messages}"

    # Narrowing must surface as an info log so users can see what envelope
    # their engine actually has.
    info_messages = [
        rec.message for rec in caplog.records if rec.levelno == logging.INFO
    ]
    assert any("Narrowing engine profile" in m for m in info_messages), info_messages

    # And the engine-side resolution must actually narrow to (12, 18).
    _, sym_dim = _first_sym_placeholder(ep, sym_dim=0)
    assert sym_dim is not None
    info = extract_var_range_info(sym_dim, user_symbol_bounds=bounds)
    assert info["min"] == 12, info
    assert info["max"] == 18, info


@pytest.mark.unit
def test_build_user_symbol_bounds_no_warning_on_dim_dynamic(caplog):
    """Dim.DYNAMIC + Input is the intended fill case -- no mismatch warning."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(2, 8, device="cuda")
    ep = torch.export.export(
        model,
        (sample,),
        dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
        strict=False,
    )
    fill_input = Input(
        min_shape=(1, 8),
        opt_shape=(4, 8),
        max_shape=(8, 8),
        dtype=torch.float32,
    )

    import logging

    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.dynamo._compiler"):
        _build_user_symbol_bounds(ep.module(), [fill_input], {})

    mismatch_warnings = [
        rec.message
        for rec in caplog.records
        if rec.levelno >= logging.WARNING and "differ from the exporter" in rec.message
    ]
    assert not mismatch_warnings, f"unexpected mismatch warnings: {mismatch_warnings}"


@pytest.mark.unit
def test_build_user_symbol_bounds_mixed_tensor_and_symint():
    """Tensor + SymInt placeholders: only the tensor contributes to the map
    (SymInt is filtered by the ``isinstance(fake_val, torch.Tensor)`` guard)."""

    class _TensorAndScalar(torch.nn.Module):
        def forward(self, x, k):
            return x.relu() + k

    model = _TensorAndScalar().eval().cuda()
    x_sample = torch.randn(2, 8, device="cuda")
    k_sample = 4  # >1 to avoid 0/1 specialization

    dyn = torch.export.Dim.DYNAMIC
    ep = torch.export.export(
        model,
        (x_sample, k_sample),
        dynamic_shapes=({0: dyn}, dyn),
        strict=False,
    )

    placeholders = [n for n in ep.graph.nodes if n.op == "placeholder"]
    metas = [p.meta.get("val") for p in placeholders]
    if not any(isinstance(m, torch.SymInt) for m in metas):
        pytest.skip("export collapsed the scalar SymInt on this PyTorch")
    assert any(isinstance(m, torch.Tensor) for m in metas)

    tensor_input = Input(
        min_shape=(1, 8),
        opt_shape=(4, 8),
        max_shape=(8, 8),
        dtype=torch.float32,
    )
    # Index-aligned placeholder; never inspected (SymInt path skips it).
    scalar_input = Input(
        min_shape=(1,),
        opt_shape=(1,),
        max_shape=(1,),
        dtype=torch.int64,
    )

    bounds = _build_user_symbol_bounds(ep.module(), [tensor_input, scalar_input], {})
    assert len(bounds) == 1, bounds
    ((sym, (lo, hi)),) = bounds.items()
    assert isinstance(sym, sympy.Symbol)
    assert (lo, hi) == (1, 8)


@pytest.mark.unit
def test_build_user_symbol_bounds_mixed_arg_and_kwarg():
    """Positional args (matched by index) and kwargs (matched by name) both
    contribute to the map."""

    class _TwoTensors(torch.nn.Module):
        def forward(self, x, y):
            # Independent paths so the exporter doesn't unify x[0] and y[0].
            return x.relu(), y.relu()

    model = _TwoTensors().eval().cuda()
    # Distinct sizes discourage symbol unification.
    x_sample = torch.randn(2, 8, device="cuda")
    y_sample = torch.randn(3, 8, device="cuda")

    dyn = torch.export.Dim.DYNAMIC
    ep = torch.export.export(
        model,
        args=(x_sample,),
        kwargs={"y": y_sample},
        dynamic_shapes={"x": {0: dyn}, "y": {0: dyn}},
        strict=False,
    )

    placeholder_names = [n.target for n in ep.graph.nodes if n.op == "placeholder"]
    assert "x" in placeholder_names, placeholder_names
    assert "y" in placeholder_names, placeholder_names

    # Distinct bounds for x and y so we can verify the kwarg path by value.
    x_input = Input(
        min_shape=(1, 8),
        opt_shape=(2, 8),
        max_shape=(8, 8),
        dtype=torch.float32,
    )
    y_input = Input(
        min_shape=(2, 8),
        opt_shape=(3, 8),
        max_shape=(6, 8),
        dtype=torch.float32,
    )

    bounds = _build_user_symbol_bounds(
        ep.module(),
        sample_arg_inputs=[x_input],
        sample_kwarg_inputs={"y": y_input},
    )

    bounds_values = set(bounds.values())
    assert (1, 8) in bounds_values, bounds  # positional-arg branch
    assert (2, 6) in bounds_values, bounds  # kwarg branch
    assert len(bounds) == 2, bounds


@pytest.mark.unit
@pytest.mark.critical
def test_dim_dynamic_save_preserves_range_constraints(tmpdir):
    """End-to-end: Dim.DYNAMIC + Input(max_shape) round-trips through
    ``torchtrt.save`` without mutating the exporter's range_constraints."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(2, 8, device="cuda")

    dyn_batch = torch.export.Dim.DYNAMIC
    exp_program = torch.export.export(
        model, (sample,), dynamic_shapes=({0: dyn_batch},), strict=False
    )

    # Snapshot before Torch-TRT touches anything.
    expected_constraints = {
        str(k): (v.lower, v.upper) for k, v in exp_program.range_constraints.items()
    }

    compile_spec = {
        "inputs": [
            Input(
                min_shape=(1, 8),
                opt_shape=(4, 8),
                max_shape=(8, 8),
                dtype=torch.float32,
            )
        ],
        "ir": "dynamo",
        "min_block_size": 1,
        "cache_built_engines": False,
        "reuse_cached_engines": False,
    }
    trt_module = torchtrt.dynamo.compile(exp_program, **compile_spec)

    after_constraints = {
        str(k): (v.lower, v.upper) for k, v in exp_program.range_constraints.items()
    }
    assertions.assertEqual(expected_constraints, after_constraints)

    trt_ep_path = os.path.join(tmpdir, "trt_dim_dynamic.ep")
    torchtrt.save(
        trt_module,
        trt_ep_path,
        output_format="exported_program",
        arg_inputs=[sample],
        dynamic_shapes=({0: dyn_batch},),
        retrace=True,
    )

    reloaded = torch.export.load(trt_ep_path)
    reloaded_constraints = {
        str(k): (v.lower, v.upper) for k, v in reloaded.range_constraints.items()
    }
    assertions.assertEqual(expected_constraints, reloaded_constraints)

    # Engine accepts shapes across [min_shape, max_shape].
    trt_module(torch.randn(1, 8, device="cuda"))
    trt_module(torch.randn(4, 8, device="cuda"))
    trt_module(torch.randn(8, 8, device="cuda"))

    # And rejects shapes beyond ``Input.max_shape`` even though the eager
    # graph is unbounded -- ``Input(max_shape=8)`` is a strict cap. Pins
    # the contract against regressions that re-widen to the heuristic.
    too_big = torch.randn(16, 8, device="cuda")
    with assertions.assertRaises(Exception):
        trt_module(too_big)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
