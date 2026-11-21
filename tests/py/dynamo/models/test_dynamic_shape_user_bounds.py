# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# type: ignore
"""Tests for ``user_symbol_bounds`` plumbed from ``compile_module`` through
the partitioner into ``extract_var_range_info``."""

import os
import sys
import unittest

import pytest
import sympy
import torch
import torch_tensorrt as torchtrt
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo._compiler import _build_user_symbol_bounds
from torch_tensorrt.dynamo.partitioning.common import (
    construct_dynamic_input,
    construct_submodule_inputs,
)
from torch_tensorrt.dynamo.runtime import TorchTensorRTModule
from torch_tensorrt.dynamo.utils import extract_var_range_info

import tensorrt as trt  # isort: skip  # imported after torch_tensorrt for RTX alias

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


def _engine_profiles(trt_mod: torch.nn.Module) -> dict:
    """``{input name: (min, opt, max)}`` of the single engine in ``trt_mod``."""
    plans = [
        m.serialized_engine
        for _, m in trt_mod.named_modules()
        if isinstance(m, TorchTensorRTModule)
    ]
    assert len(plans) == 1, plans
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(
        plans[0]
    )
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    return {
        name: tuple(tuple(s) for s in engine.get_tensor_profile_shape(name, 0))
        for name in names
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
    }


class _Reshape(torch.nn.Module):
    def forward(self, x):
        return x.reshape(-1, 4, 8) * 2


class _CatAdd(torch.nn.Module):
    def forward(self, x, y):
        return torch.cat([x, x]) + y


class _SmallLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.linear(x).relu()


@torch.library.custom_op("torchtrt_profile_test::dynamic_rows", mutates_args=())
def _profile_dynamic_rows(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_profile_dynamic_rows.register_fake
def _profile_dynamic_rows_fake(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty(torch.library.get_ctx().new_dynamic_size(), x.shape[1])


class _ConstrainedDynamicRows(torch.nn.Module):
    def __init__(self, lower: int, upper: int) -> None:
        super().__init__()
        self.lower = lower
        self.upper = upper

    def forward(self, x):
        rows = torch.ops.torchtrt_profile_test.dynamic_rows(x)
        torch._check(rows.shape[0] >= self.lower)
        torch._check(rows.shape[0] <= self.upper)
        return rows


def _constrained_dynamic_shape(lower: int, upper: int) -> torch.Size:
    exported = torch.export.export(
        _ConstrainedDynamicRows(lower, upper), (torch.ones(1100, 4),)
    )
    dynamic_rows = next(
        node
        for node in exported.graph.nodes
        if node.target == torch.ops.torchtrt_profile_test.dynamic_rows.default
    )
    return dynamic_rows.meta["val"].shape


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

    info_with_user = extract_var_range_info(
        sym_dim, user_symbol_bounds={expr: (1, 8, None)}
    )
    assert info_with_user["max"] == 8
    assert info_with_user["min"] == 1

    # Different map on the same SymInt must return the new bounds (no caching).
    info_wider = extract_var_range_info(
        sym_dim, user_symbol_bounds={expr: (1, 16, None)}
    )
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

    info = extract_var_range_info(sym_dim, user_symbol_bounds={expr: (0, 8, None)})
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

    info = extract_var_range_info(sym_dim, user_symbol_bounds={expr: (1, 16, None)})
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
    bounds = _build_user_symbol_bounds(
        ep.module(), [dynamic_input], {}, ep.graph_signature
    )
    assert len(bounds) == 1
    ((sym, (lo, hi, opt)),) = bounds.items()
    assert isinstance(sym, sympy.Symbol)
    assert (lo, hi, opt) == (1, 8, 4)

    # Static input -> empty map.
    static_input = Input(shape=(2, 8), dtype=torch.float32)
    assert (
        _build_user_symbol_bounds(ep.module(), [static_input], {}, ep.graph_signature)
        == {}
    )


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
        _build_user_symbol_bounds(ep.module(), [input_min1], {}, ep.graph_signature)

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
        _build_user_symbol_bounds(ep.module(), [input_too_low], {}, ep.graph_signature)
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
        _build_user_symbol_bounds(ep.module(), [too_wide_input], {}, ep.graph_signature)
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
        _build_user_symbol_bounds(ep.module(), [matching_input], {}, ep.graph_signature)

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
        bounds = _build_user_symbol_bounds(
            ep.module(), [subset_input], {}, ep.graph_signature
        )

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
    assert info["opt"] == 15, info


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
        _build_user_symbol_bounds(ep.module(), [fill_input], {}, ep.graph_signature)

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

    bounds = _build_user_symbol_bounds(
        ep.module(), [tensor_input, scalar_input], {}, ep.graph_signature
    )
    assert len(bounds) == 1, bounds
    ((sym, (lo, hi, opt)),) = bounds.items()
    assert isinstance(sym, sympy.Symbol)
    assert (lo, hi, opt) == (1, 8, 4)


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
        graph_signature=ep.graph_signature,
    )

    bounds_values = set(bounds.values())
    assert (1, 8, 2) in bounds_values, bounds  # positional-arg branch
    assert (2, 6, 3) in bounds_values, bounds  # kwarg branch
    assert len(bounds) == 2, bounds


@pytest.mark.unit
@pytest.mark.parametrize("opt_shape", [(2, 1), (2,)])
def test_build_user_symbol_bounds_ignores_opt_outside_range(opt_shape, caplog):
    """An ``opt_shape`` outside ``[min_shape, max_shape]``, or too short, is
    ignored with a warning, so the dim keeps the export example's opt."""
    import logging

    class _Double(torch.nn.Module):
        def forward(self, x):
            return x * 2

    ep = torch.export.export(
        _Double(),
        (torch.randn(2, 4),),
        dynamic_shapes=({1: torch.export.Dim("n", min=1, max=16)},),
        strict=False,
    )
    spec = Input(min_shape=(2, 2), opt_shape=opt_shape, max_shape=(2, 16))
    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.dynamo._compiler"):
        bounds = _build_user_symbol_bounds(ep.module(), [spec], {}, ep.graph_signature)
    assert list(bounds.values()) == [(2, 16, None)], bounds
    assert any("does not fit" in r.message for r in caplog.records)


class _HalfRows(torch.nn.Module):
    def forward(self, x):
        return x.flatten().reshape(x.shape[0] // 2, -1)


class _FirstHalf(torch.nn.Module):
    def forward(self, x):
        return x.reshape(2, -1, 8)[0]


class _CountRows(torch.nn.Module):
    def forward(self, x):
        count = torch.nonzero(x[:, 0] > 0).shape[0]
        return x.flatten().reshape(count, -1)


class _ScaleByCount(torch.nn.Module):
    def forward(self, x):
        count = torch.nonzero(x[:, 0] > 0).shape[0]
        torch._check(count >= 1)
        return x * count


@pytest.mark.unit
@pytest.mark.parametrize(
    "model, min_rows, opt_rows, deferred, expected_opt",
    [
        (_Reshape(), 4, 6, False, None),  # reshape(-1, 4, 8) needs rows % 4 == 0
        (_Reshape(), 4, 6, True, None),  # the same rule as a deferred assert
        (_Reshape(), 4, 4, False, 4),  # rows // 4 != 1 also fails at the min 4
        (_FirstHalf(), 1, 9, False, None),  # rows % 2 == 0 also fails at the min 1
        (_HalfRows(), 1, 1, False, None),  # a rule divides by 1 // 2 = 0
        (_CountRows(), 1, 1, False, None),  # a rule names a data-dependent size
        (_CountRows(), 1, 16, False, 16),  # opt equals the export example's 16
        (_ScaleByCount(), 1, 1, False, 1),  # count >= 1 does not name rows
    ],
)
def test_build_user_symbol_bounds_checks_opt_against_guards(
    model, min_rows, opt_rows, deferred, expected_opt, caplog
):
    """TensorRT checks the shape rules at opt, so an opt that breaks a guard,
    or at which a guard cannot be evaluated, is dropped with a warning instead
    of failing the build. An opt equal to the export example is not checked."""
    import logging

    x = torch.randn(16, 8)
    x[:8, 0] = 1.0
    x[8:, 0] = -1.0
    ep = torch.export.export(
        model,
        (x,),
        dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
        strict=False,
        prefer_deferred_runtime_asserts_over_guards=deferred,
    )
    spec = Input(min_shape=(min_rows, 8), opt_shape=(opt_rows, 8), max_shape=(64, 8))
    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.dynamo._compiler"):
        bounds = _build_user_symbol_bounds(ep.module(), [spec], {}, ep.graph_signature)
    assert [opt for _, _, opt in bounds.values()] == [expected_opt], bounds
    warned = any("shape guard" in r.message for r in caplog.records)
    assert warned == (expected_opt is None)


class _CausalMask(torch.nn.Module):
    def forward(self, x):
        rows = x.shape[0]
        return x.sum(1) + torch.triu(torch.full((rows, rows), float("-inf")), 1)


@pytest.mark.unit
def test_build_user_symbol_bounds_keeps_opt_at_guard_false_at_engine_min(tmpdir):
    """After ``torch.export.save`` and ``load``, re-tracing a causal mask adds
    guards such as ``rows**2 >= 2``. They are false at the engine min 1 too,
    where the engine builds without ``opt_shape``, so they are not TensorRT
    rules and opt 1 is kept."""
    ep = torch.export.export(
        _CausalMask(),
        (torch.randn(16, 8),),
        dynamic_shapes=({0: torch.export.Dim("rows", min=1, max=32)},),
        strict=False,
    )
    path = os.path.join(tmpdir, "causal.pt2")
    torch.export.save(ep, path)
    ep = torch.export.load(path).run_decompositions({})
    spec = Input(min_shape=(1, 8), opt_shape=(1, 8), max_shape=(32, 8))
    bounds = _build_user_symbol_bounds(ep.module(), [spec], {}, ep.graph_signature)
    assert [opt for _, _, opt in bounds.values()] == [1], bounds


@pytest.mark.unit
@pytest.mark.parametrize(
    "x_opt, y_opt, warns",
    [
        (3, 6, False),
        (3, 7, True),
        (20, 8, False),  # x's opt is ignored, so 2*b keeps the example's 8
    ],
)
def test_build_user_symbol_bounds_warns_when_derived_opt_is_ignored(
    x_opt, y_opt, warns, caplog
):
    """The opt of a derived dim such as ``2*b`` follows ``b``, so a different
    ``opt_shape`` value for it is ignored with a warning."""
    import logging

    b = torch.export.Dim("b", min=1, max=16)
    ep = torch.export.export(
        _CatAdd(),
        (torch.randn(4, 8), torch.randn(8, 8)),
        dynamic_shapes=({0: b}, {0: 2 * b}),
        strict=False,
    )
    specs = [
        Input(min_shape=(1, 8), opt_shape=(x_opt, 8), max_shape=(16, 8)),
        Input(min_shape=(2, 8), opt_shape=(y_opt, 8), max_shape=(32, 8)),
    ]
    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.dynamo._compiler"):
        _build_user_symbol_bounds(ep.module(), specs, {}, ep.graph_signature)
    assert any("derived" in r.message for r in caplog.records) == warns


@pytest.mark.unit
def test_build_user_symbol_bounds_keeps_example_opt_with_profiles():
    """Each profile of ``Input(profiles=...)`` brings its own opts, so another
    input's ``opt_shape`` is not used: the guard check could not see the
    opts of later profiles."""

    class _Slice(torch.nn.Module):
        def forward(self, x, y):
            if y.shape[0] > x.shape[0]:
                return x * 0
            return x[: y.shape[0]] + y

    ep = torch.export.export(
        _Slice(),
        (torch.randn(16, 8), torch.randn(2, 8)),
        dynamic_shapes=(
            {0: torch.export.Dim("a", min=1, max=64)},
            {0: torch.export.Dim("b", min=1, max=64)},
        ),
        strict=False,
    )
    x_spec = Input(
        profiles=[
            {"min_shape": (1, 8), "opt_shape": (4, 8), "max_shape": (8, 8)},
            {"min_shape": (8, 8), "opt_shape": (32, 8), "max_shape": (64, 8)},
        ]
    )
    y_spec = Input(min_shape=(1, 8), opt_shape=(2, 8), max_shape=(8, 8))
    bounds = _build_user_symbol_bounds(
        ep.module(), [x_spec, y_spec], {}, ep.graph_signature
    )
    assert [opt for _, _, opt in bounds.values()] == [None, None], bounds


@pytest.mark.unit
def test_construct_submodule_inputs_user_opt_with_data_dependent_input():
    """A data-dependent input dim has no export example and keeps the midpoint
    opt, without making another input of the engine lose its user opt."""

    class _Rows(torch.nn.Module):
        def forward(self, x):
            rows = torch.ops.torchtrt_profile_test.dynamic_rows(x)
            torch._check(rows.shape[0] >= 1000)
            torch._check(rows.shape[0] <= 1400)
            return rows

    exported = torch.export.export(
        _Rows(),
        (torch.ones(1100, 4),),
        dynamic_shapes=({0: torch.export.Dim("b", min=1, max=2000)},),
    )
    x, b = _first_sym_placeholder(exported)
    rows = next(
        node
        for node in exported.graph.nodes
        if node.target == torch.ops.torchtrt_profile_test.dynamic_rows.default
    )
    graph = torch.fx.Graph()
    for name, val in (("x", x), ("rows", rows.meta["val"])):
        graph.placeholder(name).meta["val"] = val
    graph.output(None)

    x_spec, rows_spec = construct_submodule_inputs(
        torch.fx.GraphModule(torch.nn.Module(), graph),
        user_symbol_bounds={b.node.expr: (1, 2000, 3)},
    )
    assert x_spec.shape["opt_shape"] == (3, 4)
    assert rows_spec.shape["opt_shape"] == (1200, 4)


@pytest.mark.unit
def test_build_user_symbol_bounds_checks_every_input_sharing_a_symbol(caplog):
    """A different (min, max, opt) for one shared dimension is reported
    instead of being dropped silently."""

    class _Add(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    n = torch.export.Dim("n", min=1, max=16)
    ep = torch.export.export(
        _Add(),
        (torch.randn(4, 8), torch.randn(4, 8)),
        dynamic_shapes=({0: n}, {0: n}),
        strict=False,
    )
    x_spec = Input(min_shape=(1, 8), opt_shape=(2, 8), max_shape=(16, 8))
    y_spec = Input(min_shape=(1, 8), opt_shape=(8, 8), max_shape=(16, 8))

    import logging

    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.dynamo._compiler"):
        bounds = _build_user_symbol_bounds(
            ep.module(), [x_spec, y_spec], {}, ep.graph_signature
        )
    assert list(bounds.values()) == [(1, 16, 2)], bounds
    assert any("shares dynamic dimension" in r.message for r in caplog.records)


@pytest.mark.unit
@pytest.mark.parametrize("save_load", [False, True])
def test_compile_exported_program_uses_input_opt_shape(save_load, tmpdir):
    """The engine profile's opt follows ``Input.opt_shape``, not the export
    example, also after ``torch.export.save`` and ``torch.export.load``, where
    compile finds the guard ``batch != 1``."""

    model = _SmallLinear().eval().cuda()
    example = torch.randn(16, 8, device="cuda")
    ep = torch.export.export(
        model,
        (example,),
        dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=32)},),
        strict=False,
    )
    if save_load:
        path = os.path.join(tmpdir, "linear.pt2")
        torch.export.save(ep, path)
        ep = torch.export.load(path)
    spec = Input(
        min_shape=(1, 8),
        opt_shape=(1, 8),
        max_shape=(32, 8),
        dtype=torch.float32,
    )
    trt_mod = torchtrt.dynamo.compile(ep, inputs=[spec], min_block_size=1)
    assert _engine_profiles(trt_mod) == {"x": ((1, 8), (1, 8), (32, 8))}

    x = torch.randn(3, 8, device="cuda")
    torch.testing.assert_close(trt_mod(x), model(x), rtol=1e-2, atol=1e-2)


@pytest.mark.unit
@pytest.mark.parametrize(
    "dim_max, opt_rows, expected_x_opt",
    [
        (None, 3, 3),  # fits, so the user opt is used for b and 2*b
        # Known limitation: 2*b ignores max_shape and gets the default max
        # 16384, so 2*9000 does not fit.
        (None, 9000, 16),
        (10000, 1, 16),  # 2*1 is below the min of 2*b (b's min is 2)
    ],
)
def test_compile_exported_program_user_opt_with_derived_dim(
    dim_max, opt_rows, expected_x_opt
):
    """With a derived input dim (``2*b``), the user opt is used only if it fits
    every engine input; otherwise the engine keeps the export example's opt, so
    models that compile without ``opt_shape`` still compile."""

    b = torch.export.Dim("b", max=dim_max)
    ep = torch.export.export(
        _CatAdd(),
        (torch.randn(16, 8, device="cuda"), torch.randn(32, 8, device="cuda")),
        dynamic_shapes=({0: b}, {0: 2 * b}),
        strict=False,
    )
    specs = [
        Input(min_shape=(1, 8), opt_shape=(opt_rows, 8), max_shape=(10000, 8)),
        Input(min_shape=(2, 8), opt_shape=(2 * opt_rows, 8), max_shape=(20000, 8)),
    ]
    trt_mod = torchtrt.dynamo.compile(ep, inputs=specs, min_block_size=1)
    profiles = _engine_profiles(trt_mod)
    assert profiles["x"][1] == (expected_x_opt, 8), profiles
    assert profiles["y"][1] == (2 * expected_x_opt, 8), profiles

    x, y = torch.randn(3, 8, device="cuda"), torch.randn(6, 8, device="cuda")
    torch.testing.assert_close(trt_mod(x, y), torch.cat([x, x]) + y)


@pytest.mark.unit
@pytest.mark.parametrize("save_load", [False, True])
def test_compile_exported_program_drops_opt_that_breaks_a_guard(
    save_load, tmpdir, caplog
):
    """An opt that breaks a shape guard is dropped before the engine build, so
    the model still compiles with the export example's opt, also after
    ``torch.export.save`` and ``torch.export.load``."""
    import logging

    ep = torch.export.export(
        _Reshape(),
        (torch.randn(16, 8, device="cuda"),),
        dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
        strict=False,
    )
    if save_load:
        path = os.path.join(tmpdir, "reshape.pt2")
        torch.export.save(ep, path)
        ep = torch.export.load(path)
    spec = Input(
        min_shape=(4, 8), opt_shape=(6, 8), max_shape=(64, 8), dtype=torch.float32
    )
    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.dynamo._compiler"):
        trt_mod = torchtrt.dynamo.compile(ep, inputs=[spec], min_block_size=1)
    assert any("shape guard" in r.message for r in caplog.records)
    assert _engine_profiles(trt_mod) == {"x": ((4, 8), (16, 8), (64, 8))}

    x = torch.randn(8, 8, device="cuda")
    torch.testing.assert_close(trt_mod(x), _Reshape()(x))


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


@pytest.mark.unit
def test_construct_dynamic_input_uses_profile_midpoint():
    symbolic_shape = _constrained_dynamic_shape(1000, 1200)
    input_spec = construct_dynamic_input(
        symbolic_shape, torch.float32, name="dynamic_rows"
    )

    assert input_spec.shape["min_shape"] == (1000, 4)
    assert input_spec.shape["opt_shape"] == (1100, 4)
    assert input_spec.shape["max_shape"] == (1200, 4)


@pytest.mark.unit
def test_construct_dynamic_input_unbounded_zero_min_uses_nonzero_max():
    """An unbounded data-dependent extent must not collapse to [0, 0, 0]."""

    class UnboundedDynamicRows(torch.nn.Module):
        def forward(self, x):
            return torch.ops.torchtrt_profile_test.dynamic_rows(x)

    exported = torch.export.export(UnboundedDynamicRows(), (torch.ones(4, 4),))
    dynamic_rows = next(
        node
        for node in exported.graph.nodes
        if node.target == torch.ops.torchtrt_profile_test.dynamic_rows.default
    )
    symbolic_shape = dynamic_rows.meta["val"].shape
    range_info = extract_var_range_info(symbolic_shape[0])

    assert range_info["min"] == 0
    assert range_info["max"] is None

    input_spec = construct_dynamic_input(
        symbolic_shape, torch.float32, name="dynamic_rows"
    )
    assert input_spec.shape["min_shape"] == (0, 4)
    assert input_spec.shape["opt_shape"] == (2048, 4)
    assert input_spec.shape["max_shape"] == (4096, 4)


@pytest.mark.unit
def test_construct_dynamic_input_rejects_invalid_profile(monkeypatch):
    """Generated profiles are validated before reaching the TensorRT builder."""

    symbolic_shape = _constrained_dynamic_shape(1, 8)
    monkeypatch.setattr(
        "torch_tensorrt.dynamo.partitioning.common.extract_var_range_info",
        lambda *_args, **_kwargs: {"min": 1, "opt": 9, "max": 8},
    )

    with pytest.raises(
        ValueError,
        match=(
            "Invalid optimization profile for input 'dynamic_rows' at dimension 0: "
            "expected min <= opt <= max, but got min=1, opt=9, max=8"
        ),
    ):
        construct_dynamic_input(symbolic_shape, torch.float32, name="dynamic_rows")


@pytest.mark.unit
def test_largest_size_symbol_bound_is_treated_as_unbounded():
    symbolic_shape = _constrained_dynamic_shape(1, sys.maxsize - 1)
    range_info = extract_var_range_info(symbolic_shape[0])
    input_spec = construct_dynamic_input(
        symbolic_shape, torch.float32, name="dynamic_rows"
    )

    assert range_info["max"] is None
    assert input_spec.shape["min_shape"] == (1, 4)
    assert input_spec.shape["opt_shape"] == (2048, 4)
    assert input_spec.shape["max_shape"] == (4096, 4)


@pytest.mark.unit
def test_construct_dynamic_input_preserves_zero_minimum():
    """Data-dependent extents retain their legal zero profile minimum."""

    class Nonzero(torch.nn.Module):
        def forward(self, x):
            return torch.nonzero(x)

    exported = torch.export.export(
        Nonzero(), (torch.tensor([True, False, True, False]),)
    )
    nonzero = next(
        node
        for node in exported.graph.nodes
        if node.target == torch.ops.aten.nonzero.default
    )
    fake_value = nonzero.meta["val"]
    assert isinstance(fake_value.shape[0], torch.SymInt)

    input_spec = construct_dynamic_input(
        fake_value.shape, fake_value.dtype, name="nonzero_output"
    )
    assert input_spec.shape["min_shape"] == (0, 1)
    assert input_spec.shape["opt_shape"] == (2, 1)
    assert input_spec.shape["max_shape"] == (4, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
