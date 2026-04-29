# type: ignore
"""Regression tests for user-supplied dynamic-shape bounds threading.

These tests exercise the read-only ``user_symbol_bounds`` map plumbed from
``compile_module`` through the partitioner into
``extract_var_range_info``. The map fills in upper bounds left unbounded by
``Dim.DYNAMIC`` *without* mutating the exporter's ``ShapeEnv``, so the
original ``range_constraints`` are preserved across
``torch_tensorrt.save(..., output_format="exported_program")``.
"""

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


class _SmallLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.linear(x).relu()


@pytest.mark.unit
def test_extract_var_range_info_fills_unbounded_max_from_user():
    """``extract_var_range_info`` must use ``user_symbol_bounds`` only when
    the exporter leaves ``var_to_range.upper`` as ``int_oo``."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(2, 8, device="cuda")

    # ``Dim.DYNAMIC`` (sentinel) leaves the upper bound unbounded.
    dyn_batch = torch.export.Dim.DYNAMIC
    ep = torch.export.export(
        model, (sample,), dynamic_shapes=({0: dyn_batch},), strict=False
    )

    # Pull the SymInt for the dynamic batch dim from the placeholder meta.
    placeholders = [n for n in ep.graph.nodes if n.op == "placeholder"]
    assert placeholders, "expected a placeholder for the model input"
    fake_val = placeholders[0].meta["val"]
    sym_dim = fake_val.size()[0]
    assert isinstance(sym_dim, torch.SymInt)
    expr = sym_dim.node.expr
    assert isinstance(expr, sympy.Symbol)

    # Without the user map the upper bound is unbounded -> ``max`` is None.
    info_no_user = extract_var_range_info(sym_dim)
    assert info_no_user["max"] is None

    # With the user map providing max=8, the exporter's gap is filled.
    # The lower stays at 1 (Dynamo's 0/1 specialization rewrite in
    # ``extract_var_range_info``); user_min=1 intersects to the same value.
    info_with_user = extract_var_range_info(sym_dim, user_symbol_bounds={expr: (1, 8)})
    assert info_with_user["max"] == 8
    assert info_with_user["min"] == 1

    # A second call with a *different* user map must return the new bounds:
    # ``extract_var_range_info`` is read-only w.r.t. ``ShapeEnv`` and must
    # therefore track whatever map the caller hands it on each invocation.
    info_with_user_wider = extract_var_range_info(
        sym_dim, user_symbol_bounds={expr: (1, 16)}
    )
    assert info_with_user_wider["max"] == 16
    assert info_with_user_wider["min"] == 1

    # And dropping the user map again must revert to the unbounded result -
    # i.e. the previous calls did not mutate any cached state.
    info_revert = extract_var_range_info(sym_dim)
    assert info_revert["max"] is None


@pytest.mark.unit
def test_extract_var_range_info_does_not_widen_lower_bound():
    """The user's lower bound must be intersected (``max(exporter, user)``)
    so the exporter's 0/1 specialization (lower == 1) survives even when
    the user passes ``min_shape=0``."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(2, 8, device="cuda")
    ep = torch.export.export(
        model,
        (sample,),
        dynamic_shapes=({0: torch.export.Dim.DYNAMIC},),
        strict=False,
    )
    placeholders = [n for n in ep.graph.nodes if n.op == "placeholder"]
    sym_dim = placeholders[0].meta["val"].size()[0]
    expr = sym_dim.node.expr

    info = extract_var_range_info(sym_dim, user_symbol_bounds={expr: (0, 8)})
    assert (
        info["min"] == 1
    ), "exporter's lower bound (1) must not be widened to user's 0"
    assert info["max"] == 8


@pytest.mark.unit
def test_extract_var_range_info_does_not_override_finite_max():
    """When the exporter already provides a finite upper bound (e.g. via
    ``Dim(max=...)``), the user's value must be ignored."""

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(2, 8, device="cuda")
    ep = torch.export.export(
        model,
        (sample,),
        dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=4)},),
        strict=False,
    )
    placeholders = [n for n in ep.graph.nodes if n.op == "placeholder"]
    sym_dim = placeholders[0].meta["val"].size()[0]
    expr = sym_dim.node.expr

    info = extract_var_range_info(sym_dim, user_symbol_bounds={expr: (1, 16)})
    assert (
        info["max"] == 4
    ), "finite exporter upper bound must win over user_symbol_bounds"


@pytest.mark.unit
def test_extract_var_range_info_handles_sympy_oo_from_bound_sympy():
    """Composite symbolic expressions (e.g. ``s0 + s1``) take the
    ``shape_env.bound_sympy`` fallback inside ``extract_var_range_info``.
    Sympy's arithmetic returns the *float-typed* ``sympy.oo`` for unbounded
    operands, which is a different object from PyTorch's ``int_oo``. A naive
    ``!= int_oo`` guard misses this case and crashes when ``int(sympy.oo)`` is
    attempted (``AttributeError: 'Infinity' object has no attribute '_mpf_'``).

    This test reproduces that path: a model that concatenates two tensors
    along a dynamic batch dimension, so the result's batch dim is
    ``s0 + s1`` -- a composite expression whose ``bound_sympy`` upper is
    ``sympy.oo`` when both ``s0`` and ``s1`` come from ``Dim.DYNAMIC``.
    """

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
        pytest.skip(
            "PyTorch version did not preserve a cat node with a composite "
            "symbolic batch dim; skipping regression."
        )

    composite_dim = cat_node.meta["val"].size()[0]
    if not isinstance(composite_dim, torch.SymInt):
        pytest.skip("composite batch dim was specialized to a concrete int")
    composite_expr = composite_dim.node.expr
    if isinstance(composite_expr, sympy.Symbol):
        pytest.skip(
            "expected a composite expression (e.g. s0 + s1); got a plain "
            "symbol on this PyTorch version"
        )

    # Without the fix, this call raises ``AttributeError`` from sympy when it
    # tries to coerce ``sympy.oo`` to int. With the fix, the unbounded upper
    # is reported as ``None`` so callers can fall back gracefully.
    info = extract_var_range_info(composite_dim)
    assert info["max"] is None, (
        f"composite expression with unbounded operand should report max=None, "
        f"got {info['max']!r}"
    )
    assert info["min"] is not None, "min should still resolve to an int"


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
def test_build_user_symbol_bounds_raises_when_input_extends_outside_export():
    """When the exporter declared finite bounds (e.g. ``Dim("batch",
    min=10, max=20)``) and the user passes an ``Input`` that extends
    *outside* that range, shapes the user listed in
    ``min_shape``/``max_shape`` are guaranteed to fail at runtime (the
    TRT engine profile follows the exporter). We surface that as a hard
    ``ValueError`` at compile time rather than letting the user discover
    it via a confusing TRT runtime error.
    """

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(10, 8, device="cuda")
    ep = torch.export.export(
        model,
        (sample,),
        dynamic_shapes=({0: torch.export.Dim("batch", min=10, max=20)},),
        strict=False,
    )

    # User declares a profile that extends below the exporter's lower
    # bound (user_min=2 < exp_min=10). The exporter would refuse batch=2
    # at runtime; raise at compile.
    incompatible_input = Input(
        min_shape=(2, 8),
        opt_shape=(5, 8),
        max_shape=(10, 8),
        dtype=torch.float32,
    )

    with pytest.raises(ValueError) as exc_info:
        _build_user_symbol_bounds(ep.module(), [incompatible_input], {})
    msg = str(exc_info.value)
    assert "extend outside" in msg
    assert "shape outside profile" in msg
    # The actionable guidance: either re-export, or stay inside the range.
    assert "re-export" in msg
    assert "Dim(" in msg
    # Both numeric pairs should appear so the user can see the conflict.
    assert "min=2" in msg and "max=10" in msg
    assert "min=10" in msg and "max=20" in msg


@pytest.mark.unit
def test_build_user_symbol_bounds_raises_when_input_max_above_export():
    """Symmetric to the lower-bound case: user_max > exp_max also
    guarantees a runtime error (TRT will reject shapes above the
    exporter's max). Compile must fail."""

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
        max_shape=(25, 8),  # exceeds exp_max=20
        dtype=torch.float32,
    )
    with pytest.raises(ValueError) as exc_info:
        _build_user_symbol_bounds(ep.module(), [too_wide_input], {})
    assert "extend outside" in str(exc_info.value)


@pytest.mark.unit
def test_build_user_symbol_bounds_no_warning_on_matching_bounds(caplog):
    """If the user's ``Input`` bounds exactly match the exporter's
    contract, no warning or error should fire -- the user got it right."""

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
    assert (
        not warning_messages
    ), f"matching bounds should not produce any warning, got: {warning_messages}"


@pytest.mark.unit
def test_build_user_symbol_bounds_warns_on_subset_input(caplog):
    """When the user's ``Input`` is a *strict subset* of the exporter's
    range (e.g. user `[12, 18]` inside exporter `[10, 20]`), no shape
    the user declared will fail at runtime -- the engine profile (which
    follows the exporter) accepts everything in the user's range. So
    this is *not* an error. But the user's narrower profile is silently
    widened, which is surprising; emit a warning so they know to
    re-export if they really want the narrower engine.
    """

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

    with caplog.at_level(logging.WARNING, logger="torch_tensorrt.dynamo._compiler"):
        # Must not raise.
        _build_user_symbol_bounds(ep.module(), [subset_input], {})

    warning_messages = [
        rec.message for rec in caplog.records if rec.levelno >= logging.WARNING
    ]
    assert (
        warning_messages
    ), "subset Input bounds should produce a 'narrower than exporter' warning"
    msg = "\n".join(warning_messages)
    assert "narrower than the exporter" in msg
    assert "engine profile will use the exporter's wider" in msg
    assert "re-export" in msg


@pytest.mark.unit
def test_build_user_symbol_bounds_no_warning_on_dim_dynamic(caplog):
    """``Dim.DYNAMIC`` leaves the exporter's upper unbounded; the user's
    ``Input`` bounds *fill the gap* there (the intended use). No warning
    should fire even if the user's lower differs from the exporter's
    (post 0/1-specialization) lower."""

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
    assert not mismatch_warnings, (
        f"Dim.DYNAMIC + Input is the intended fill case, no mismatch warning expected, "
        f"got: {mismatch_warnings}"
    )


@pytest.mark.unit
def test_build_user_symbol_bounds_mixed_tensor_and_symint():
    """Mix of tensor + SymInt placeholders: only the tensor placeholder
    contributes symbols to the user-bounds map. The SymInt placeholder
    is silently skipped by the ``isinstance(fake_val, torch.Tensor)``
    guard in ``_build_user_symbol_bounds``; downstream
    ``construct_submodule_inputs`` still handles the SymInt via its
    dedicated branch (falling back to the ``min*2^12`` heuristic since
    no entry for that symbol lands in ``user_symbol_bounds``).
    """

    class _TensorAndScalar(torch.nn.Module):
        def forward(self, x, k):
            return x.relu() + k

    model = _TensorAndScalar().eval().cuda()
    x_sample = torch.randn(2, 8, device="cuda")
    k_sample = 4  # avoid 0/1 specialization on the scalar input

    dyn = torch.export.Dim.DYNAMIC
    ep = torch.export.export(
        model,
        (x_sample, k_sample),
        dynamic_shapes=({0: dyn}, dyn),
        strict=False,
    )

    # Sanity: the export should produce both a tensor and a SymInt
    # placeholder. Some PyTorch versions may collapse the scalar input
    # if the dynamism can't be propagated through the graph - in that
    # case we just exercise the tensor-only path.
    placeholders = [n for n in ep.graph.nodes if n.op == "placeholder"]
    metas = [p.meta.get("val") for p in placeholders]
    if not any(isinstance(m, torch.SymInt) for m in metas):
        pytest.skip(
            "Export collapsed the scalar SymInt placeholder on this "
            "PyTorch version; nothing left to exercise for the mix case."
        )
    assert any(
        isinstance(m, torch.Tensor) for m in metas
    ), "expected at least one tensor placeholder"

    tensor_input = Input(
        min_shape=(1, 8),
        opt_shape=(4, 8),
        max_shape=(8, 8),
        dtype=torch.float32,
    )
    # The user is allowed to pass an ``Input`` for the SymInt argument
    # (Torch-TRT treats it as a 1-D shape tensor downstream), but
    # ``_build_user_symbol_bounds`` will not record it because
    # ``meta["val"]`` is a ``torch.SymInt``, not a ``torch.Tensor``.
    scalar_input = Input(
        min_shape=(1,),
        opt_shape=(4,),
        max_shape=(8,),
        dtype=torch.int64,
    )

    bounds = _build_user_symbol_bounds(ep.module(), [tensor_input, scalar_input], {})

    # Exactly one symbol recorded - from x's dynamic dim 0. The SymInt
    # placeholder contributed nothing. (The fact that the recorded
    # bounds equal the *tensor* input's (1, 8) - not the scalar input's
    # (1, 8), which happens to be the same numerically - is verified by
    # checking that exactly one entry exists, ruling out double-recording.)
    assert (
        len(bounds) == 1
    ), f"expected only the tensor placeholder to contribute, got {bounds}"
    ((sym, (lo, hi)),) = bounds.items()
    assert isinstance(sym, sympy.Symbol)
    assert (lo, hi) == (1, 8)


@pytest.mark.unit
def test_build_user_symbol_bounds_mixed_arg_and_kwarg():
    """Both positional and keyword inputs must contribute to the
    user-bounds map. Positional args are matched to placeholders by
    index; kwargs are matched by parameter name (``node.target``).
    """

    class _TwoTensors(torch.nn.Module):
        def forward(self, x, y):
            # Two independent paths so x's dim 0 and y's dim 0 are not
            # unified by the exporter into a single sympy symbol.
            return x.relu(), y.relu()

    model = _TwoTensors().eval().cuda()
    # Distinct sample sizes for dim 0 to discourage symbol unification.
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
    assert (
        "x" in placeholder_names
    ), f"expected positional arg 'x' as placeholder, got {placeholder_names}"
    assert (
        "y" in placeholder_names
    ), f"expected kwarg 'y' as placeholder, got {placeholder_names}"

    # Use *distinct* numerical bounds for x and y so we can verify by
    # value that the kwarg path actually contributed (rather than just
    # silently double-counting x's bounds).
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

    # Two placeholders, each with a single dynamic dim, no unification:
    # we expect exactly two entries with their respective bounds.
    bounds_values = set(bounds.values())
    assert (1, 8) in bounds_values, (
        f"positional x's bounds (1, 8) not recorded - "
        f"the positional-arg branch may be broken. got: {bounds}"
    )
    assert (2, 6) in bounds_values, (
        f"kwarg y's bounds (2, 6) not recorded - "
        f"the kwarg branch may be broken. got: {bounds}"
    )
    assert len(bounds) == 2, f"expected exactly 2 entries, got {bounds}"


@pytest.mark.unit
@pytest.mark.critical
def test_dim_dynamic_save_preserves_range_constraints(tmpdir):
    """End-to-end regression: exporting with ``Dim.DYNAMIC``, compiling
    with ``Input(max_shape=8)``, and round-tripping through
    ``torch_tensorrt.save(..., output_format="exported_program")`` must
    leave the exporter's ``range_constraints`` untouched.
    """

    model = _SmallLinear().eval().cuda()
    sample = torch.randn(2, 8, device="cuda")

    dyn_batch = torch.export.Dim.DYNAMIC
    exp_program = torch.export.export(
        model, (sample,), dynamic_shapes=({0: dyn_batch},), strict=False
    )

    # Snapshot range_constraints **before** Torch-TRT touches anything.
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

    # The exporter's range_constraints must NOT have been mutated by compile.
    after_constraints = {
        str(k): (v.lower, v.upper) for k, v in exp_program.range_constraints.items()
    }
    assertions.assertEqual(expected_constraints, after_constraints)

    # Save + reload as ExportedProgram and verify constraints survive.
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

    # Sanity: the compiled engine must accept inputs across the user's
    # declared profile - both at the lower edge (min_shape=1) and at the
    # upper edge (max_shape=8).
    trt_module(torch.randn(1, 8, device="cuda"))
    trt_module(torch.randn(4, 8, device="cuda"))
    trt_module(torch.randn(8, 8, device="cuda"))

    # And it must REJECT inputs beyond max_shape, even though the model
    # graph (exported with ``Dim.DYNAMIC``) is itself unbounded and could
    # theoretically handle batch=16 in eager. The TRT engine's profile is
    # the binding runtime envelope: ``Input(max_shape=8)`` is the user
    # opting into a strict cap. If a user wants batch=16 they must either
    # re-compile with ``max_shape>=16`` or omit ``max_shape`` (heuristic
    # fallback). This pins down the contract to prevent regressions where
    # ``Input.max_shape`` would silently widen back to the heuristic.
    too_big = torch.randn(16, 8, device="cuda")
    with assertions.assertRaises(Exception):
        trt_module(too_big)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
