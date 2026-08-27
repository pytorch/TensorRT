"""Unit tests for the complex_decomposition_adapter lowering pass (issue #4390).

These tests exercise the TRT-specific glue around PyTorch's upstream complex
decomposition -- they do NOT require a GPU or a TRT build:

  * _normalize_complex_boundary_for_trt: folds the aten.view_as_real /
    aten.complex / real / imag seams that decompose_complex_in_graph leaves
    behind into the interleaved [..., 2] real layout the rest of the TRT flow
    expects.
  * the torch-version feature gate: when the upstream API is unavailable the
    adapter must fall back to the legacy complex_graph_detection pass.
"""

import pytest
import torch
from torch_tensorrt.dynamo.lowering.passes import complex_decomposition_adapter as cda

aten = torch.ops.aten


def _targets(gm):
    return [n.target for n in gm.graph.nodes if n.op == "call_function"]


# ---------------------------------------------------------------------------
# _normalize_complex_boundary_for_trt
# ---------------------------------------------------------------------------


def test_normalize_folds_real_imag_of_complex():
    """real(complex(re,im)) -> re ;  imag(complex(re,im)) -> im, and the
    aten.complex / real / imag nodes are erased."""

    class M(torch.nn.Module):
        def forward(self, re, im):
            z = torch.ops.aten.complex.default(re, im)
            return torch.ops.aten.real.default(z), torch.ops.aten.imag.default(z)

    re = torch.randn(4)
    im = torch.randn(4)
    gm = torch.fx.symbolic_trace(M())

    gm = cda._normalize_complex_boundary_for_trt(gm)
    targets = _targets(gm)

    assert aten.real.default not in targets
    assert aten.imag.default not in targets
    assert aten.complex.default not in targets  # unused after folding -> DCE'd
    # outputs should be re, im directly
    out_re, out_im = gm(re, im)
    assert torch.equal(out_re, re)
    assert torch.equal(out_im, im)


def test_normalize_packs_surviving_complex_to_interleaved():
    """A complex(re,im) whose result is actually used becomes
    cat([unsqueeze(re,-1), unsqueeze(im,-1)], -1) tagged is_complex_layout."""

    class M(torch.nn.Module):
        def forward(self, re, im):
            return torch.ops.aten.complex.default(re, im)

    re = torch.randn(3)
    im = torch.randn(3)
    gm = torch.fx.symbolic_trace(M())

    gm = cda._normalize_complex_boundary_for_trt(gm)
    targets = _targets(gm)

    assert aten.complex.default not in targets
    assert aten.cat.default in targets
    assert aten.unsqueeze.default in targets

    # the packed cat node must carry the complex-layout tag
    cat_nodes = [n for n in gm.graph.nodes if n.target == aten.cat.default]
    assert cat_nodes and cat_nodes[0].meta.get("is_complex_layout") is True

    # numerically: output is the [...,2] interleaved layout of re/im
    out = gm(re, im)
    assert out.shape == (3, 2)
    assert torch.equal(out[..., 0], re)
    assert torch.equal(out[..., 1], im)


def test_normalize_folds_view_as_real_of_view_as_complex():
    """view_as_real(view_as_complex(x)) -> x, and both nodes are erased."""

    class M(torch.nn.Module):
        def forward(self, x):
            z = torch.ops.aten.view_as_complex.default(x)
            return torch.ops.aten.view_as_real.default(z)

    from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
        clean_up_graph_after_modifications,
    )

    x = torch.randn(3, 2)
    gm = torch.fx.symbolic_trace(M())

    gm = cda._normalize_complex_boundary_for_trt(gm)
    # The view_as_complex node is left without users, and the adapter runs this
    # right after the pass to collect nodes in exactly that state.
    gm = clean_up_graph_after_modifications(gm)
    targets = _targets(gm)

    assert aten.view_as_real.default not in targets
    assert aten.view_as_complex.default not in targets
    assert torch.equal(gm(x), x)


def test_normalize_folds_view_as_real_of_complex():
    """view_as_real(complex(re,im)) -> the interleaved [..., 2] layout.

    Without this, Pass C would rewrite the complex() node into that layout and
    leave a view_as_real reading it, which is no longer a complex tensor.
    """

    class M(torch.nn.Module):
        def forward(self, re, im):
            z = torch.ops.aten.complex.default(re, im)
            return torch.ops.aten.view_as_real.default(z)

    re = torch.randn(3)
    im = torch.randn(3)
    gm = torch.fx.symbolic_trace(M())

    gm = cda._normalize_complex_boundary_for_trt(gm)
    targets = _targets(gm)

    assert aten.view_as_real.default not in targets
    assert aten.complex.default not in targets

    out = gm(re, im)
    assert out.shape == (3, 2)
    assert torch.equal(out[..., 0], re)
    assert torch.equal(out[..., 1], im)


# ---------------------------------------------------------------------------
# whole pass over a graph that is real at both ends and complex in the middle
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not cda.has_complex_decomposition(),
    reason="decompose_complex_in_graph requires torch>=2.14.dev",
)
def test_interior_complex_region_leaves_nothing_trt_cannot_convert():
    """A rotary-embedding shaped graph must come out with real ops only.

    The complex region here sits in the middle: the graph takes a real tensor
    and returns one, so upstream meets it at view_as_real rather than at
    aten.complex, and its retrace also brings back ops the decomposition and
    constant-folding stages had already dealt with. Any of those left behind
    is not just an unconverted op: the partitioner cuts the graph around it and
    the model comes back as several engines with PyTorch blocks in between.
    """
    from torch_tensorrt.dynamo._settings import CompilationSettings
    from torch_tensorrt.dynamo.lowering import get_decompositions

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "freqs",
                torch.complex(torch.randn(1, 2, 1, 1), torch.randn(1, 2, 1, 1)),
            )

        def forward(self, x):
            z = torch.view_as_complex(x.reshape(1, 2, 1, 1, 2))
            return torch.view_as_real(z * self.freqs)

    model = M().eval()
    x = torch.randn(1, 2, 1, 2)
    expected = model(x)

    gm = (
        torch.export.export(model, (x,))
        .run_decompositions(get_decompositions())
        .module()
    )
    out = cda.complex_decomposition_adapter(gm, CompilationSettings())

    targets = set(_targets(out))
    for target in (
        aten.view_as_complex.default,
        aten.view_as_real.default,
        aten.complex.default,
        aten.alias.default,
        aten.stack.default,
    ):
        assert target not in targets, f"{target} survived the pass"

    torch.testing.assert_close(out(x), expected)


# ---------------------------------------------------------------------------
# feature gate / fallback
# ---------------------------------------------------------------------------


def test_gate_falls_back_to_legacy_on_old_torch(monkeypatch):
    """When has_complex_decomposition() is False, the adapter must delegate to
    the legacy complex_graph_detection pass (and NOT import the upstream API)."""
    calls = {}

    def fake_legacy(gm, settings):
        calls["legacy"] = True
        return gm

    monkeypatch.setattr(cda, "has_complex_decomposition", lambda: False)
    monkeypatch.setattr(cda, "complex_graph_detection", fake_legacy)

    gm = torch.fx.symbolic_trace(torch.nn.Identity())
    out = cda.complex_decomposition_adapter(gm, settings=object())

    assert calls.get("legacy") is True
    assert out is gm


@pytest.mark.skipif(
    not cda.has_complex_decomposition(),
    reason="decompose_complex_in_graph requires torch>=2.14.dev",
)
def test_upstream_api_importable():
    """Smoke check: on a supported torch, the upstream entry point is importable
    at the path the adapter uses."""
    from torch._functorch._aot_autograd.complex_decomposition import (  # noqa: F401
        decompose_complex_in_graph,
    )


def test_default_tracks_has_complex_decomposition():
    """use_complex_decomposition's default tracks has_complex_decomposition()
    rather than a fixed value, so this new path is auto-enabled on
    torch>=2.14.dev instead of being opt-in (see PR description). Lock this
    in so an accidental hardcoded default doesn't silently change that."""
    from torch_tensorrt.dynamo import _defaults
    from torch_tensorrt.dynamo._settings import CompilationSettings

    assert _defaults.USE_COMPLEX_DECOMPOSITION == cda.has_complex_decomposition()
    assert (
        CompilationSettings().use_complex_decomposition
        == cda.has_complex_decomposition()
    )


# ---------------------------------------------------------------------------
# I/O-signature metadata survival across the functional decompose call (#7)
# ---------------------------------------------------------------------------


def test_io_metadata_reattached_to_returned_module(monkeypatch):
    """decompose_complex_in_graph returns a NEW GraphModule, so the complex I/O
    signature captured pre-decompose must be re-attached onto the returned module
    -- otherwise _insert_complex_io_adapters reads empty meta and drops the
    complex boundary. We stub the upstream call to return a fresh module (empty
    meta) and assert the adapter carries the signature across."""
    import sys
    import types

    class M(torch.nn.Module):
        def forward(self, x):
            return x

    gm = torch.fx.symbolic_trace(M())
    # Mark the placeholder/output as complex so the capture returns index [0].
    for n in gm.graph.nodes:
        if n.op == "placeholder":
            n.meta["val"] = torch.zeros(4, dtype=torch.complex64)

    # A DIFFERENT module object that the stubbed decompose "returns" (functional
    # semantics), with a fresh empty .meta -- exactly the #7 hazard.
    new_gm = torch.fx.symbolic_trace(M())

    def fake_decompose(g, flat_args, *a, **k):
        return new_gm

    fake_mod = types.ModuleType("torch._functorch._aot_autograd.complex_decomposition")
    fake_mod.decompose_complex_in_graph = fake_decompose
    monkeypatch.setitem(
        sys.modules,
        "torch._functorch._aot_autograd.complex_decomposition",
        fake_mod,
    )
    monkeypatch.setattr(cda, "has_complex_decomposition", lambda: True)

    # Real settings, not a stub: the adapter reads the decomposition options off
    # them to build the table it hands to the retrace.
    from torch_tensorrt.dynamo._settings import CompilationSettings

    out = cda.complex_decomposition_adapter(gm, settings=CompilationSettings())

    # Functional: we did NOT get the original module back...
    assert out is not gm
    # ...and the captured signature rode onto the returned module.
    assert out.meta.get("complex_output_indices") == [0]
    assert out.meta.get("complex_input_names") == ["x"]


# ---------------------------------------------------------------------------
# _fake_flat_args: prefer the live fake (ShapeEnv-carrying) over tensor_meta (#3)
# ---------------------------------------------------------------------------


def test_fake_flat_args_prefers_val_over_tensor_meta():
    """The retrace must receive the placeholders' existing fake tensors unchanged
    (so the export ShapeEnv/SymInts survive). Only when 'val' is absent do we
    rebuild from tensor_meta (a lossy, static fallback)."""

    class M(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    gm = torch.fx.symbolic_trace(M())
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]

    live_fake = torch.zeros(2, 3, dtype=torch.float32)
    placeholders[0].meta["val"] = live_fake

    class _TM:  # minimal tensor_meta stand-in (shape/dtype/device only)
        shape = (4,)
        dtype = torch.float32
        device = torch.device("cpu")

    placeholders[1].meta.pop("val", None)
    placeholders[1].meta["tensor_meta"] = _TM()

    args = cda._fake_flat_args(gm)

    # The live fake is passed through by identity (ShapeEnv preserved untouched).
    assert args[0] is live_fake
    # The val-less placeholder is rebuilt from tensor_meta (static fallback).
    assert tuple(args[1].shape) == (4,)


def test_fake_flat_args_raises_on_symbolic_tensor_meta_without_val():
    """If a placeholder has no "val" and its tensor_meta.shape contains a
    symbolic dim, we can't synthesize a concrete example tensor safely --
    fail loudly instead of silently guessing a wrong concrete shape."""

    class M(torch.nn.Module):
        def forward(self, x):
            return x + 1

    x = torch.randn(4)
    ep = torch.export.export(
        M(), (x,), dynamic_shapes={"x": {0: torch.export.Dim("n", min=1, max=16)}}
    )
    gm = ep.module()
    placeholder = next(n for n in gm.graph.nodes if n.op == "placeholder")
    sym_dim = placeholder.meta["val"].shape[0]  # a real SymInt, from real export

    class _TM:  # tensor_meta stand-in with a symbolic dim
        shape = (sym_dim,)
        dtype = torch.float32
        device = torch.device("cpu")

    placeholder.meta.pop("val", None)
    placeholder.meta["tensor_meta"] = _TM()

    with pytest.raises(RuntimeError, match="symbolic dimension"):
        cda._fake_flat_args(gm)
