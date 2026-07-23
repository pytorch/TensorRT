"""Adopt PyTorch's upstream complex decomposition (pytorch/pytorch#169832).

This is the "augment, not replace" path proposed in issue #4390.  Instead of the
hand-maintained per-op rewriter in ``complex_graph_rewrite.py``, we delegate the
interior complex -> real expansion to PyTorch's ``decompose_complex_in_graph``
(a tensor-subclass / SoA retrace) and keep only the TRT-specific glue:

  1. capture the complex I/O signature before rewriting (drives the boundary
     adapters in ``_compiler._insert_complex_io_adapters``);
  2. call upstream to expand every complex op into real ops on separate re/im;
  3. normalize the SoA seams (``aten.complex`` / ``aten.real`` / ``aten.imag``)
     into the interleaved ``[..., 2]`` layout the rest of the TRT flow expects,
     so the engine only ever sees real tensors (Option A in the RFC).

Gated by ``settings.use_complex_decomposition``; falls back to the legacy pass on
older torch (the upstream API only exists in torch>=2.14.0.dev).

NOTE: the exact shape of what ``decompose_complex_in_graph`` emits at the
boundary must be confirmed by a spike (torch is required to run it).  The steps
marked ``TODO(spike)`` encode assumptions to validate against real output.
"""

import logging

import torch
from torch.fx import GraphModule
from torch_tensorrt._features import has_complex_decomposition
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

# Reuse the I/O-signature capture helpers already written for the legacy pass.
from .complex_graph_rewrite import (
    _get_complex_input_dtypes,
    _get_complex_input_names,
    _get_complex_output_indices,
    complex_graph_detection,
)

logger = logging.getLogger(__name__)


def complex_decomposition_adapter(
    gm: GraphModule, settings: CompilationSettings
) -> GraphModule:
    """Delegate complex decomposition to upstream, keep TRT boundary glue ours."""
    if not has_complex_decomposition():
        logger.warning(
            "complex decomposition requires torch>=2.14.dev; falling back to the "
            "legacy complex_graph_detection pass."
        )
        return complex_graph_detection(gm, settings)

    # (1) Capture the complex I/O signature BEFORE any rewrite mutates the graph.
    #     Identical contract to the legacy pass so _insert_complex_io_adapters
    #     keeps working unchanged.
    gm.meta["complex_output_indices"] = _get_complex_output_indices(gm)
    gm.meta["complex_input_names"] = _get_complex_input_names(gm)
    gm.meta["complex_input_dtypes"] = _get_complex_input_dtypes(gm)
    if not gm.meta["complex_input_names"] and not gm.meta["complex_output_indices"]:
        # No complex I/O and no interior complex work -> nothing to do.  (Interior-
        # only complex still shows up via inputs/outputs of the complex region, so
        # this early-out matches the legacy pass's behavior.)
        if not _graph_has_complex(gm):
            return gm

    # (2) Hand the interior expansion to upstream.
    from torch._functorch._aot_autograd.complex_decomposition import (
        decompose_complex_in_graph,
    )

    flat_args = _fake_flat_args(gm)
    logger.debug("complex_decomposition_adapter: retracing under ComplexTensor")
    gm = decompose_complex_in_graph(gm, flat_args)

    # (3) Normalize SoA seams into the interleaved [..., 2] layout used by TRT.
    gm = _normalize_complex_boundary_for_trt(gm)
    gm = clean_up_graph_after_modifications(gm)
    return gm


def _graph_has_complex(gm: GraphModule) -> bool:
    from torch_tensorrt.dynamo.utils import COMPLEX_DTYPES

    for node in gm.graph.nodes:
        val = node.meta.get("val", None)
        if val is not None and getattr(val, "dtype", None) in COMPLEX_DTYPES:
            return True
    return False


def _fake_flat_args(gm: GraphModule) -> list:
    """Build the example inputs the upstream retrace (make_fx) needs.

    We reuse the placeholders' existing fake tensors so the export ShapeEnv /
    SymInt ranges are preserved through the retrace.  Whether dynamic shapes
    actually survive make_fx is RFC open-question #1 -- assert-worthy in the spike.
    """
    args = []
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        val = node.meta.get("val", None)
        if val is None:
            tm = node.meta["tensor_meta"]
            val = torch.empty(tm.shape, dtype=tm.dtype, device=tm.device)
        args.append(val)
    return args


def _normalize_complex_boundary_for_trt(gm: GraphModule) -> GraphModule:
    """Fold upstream's SoA seams into the interleaved ``[..., 2]`` real layout.

    Upstream leaves the graph in terms of separate re/im joined by
    ``aten.complex(re, im)`` and unpacked by ``aten.real`` / ``aten.imag``.  TRT
    has no converter for any of those.  We:

      * ``real(z)`` / ``imag(z)`` where ``z = complex(re, im)``  -> re / im
        (cancel the pack/unpack round-trip)
      * remaining ``aten.complex(re, im)``  -> ``stack([re, im], -1)`` tagged as
        complex-layout, so the [..., 2] tensor flows to the boundary adapters.

    TODO(spike): confirm against real decompose_complex_in_graph output --
    the op set at the boundary (aten.complex vs view_as_complex, real vs
    select) and whether inputs are unpacked with real/imag or view_as_real.
    """
    g = gm.graph
    aten = torch.ops.aten

    # Pass A: cancel real(complex(re,im)) / imag(complex(re,im)) round-trips.
    for node in list(g.nodes):
        if node.op != "call_function" or node.target not in (
            aten.real.default,
            aten.imag.default,
        ):
            continue
        # node is  real(z) / imag(z);  node.args[0] is that single arg z -- the
        # tensor being unpacked (the "source" producer feeding this unpack).
        src = node.args[0]
        # Only fold when z was itself produced by aten.complex(re, im): then
        # real(complex(re,im)) == re and imag(complex(re,im)) == im, so the
        # pack/unpack round-trip is pure noise.  (isinstance guards against a
        # non-Node arg, e.g. a constant, which has no .target.)
        if isinstance(src, torch.fx.Node) and src.target == aten.complex.default:
            # aten.complex stores re at args[0], im at args[1] -> pick the half
            # this unpack wanted: real -> 0, imag -> 1.
            idx = 0 if node.target == aten.real.default else 1
            # Rewire every consumer of real(z)/imag(z) straight to re/im, then
            # delete the now-orphaned real()/imag() node.
            node.replace_all_uses_with(src.args[idx])
            g.erase_node(node)

    # Pass B: turn surviving aten.complex(re, im) into stack([re, im], -1).
    for node in list(g.nodes):
        if node.op != "call_function" or node.target != aten.complex.default:
            continue
        re, im = node.args[0], node.args[1]
        with g.inserting_before(node):
            re_u = g.call_function(aten.unsqueeze.default, (re, -1))
            im_u = g.call_function(aten.unsqueeze.default, (im, -1))
            packed = g.call_function(aten.cat.default, ([re_u, im_u], -1))
        packed.meta["is_complex_layout"] = True
        node.replace_all_uses_with(packed)
        g.erase_node(node)

    g.lint()
    gm.recompile()
    return gm
