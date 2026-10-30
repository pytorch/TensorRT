# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Adopt PyTorch's upstream complex decomposition (pytorch/pytorch#169832).

This is the "augment, not replace" path proposed in issue #4390.  Instead of the
hand-maintained per-op rewriter in ``complex_graph_rewrite.py``, we delegate the
interior complex -> real expansion to PyTorch's ``decompose_complex_in_graph``
(a tensor-subclass / SoA retrace) and keep only the TRT-specific glue:

  1. capture the complex I/O signature before rewriting (drives the boundary
     adapters in ``_compiler._insert_complex_io_adapters``);
  2. call upstream to expand every complex op into real ops on separate re/im;
     for dynamic inputs, use its wrapper with symbolic tracing to avoid
     specializing dimensions in tensor factories such as ``arange(seq)``;
  3. put back what the retrace undid: upstream returns a module rebuilt from
     scratch, so the decomposition and constant-folding stages that ran earlier
     find their work partly reverted;
  4. normalize the seams (``aten.view_as_real`` / ``aten.complex`` /
     ``aten.real`` / ``aten.imag``) into the interleaved ``[..., 2]`` layout the
     rest of the TRT flow expects, so the engine only ever sees real tensors
     (Option A in the RFC).

Gated by ``settings.use_complex_decomposition``; falls back to the legacy pass on
older torch (the upstream API only exists in torch>=2.14.0.dev), or if
``decompose_complex_in_graph`` itself raises (e.g. an op it doesn't support yet,
such as ``aten.log2``/``aten.log10`` as of torch 2.14.dev).
"""

import gc
import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional

import torch
from torch._export.utils import _detect_fake_mode_from_gm
from torch.fx import GraphModule
from torch.fx.experimental.symbolic_shapes import free_symbols
from torch_tensorrt._features import has_complex_decomposition
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.constant_folding import constant_fold
from torch_tensorrt.dynamo.lowering.passes.mark_constant_fold_exclusions import (
    mark_constant_fold_exclusions,
)
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)
from torch_tensorrt.dynamo.lowering.passes.remove_alias import remove_alias

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
    #     We stash it in LOCALS (not gm.meta) because decompose_complex_in_graph is
    #     functional -- it returns a NEW GraphModule via make_fx -- so anything we
    #     wrote on this instance's module-level .meta would not ride along.  We
    #     re-attach onto the returned module in step (4).  These three keys are the
    #     contract _insert_complex_io_adapters reads post-partition to reinsert the
    #     complex I/O boundary; losing them silently drops the boundary adapters and
    #     the model returns real [...,2] instead of complex.
    complex_output_indices = _get_complex_output_indices(gm)
    complex_input_names = _get_complex_input_names(gm)
    complex_input_dtypes = _get_complex_input_dtypes(gm)
    # _compiler.py reads _in_spec/_out_spec off this module post-lowering to
    # unflatten the final result back to the user's original return shape
    # (e.g. a bare tensor instead of a 1-tuple). decompose_complex_in_graph's
    # make_fx retrace returns a plain GraphModule that doesn't carry these
    # torch.export-specific attributes, so without re-attaching them below,
    # the unflatten step is silently skipped and callers get a raw tuple.
    original_in_spec = getattr(gm, "_in_spec", None)
    original_out_spec = getattr(gm, "_out_spec", None)
    original_codegen = gm.graph._codegen
    if not complex_input_names and not complex_output_indices:
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
    # The retrace calls the module with flat positional args. Under the exported
    # program's pytree codegen it would return the user's output structure, which
    # make_fx records as built by the graph (a namedtuple becomes one node); under
    # a plain codegen it returns the graph's flat outputs, which the original
    # codegen then unflattens again.
    gm.graph._codegen = torch.fx.graph.CodeGen()
    gm.recompile()
    decomposed_gm: Optional[GraphModule] = None
    try:
        # Both retraces return a new graph. Keep the original for the existing
        # unsupported-op fallback rather than replacing it before success.
        decompose = (
            _decompose_complex_symbolically
            if any(free_symbols(arg) for arg in flat_args)
            else decompose_complex_in_graph
        )
        with _offloaded_constants_on_device(gm) as offloaded:
            decomposed_gm = decompose(
                gm,
                flat_args,
                decompositions=_trt_decomposition_table(settings, gm),
            )
    except Exception as e:
        # decompose_complex_in_graph is upstream, experimental PyTorch code
        # (torch._functorch._aot_autograd.complex_decomposition) with
        # incomplete op coverage today -- e.g. ComplexTensor has no
        # __torch_dispatch__ for aten.log2/log10 as of torch 2.14.dev. Treat
        # any failure here as "this op set isn't supported yet" and degrade
        # to the legacy, well-tested rewriter for the whole graph rather
        # than hard-failing the compile.
        logger.warning(
            f"complex decomposition failed ({type(e).__name__}: {e}); "
            "falling back to the legacy complex_graph_detection pass."
        )
    gm.graph._codegen = original_codegen  # the caller's module, as it was
    gm.recompile()
    if decomposed_gm is None:
        # Not inside the except block: there the failed retrace's frames are still
        # alive, and with them its device copies of the offloaded constants.
        return complex_graph_detection(gm, settings)
    _restore_offloaded_constants(decomposed_gm, offloaded)
    if offloaded:
        # The retrace leaves the device copies in reference cycles. Free them now
        # rather than holding them through the engine build.
        gc.collect()
    gm = decomposed_gm

    # The retrace brings back alias nodes. aten.alias has no converter, so the
    # partitioner would split the graph around them.
    gm = remove_alias(gm, settings)

    # (3) Fold constants again. The retrace builds the module from scratch, so
    #     a constant that the constant-folding stage had already reduced to one
    #     frozen tensor comes back as a live chain of ops reading it, and the
    #     converters for those ops expect a TRT tensor rather than a weight.
    gm = mark_constant_fold_exclusions(gm, settings)
    gm = constant_fold(gm, settings)

    # (4) Normalize the seams into the interleaved [..., 2] layout used by TRT.
    gm = _normalize_complex_boundary_for_trt(gm)
    gm = clean_up_graph_after_modifications(gm)

    # (5) Re-attach the captured I/O signature onto the RETURNED module so
    #     _insert_complex_io_adapters can restore the complex boundary.
    gm.meta["complex_output_indices"] = complex_output_indices
    gm.meta["complex_input_names"] = complex_input_names
    gm.meta["complex_input_dtypes"] = complex_input_dtypes
    if original_in_spec is not None:
        gm._in_spec = original_in_spec
    if original_out_spec is not None:
        gm._out_spec = original_out_spec
    # The retrace returns a plain codegen (positional flat args); restore the
    # exported program's, which takes the inputs by name
    gm.graph._codegen = original_codegen
    gm.recompile()
    return gm


def _decompose_complex_symbolically(
    gm: GraphModule, flat_args: list[Any], decompositions: dict[Any, Any]
) -> GraphModule:
    """Use upstream's complex dispatch with a symbolic make_fx retrace.

    Upstream's entry point hardcodes real tracing, which specializes symbols in
    factories such as arange(seq). Until it accepts a tracing mode, mirror its
    wrapper here while keeping its dispatch and mutation checks. Sending dynamic
    graphs to the legacy rewriter instead regresses complex operator correctness.
    """
    from torch._functorch._aot_autograd import complex_decomposition as upstream
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch.utils import _pytree as pytree

    upstream._assert_no_incorrect_aliasing_mutation(gm)

    def wrapper(*args: Any) -> Any:
        wrapped = tuple(upstream._maybe_wrap(arg) for arg in args)
        with upstream.WrapComplexMode():
            result = torch.fx.Interpreter(gm).run(*wrapped)
        return pytree.tree_map(upstream._maybe_unwrap, result)

    with torch.fx.traceback.preserve_node_meta():
        return make_fx(
            wrapper, decomposition_table=decompositions, tracing_mode="symbolic"
        )(*flat_args)


def _trt_decomposition_table(
    settings: CompilationSettings, graph_module: GraphModule
) -> dict[Any, Any]:
    """The op set the rest of the TRT flow expects to see.

    ``decompose_complex_in_graph`` re-traces through ``make_fx``, so the module
    it returns is built from whatever ``ComplexTensor`` dispatched to, not from
    the ops that survived the decomposition run at export time. Without a table
    that retrace re-introduces ops the flow has already decided to expand
    (``aten.stack``), long after the stage that would have handled them, and
    the partitioner then cuts the graph around them. Handing the retrace the
    same table keeps the two in step.
    """
    from torch_tensorrt.dynamo.lowering import (
        filter_decomposition_table,
        get_decompositions,
    )

    decompositions: dict[Any, Any] = filter_decomposition_table(
        get_decompositions(
            settings.enable_experimental_decompositions,
            settings.decompose_attention,
            settings.use_distributed_mode_trace,
            use_fp32_acc=settings.use_fp32_acc,
        ),
        graph_module,
    )
    return decompositions


def _graph_has_complex(gm: GraphModule) -> bool:
    from torch_tensorrt.dynamo.utils import COMPLEX_DTYPES

    for node in gm.graph.nodes:
        val = node.meta.get("val", None)
        if val is not None and getattr(val, "dtype", None) in COMPLEX_DTYPES:
            return True
    return False


@contextmanager
def _offloaded_constants_on_device(
    gm: GraphModule,
) -> Iterator[dict[int, torch.Tensor]]:
    """Hand the retrace each CPU-offloaded constant on the device it was computed on.

    With offload_module_to_cpu, constant folding keeps the folded values on the CPU
    while their metadata still records the device that computed them. The retrace
    runs the graph, so a CPU constant next to device activations fails with a device
    mismatch. Copy each such constant to its metadata device for the retrace only;
    yields a map from the id of each copy to its CPU tensor, for
    _restore_offloaded_constants.
    """
    swapped = {}
    for node in gm.graph.nodes:
        target = str(node.target)
        if node.op != "get_attr" or not hasattr(gm, target):
            continue
        real, val = getattr(gm, target), node.meta.get("val", None)
        if (
            isinstance(real, torch.Tensor)
            and isinstance(val, torch.Tensor)
            and real.device != val.device
        ):
            swapped[target] = (real, real.to(val.device))
    for target, (_, copy) in swapped.items():
        delattr(gm, target)
        setattr(gm, target, copy)
    try:
        yield {id(copy): real for real, copy in swapped.values()}
    finally:
        for target, (real, _) in swapped.items():
            delattr(gm, target)
            setattr(gm, target, real)


def _restore_offloaded_constants(
    gm: GraphModule, offloaded: dict[int, torch.Tensor]
) -> None:
    """Put the CPU tensors back wherever the retraced module kept a device copy."""
    for node in gm.graph.nodes:
        if node.op != "get_attr":
            continue
        real = offloaded.get(id(getattr(gm, str(node.target), None)))
        if real is not None:
            setattr(gm, str(node.target), real)


def _fake_flat_args(gm: GraphModule) -> list[torch.Tensor]:
    """Build the example inputs the upstream retrace (make_fx) needs.

    node.meta["val"] is always preferred: it's the live FakeTensor torch.export's
    fake-tensor tracing sets on every node, carrying the real ShapeEnv/SymInt
    objects, so reusing it (rather than rebuilding a tensor from its shape)
    is what makes dynamic shapes survive the retrace (see the dynamic-shape
    cases in test_complex_ops.py and test_rope_embedding.py).

    node.meta["tensor_meta"] is only consulted as a fallback, for a
    placeholder that somehow lacks "val" (e.g. a graph shape-propagated via
    torch.fx.passes.shape_prop.ShapeProp instead of torch.export). That path
    only ever records concrete shapes, so a symbolic dim reaching here means
    the graph wasn't produced the way this function expects; we refuse to
    synthesize a placeholder concrete dimension for it, since a silently
    wrong shape could retrace successfully but produce numerically wrong
    results, which is worse than failing loudly.
    """
    args = []
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        val = node.meta.get("val", None)
        if val is None:
            tm = node.meta["tensor_meta"]
            if any(isinstance(dim, torch.SymInt) for dim in tm.shape):
                raise RuntimeError(
                    f"complex_decomposition_adapter: placeholder '{node.name}' has "
                    "no node.meta['val'] and its tensor_meta.shape contains a "
                    "symbolic dimension, so a concrete example tensor can't be "
                    "synthesized for decompose_complex_in_graph. This means the "
                    "graph wasn't produced via torch.export's fake-tensor tracing "
                    "(which always sets 'val') -- pass a graph traced that way "
                    "instead of one shape-propagated via ShapeProp."
                )
            val = torch.empty(tm.shape, dtype=tm.dtype, device=tm.device)
        args.append(val)
    return args


def _normalize_complex_boundary_for_trt(gm: GraphModule) -> GraphModule:
    """Fold upstream's seams into the interleaved ``[..., 2]`` real layout.

    Every op below reinterprets a complex value as a pair of real halves, or
    the other way round, and TRT has no converter for any of them.  Leaving one
    in the graph does not just fail an op check: the partitioner cuts the graph
    around it and the model comes back split into several engines with PyTorch
    blocks in between.  We rewrite:

      * ``view_as_real(z)``  -> the interleaved ``[..., 2]`` tensor it would
        have returned, for each producer ``z`` can have (see Pass A)
      * ``real(z)`` / ``imag(z)`` where ``z = complex(re, im)``  -> re / im
        (cancel the pack/unpack round-trip)
      * remaining ``aten.complex(re, im)``  -> ``cat([re, im], -1)`` tagged as
        complex-layout, so the [..., 2] tensor flows to the boundary adapters.

    Which of these appear depends on where the complex region sits.  A graph
    whose own inputs or outputs are complex meets upstream at ``aten.complex``,
    because that is what ``ComplexTensor`` packs with on the way out.  A graph
    that is real on both ends but complex in the middle, as a rotary embedding
    is, meets it at ``view_as_real`` instead: the entry ``view_as_complex`` is
    left alone (its argument is a plain real tensor, so ``ComplexTensor`` never
    sees it), and ``ComplexTensor`` splits its halves with ``torch.real`` /
    ``torch.imag``, which are composite ops that expand to ``view_as_real``
    plus ``select`` before any subclass can intercept them.
    """
    g = gm.graph
    aten = torch.ops.aten
    fake_mode = _detect_fake_mode_from_gm(gm)

    def _to_fake(arg: Any) -> Any:
        if isinstance(arg, torch.fx.Node):
            return arg.meta["val"]
        if isinstance(arg, (list, tuple)):
            return [_to_fake(a) for a in arg]
        return arg

    def _propagate_meta(node: torch.fx.Node) -> None:
        # Newly inserted nodes (unsqueeze/cat below) have no "val" until we
        # compute one -- downstream stages (dryrun metadata checks, symbolic
        # shape extraction) require every node to carry one.
        if fake_mode is None:
            return
        arg_vals = [_to_fake(arg) for arg in node.args]
        with fake_mode:
            node.meta["val"] = node.target(*arg_vals)

    def _pack_interleaved(re: Any, im: Any) -> torch.fx.Node:
        """Build the ``[..., 2]`` tensor holding ``re`` and ``im`` side by side.

        Callers set the insertion point.  ``cat`` of two unsqueezed halves
        rather than ``stack`` because the stage that expands ``stack`` has
        already run by the time this pass does.
        """
        re_u = g.call_function(aten.unsqueeze.default, (re, -1))
        _propagate_meta(re_u)
        im_u = g.call_function(aten.unsqueeze.default, (im, -1))
        _propagate_meta(im_u)
        packed = g.call_function(aten.cat.default, ([re_u, im_u], -1))
        _propagate_meta(packed)
        return packed

    # Pass A: rewrite view_as_real over each producer it can have.
    for node in list(g.nodes):
        if node.op != "call_function" or node.target != aten.view_as_real.default:
            continue
        src = node.args[0]
        if not isinstance(src, torch.fx.Node):
            continue
        if src.target == aten.view_as_complex.default:
            # view_as_complex consumes a trailing dimension of 2 and
            # view_as_real puts it back holding the same values, so the pair is
            # an identity on the real tensor that went in.  Dropping it leaves
            # the halves to be picked out of that tensor directly.
            replacement = src.args[0]
        elif src.target == aten.complex.default:
            with g.inserting_before(node):
                replacement = _pack_interleaved(src.args[0], src.args[1])
        else:
            continue
        node.replace_all_uses_with(replacement)
        g.erase_node(node)

    # Pass B: cancel real(complex(re,im)) / imag(complex(re,im)) round-trips.
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

    # Pass C: turn surviving aten.complex(re, im) into the interleaved layout.
    for node in list(g.nodes):
        if node.op != "call_function" or node.target != aten.complex.default:
            continue
        with g.inserting_before(node):
            packed = _pack_interleaved(node.args[0], node.args[1])
        packed.meta["is_complex_layout"] = True
        node.replace_all_uses_with(packed)
        g.erase_node(node)

    g.lint()
    gm.recompile()
    return gm
