# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Lift mutated module buffers back to engine input bindings.

PyTorch's ``ExportedProgram.module()`` converts BUFFER placeholders into
``get_attr`` nodes that read the buffer from the GraphModule's state, plus a
trailing ``aten.copy_(get_attr_buffer, new_value)`` per BUFFER_MUTATION
output. From Torch-TensorRT's point of view those ``get_attr`` nodes are
parameters that get constant-folded; the buffer becomes baked into the engine
and the trailing ``copy_`` is dropped. Per-call buffer state is lost and the
KV-cache aliasing path cannot fire (the cache isn't a network input).

This module provides:

* :func:`lift_mutated_buffers` — pre-compile rewrite that turns each mutated
  buffer's ``get_attr`` into a ``placeholder`` and removes the trailing
  ``copy_``. The buffer becomes an engine input binding; downstream the
  slice_scatter converter's KV-cache fast path can recognize the cache as a
  network input and emit ``IKVCacheUpdateLayer`` with aliased I/O.

* :func:`inline_lifted_buffers_into_gm` — post-compile transform that
  registers each lifted buffer as state on the compiled GraphModule and
  rewrites the corresponding placeholder nodes to ``get_attr`` reads. The
  resulting module's ``forward`` takes only user inputs (buffers are
  threaded internally via the fx graph). Because everything is fx +
  module state, the result serializes naturally through
  ``torch_tensorrt.save`` / ``torch.export``.

What copy-back costs: only an engine-aliased (KV) write is free. A write routed to
copy-back re-attaches the *whole post-write buffer* as a graph output, because a
BUFFER_MUTATION output is defined as the buffer's new contents rather than the slice
that changed. Each call therefore pays, per copy-back buffer:

* the graph materializing the full buffer as that output — whatever applied the
  update produces the whole tensor rather than the changed slice, whether that is a
  scatter inside an engine or an op the partitioner left in PyTorch — and
* ExecuTorch's write-back pass copying that output into the caller-owned buffer
  after the delegate returns.

For a decode step writing one position into a multi-megabyte KV cache, that is a
full-cache-sized materialization plus a full-cache-sized copy to record a
single-slot update, per buffer, per call, and easily dominates the step. It is the
right correctness tradeoff — the alternative is silently losing the write — but it
is why the classifier routes a cache write to engine aliasing instead, and why a
model that unexpectedly lands in copy-back comes out slow rather than wrong.
``gm.meta['_copyback_mutation_buffers']`` lists which buffers are paying it.

Where the copy happens depends on who runs the module. A serialized program declares
the re-attached value a BUFFER_MUTATION, which the ExecuTorch runtime applies after
the delegate returns. A compiled module called directly from PyTorch applies it itself
at the end of ``forward`` (:func:`hide_copyback_outputs`), and does not return the
value. It returns the model's own outputs as a flat tuple, so a model with one
output gives a 1-tuple. So does any module whose buffers were lifted, with or
without copy-back.
"""

from __future__ import annotations

import contextlib
import logging
import operator
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings

logger = logging.getLogger(__name__)


def _write_op_is_torch_executed(
    value_node: object,
    settings: Optional[CompilationSettings] = None,
) -> bool:
    """Whether this write's op is in ``settings.torch_executed_ops``.

    Matched the way the partitioners match it, so the two agree about the op. This is
    only one of the reasons a node can end up in PyTorch (a missing converter, a failed
    capability validator, ``min_block_size`` and others do the same), so it is a
    sufficient reason to rule engine aliasing out, never a proof that aliasing happens.
    ``assert_predicted_kv_aliased`` remains the ground-truth check for the rest.
    """
    if settings is None or not settings.torch_executed_ops:
        return False
    if not (isinstance(value_node, torch.fx.Node) and value_node.op == "call_function"):
        return False
    from torch_tensorrt.dynamo.conversion._ConverterRegistry import ConverterRegistry

    excluded = set(settings.torch_executed_ops)
    target = value_node.target
    return (
        ConverterRegistry.qualified_name_or_str(target) in excluded
        or target in excluded
    )


def _effective_cache_input(cache_node: torch.fx.Node) -> Optional[torch.fx.Node]:
    """The node the converter will see as this write's cache argument, if it is a
    network input, else ``None``.

    A lifted buffer is a placeholder, and ``emit_kv_cache_update_layer`` aliases the
    cache only when the argument it is handed is one -- anything else has no input
    binding name and the converter falls back to a non-aliasing concatenation or
    scatter.

    One rewrite stands between this classification and the converter:
    ``remove_input_alias_fixing_clones``, an ATEN lowering pass inside
    ``post_lowering``, erases a ``clone`` whose placeholder has no other user, which
    is the shape ``clone -> scatter -> copy_`` leaves behind once lifting has erased
    the ``copy_``. The converter then does see a direct network input and does emit
    the KV layer, so this has to predict what the converter will be handed rather
    than what is in the graph now.

    This follows the pass's condition for a buffer that is not filed copy-back, sole
    user and all: a placeholder with another reader keeps its clone, the converter
    sees a ``call_function``, and nothing aliases. The pass also keeps the clone of a
    buffer that lifting files copy-back, so a write this does not predict KV keeps its
    clone and is not aliased either.

    Predicting the peel is not free. A cloned cache write is classified KV, so if no
    engine goes on to claim it -- at the *default* ``min_block_size=5`` a small
    model's subgraph often does not -- ``assert_predicted_kv_aliased`` fails the
    compile rather than leaving the buffer at its compile-time value. A loud failure
    beats a silently stale buffer, and this failure is reachable at the default
    ``min_block_size``, not only at ``min_block_size=1``.
    """
    if cache_node.op == "placeholder":
        return cache_node
    if not (
        cache_node.op == "call_function"
        and cache_node.target is torch.ops.aten.clone.default
        and cache_node.args
        and isinstance(cache_node.args[0], torch.fx.Node)
        and cache_node.args[0].op == "placeholder"
    ):
        return None
    placeholder = cache_node.args[0]
    users = list(placeholder.users)
    if len(users) == 1 and users[0] is cache_node:
        return placeholder
    return None


def _kv_write_will_alias(
    value_node: object,
    cache_shape: Tuple[int, ...],
    settings: Optional[CompilationSettings] = None,
) -> bool:
    """Whether the converter will emit an ``IKVCacheUpdateLayer`` (with in-place
    aliased I/O) for this mutated buffer's new-value node.

    Eligibility depends on the write dim, static shapes, and the cache being a
    direct network input, so this reuses the converters' own eligibility
    predicates -- and, for ``slice_scatter``, the converter's own derivation of
    the arguments those predicates take (:func:`resolve_slice_scatter_write`), since
    a divergence there mispredicts just as effectively as a divergence in the
    predicate. Returning ``False`` routes the write to copy-back, which is right
    whatever the converter goes on to do with it: it lowers most ineligible writes
    to a non-aliasing concatenation or scatter, returns the source in the input's
    dtype for a full overwrite, and raises for the rest (a ``slice_scatter`` with
    dynamic bounds or a bad ``dim``, an ``index_copy`` its fallback cannot express).
    The first two both have a write-back to preserve -- for a full overwrite the
    source *is* the buffer's new contents -- and the ones that raise never get as far
    as needing one. One more never reaches a converter: a ``slice_scatter`` written
    on a dynamic dim, which its validator runs in PyTorch, and which needs the
    copy-back like any other.
    Imports are local to avoid a lowering<->conversion import cycle.
    """
    if not (isinstance(value_node, torch.fx.Node) and value_node.op == "call_function"):
        return False
    args = value_node.args
    # An op the caller excluded from TensorRT never reaches a converter, so it
    # cannot emit an IKVCacheUpdateLayer and the engine will not alias it. Without
    # this the write is classified as engine-aliased, its copy_ is dropped, and
    # compile() later fails its own aliased_io cross-check. It still gets lifted and
    # copied back like any other non-aliasing write.
    if _write_op_is_torch_executed(value_node, settings):
        return False
    # The KV layer aliases the cache only if it is a direct network input; after
    # lifting, the mutated buffer is a placeholder the write op reads from.
    if not (args and isinstance(args[0], torch.fx.Node)):
        return False
    cache_input = _effective_cache_input(args[0])
    if cache_input is None:
        return False

    if value_node.target is torch.ops.aten.index_copy.default:
        from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
            _index_copy_kv_eligible,
        )

        # The validator applies its own placeholder check to args[0], and here that
        # can still be the clone lowering is about to erase. Point it at what the
        # converter will be handed instead; the partitioner, running after lowering,
        # passes nothing and reads args[0] itself.
        return bool(_index_copy_kv_eligible(value_node, input_node=cache_input))

    if value_node.target is torch.ops.aten.slice_scatter.default:
        from torch_tensorrt.dynamo.conversion.impl.slice_scatter import (
            KVWriteStatus,
            _kv_eligible,
            resolve_slice_scatter_write,
        )

        # Any status other than OK is a case in which the converter returns or
        # raises before it reaches _kv_eligible, so nothing aliases the cache.
        # Returning False routes the write to copy-back -- the copy_ is erased
        # either way, and the new value is re-attached as a graph output instead. A
        # full overwrite needs that: it returns the source, emits no KV layer, and
        # its write still has to be recorded. Two of the rest raise out of the
        # converter, so the compile aborts and the output is never reached; the
        # dynamic-dim one is validated away to PyTorch and keeps its copy-back.
        dim = args[2] if len(args) > 2 else 0
        start, end, step, status = resolve_slice_scatter_write(
            tuple(cache_shape),
            dim,
            args[3] if len(args) > 3 else None,
            args[4] if len(args) > 4 else None,
            args[5] if len(args) > 5 else None,
        )
        if status is not KVWriteStatus.OK:
            return False
        # OK is the only status that resolves all three bounds to Python ints.
        assert start is not None and end is not None and step is not None
        eligible, _reason = _kv_eligible(
            tuple(cache_shape), dim, start, end - start, step
        )
        return bool(eligible)

    return False


def aliased_input_bindings(aliased_io_maps: Iterable[Any]) -> Set[str]:
    """Collect the input-binding side of one or more ``aliased_io`` maps.

    Each map is ``output binding -> (input binding, kind)``, and a bare input binding
    is tolerated in place of the pair.
    """
    bindings: Set[str] = set()
    for amap in aliased_io_maps:
        if not amap:
            continue
        for v in amap.values():
            bindings.add(v[0] if isinstance(v, (tuple, list)) else v)
    return bindings


def assert_predicted_kv_aliased(
    aliased_in: Set[str],
    predicted_kv_bindings: List[str],
    settings: Optional[CompilationSettings] = None,
    *,
    engines_built: bool = True,
    copyback_bindings: Iterable[str] = (),
) -> None:
    """Ground-truth check on both directions of the classification.

    Each write ``lift_mutated_buffers`` classified as KV-aliased had its ``copy_``
    dropped in the expectation that the engine would alias it in place. If the
    converter did not actually emit an ``IKVCacheUpdateLayer`` for it, that
    write-back is silently lost. So assert every predicted-KV input binding
    appears in a compiled engine's ``aliased_io``, and raise loudly otherwise.
    Keyed on the ``buf_*`` binding name, which is stable across the later buffer
    rename and is what ``aliased_io`` records on the input side.

    The other direction fails just as badly and is not the same check. A write filed
    copy-back has its new value re-attached as a trailing graph output; if the engine
    aliases that buffer anyway, the runtime truncates the aliased output as an engine
    side effect while the outer graph still reads it, and the module raises on every
    call. So ``copyback_bindings`` -- the input bindings of the writes filed
    copy-back -- must be absent from ``aliased_io``.

    ``aliased_in`` is the ground truth, supplied by the caller because the two entry
    points hold it in different shapes: :func:`compile` reads it off the compiled
    submodules, while the engine converter has it on the interpreter result.

    ``engines_built=False`` says the caller produced no engine, so ``aliased_in`` is
    empty for reasons that say nothing about the predictions and every one of them
    would look unfulfilled. The check is skipped there instead. :func:`compile` passes
    ``not settings.dryrun``, because a dryrun returns before conversion; the engine
    converter takes ``dryrun`` too but never acts on it and builds either way, so it
    keeps the default.
    """
    copyback_bindings = list(copyback_bindings)
    if not predicted_kv_bindings and not copyback_bindings:
        return
    if not engines_built:
        logger.debug(
            "no engines were built, so the aliasing cross-check for predicted-KV "
            "%s and copy-back %s has nothing to verify against and is skipped",
            predicted_kv_bindings,
            copyback_bindings,
        )
        return
    missing = [b for b in predicted_kv_bindings if b not in aliased_in]
    if missing:
        cause = "the write did not end up inside a TensorRT engine"
        if settings is not None:
            if settings.min_block_size > 1:
                cause += (
                    f", most often because min_block_size "
                    f"({settings.min_block_size}) rejected the subgraph it landed "
                    "in; min_block_size=1 rules that out"
                )
            else:
                # The sibling branch's remedy is the value already in force here, so
                # naming min_block_size at all would read as "1 rejected the
                # subgraph; set it to 1" -- the value blamed and the value
                # recommended being the same is no remedy the reader can act on.
                # Name the cause instead.
                cause += (
                    ", most often because a converter or a capability validator "
                    "rejected the op"
                )
        raise RuntimeError(
            "lift_mutated_buffers classified these buffer writes as KV-cache "
            "(engine-aliased) and dropped their copy_, but the compiled engine "
            f"did not alias them (absent from aliased_io): {missing}. Their "
            "write-back would be silently dropped. The classification runs before "
            f"lowering and partitioning, so: {cause}."
        )

    unexpected = [b for b in copyback_bindings if b in aliased_in]
    if unexpected:
        # What raising here prevents: the runtime treats an aliased output as an
        # engine side effect and does not return it, so the surrounding graph reads
        # past the end of the results and the module raises on every call.
        raise RuntimeError(
            "lift_mutated_buffers classified these buffer writes as copy-back (not "
            "engine-aliased) and re-attached their new values as trailing graph "
            "outputs, but the compiled engine aliased them in place anyway (present "
            f"in aliased_io): {unexpected}. The classification runs before lowering, "
            "so a lowering pass rewrote the write into a form the converter could "
            "alias after all."
        )


def assert_no_kv_alias_markers_survived(
    gm: torch.fx.GraphModule, marked_writes: Iterable[str]
) -> None:
    """Check that lowering carried each ``_trt_no_kv_alias`` marker through to
    conversion.

    The marker is the only thing stopping the converter from aliasing a write that
    :func:`lift_mutated_buffers` deliberately filed copy-back, and it travels on
    ``node.meta`` through every pass in ``post_lowering``. A pass that rebuilds the
    node without its ``meta`` would drop it silently and the module would raise on
    every call, so the markers are read back here, where the fault can still be named.

    Called with the lowered graph and ``gm.meta['_no_kv_alias_writes']`` as recorded
    before lowering, and the two sets must match exactly. They diverge three ways: a
    recorded name keeps its node but loses the marker, so a pass rebuilt the node
    without its meta; an unrecorded node gains a marker while every recorded one keeps
    its own, so a pass copied the meta; or a recorded name goes missing and an
    unrecorded node gains a marker, so a pass replaced the node and the meta travelled
    with it. The third is the one the passes in ``post_lowering`` actually produce,
    because each carries a node's meta onto that node's own replacement.
    """
    expected = set(marked_writes)
    by_name = {node.name: node for node in gm.graph.nodes}
    lost = sorted(
        name
        for name in expected
        if name not in by_name or not by_name[name].meta.get("_trt_no_kv_alias")
    )
    gained = sorted(
        name
        for name, node in by_name.items()
        if name not in expected and node.meta.get("_trt_no_kv_alias")
    )
    if not lost and not gained:
        return
    if lost and gained:
        detail = (
            f"{lost} are gone and {gained} carry the marker instead, so a pass "
            "replaced the node and the meta travelled with it"
        )
    elif lost:
        detail = (
            f"the marker is gone from {lost} and no other node carries it, so a pass "
            "rebuilt the node without its meta"
        )
    else:
        detail = (
            f"{gained} carry the marker but were never marked, so a pass copied the "
            "meta onto another node"
        )
    # What raising here prevents: a marker rebuilt away lets the converter alias a
    # write whose new value is also a graph output, and the module raises on every
    # call; a marker copied onto a node the classifier chose to alias makes the
    # converter refuse that aliasing, which the predicted-KV cross-check then reports
    # as a partitioning failure that never happened; and a marker that travelled with
    # a replacement cannot be shown from here to have landed on the write the
    # classifier meant.
    raise RuntimeError(
        "lift_mutated_buffers set node.meta['_trt_no_kv_alias'] markers before "
        f"lowering that no longer match the lowered graph: {detail}."
    )


def erase_export_guards(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Remove ``torch.export`` ``_guards_fn`` call_module nodes.

    The partitioner does not delete these nodes: AccNodesFinder marks them
    unsupported, ``split()`` parks them on ``_run_on_gpu_*``, and
    ``compile_module`` skips convert for non-``_run_on_acc`` children. Direct
    ``convert_module(gm)`` would otherwise try to compile ``GuardsFn``.
    """
    for node in list(gm.graph.nodes):
        if (
            node.op == "call_module"
            and isinstance(node.target, str)
            and node.target == "_guards_fn"
        ):
            name = node.target
            gm.graph.erase_node(node)
            if name in dict(gm.named_children()):
                gm.delete_submodule(name)
            break
    gm.graph.lint()
    return gm


def lift_mutated_buffers(
    gm: torch.fx.GraphModule,
    settings: Optional[CompilationSettings] = None,
) -> Tuple[torch.fx.GraphModule, List[Tuple[str, str, torch.Tensor]]]:
    """Lift each mutated buffer from a ``get_attr`` to a ``placeholder``.

    A mutated buffer is identified by a trailing
    ``aten.copy_(get_attr_buffer, new_value)`` pattern, which is how
    ``ExportedProgram.module()`` represents a BUFFER_MUTATION.

    Returns ``(new_gm, lifted)`` where:

    * ``new_gm`` is a plain ``torch.fx.GraphModule`` whose ``forward``
      signature reflects the updated placeholder set. Necessary because
      ``ExportedProgram.module()`` produces a module whose forward is
      fixed by a pytree spec — recompiling alone doesn't pick up new
      placeholders.
    * ``lifted`` is a list of ``(placeholder_name, buffer_name, buffer_tensor)``
      tuples, in the order placeholders were appended (which matches the
      order they appear in the new gm's forward signature, after the
      original user inputs).

    Side effects: the trailing ``copy_`` of each mutated buffer is erased. A write
    the converter will lower to an ``IKVCacheUpdateLayer`` with in-place aliased
    I/O (an eligible ``slice_scatter`` / ``index_copy``, per
    :func:`_kv_write_will_alias`) relies on that engine aliasing for its
    write-back, so nothing further is added. Every other mutation -- a non-KV
    buffer, or a ``slice_scatter`` / ``index_copy`` that fails eligibility, whether
    the converter then lowers it to a non-aliasing concatenation or scatter,
    returns the source, or raises -- has no engine aliasing, so its new value is
    re-appended as a graph output ("copy-back") and its buffer name recorded, in
    output order, in ``gm.meta['_copyback_mutation_buffers']`` -- the
    downstream exporters (``create_trt_exp_program`` /
    ``_declare_aliased_kv_mutations_on_ep``) read that list to reclassify those
    outputs as BUFFER_MUTATIONs. Copy-back is correct but not free; see the module
    docstring for what each such buffer costs per call.

    One aliasable write is routed to copy-back anyway: one whose result nothing else
    consumes. Erasing its ``copy_`` leaves it dead, and dead-code elimination takes it
    before any engine can alias it. Copy-back makes it live again, so it needs the
    converter to leave it alone -- ``node.meta['_trt_no_kv_alias']`` on the write node
    says so, and :func:`assert_no_kv_alias_markers_survived` re-reads the markers after
    lowering.

    That marker goes on the write *node* while the classification is per *buffer*, so
    one value node written back into two buffers would put the two in conflict: the
    node marked on one buffer's behalf while the other buffer's write is predicted KV.
    ``torch.export`` rejects that shape with ``SpecViolationError`` before the
    classifier sees it, so it is a constraint on what can arrive here rather than a
    case to guard against.
    """
    # Find all aten.copy_(get_attr_X, _) calls. The first arg's target is
    # the buffer name. Some EPs emit copy_.default, others copy_.
    mutation_pairs: List[Tuple[torch.fx.Node, torch.fx.Node]] = (
        []
    )  # (copy_node, get_attr_node)
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target not in (torch.ops.aten.copy_.default, torch.ops.aten.copy_):
            continue
        if not node.args:
            continue
        target = node.args[0]
        if isinstance(target, torch.fx.Node) and target.op == "get_attr":
            mutation_pairs.append((node, target))

    if not mutation_pairs:
        return gm, []

    # Find the position to insert new placeholders (after the last existing placeholder).
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    insert_after = placeholders[-1] if placeholders else None

    lifted: List[Tuple[str, str, torch.Tensor]] = []
    seen_buffers: Dict[str, torch.fx.Node] = {}  # buffer name -> new placeholder node

    # Each mutated buffer's write-back is handled one of two ways downstream:
    #   - Engine-level aliasing (zero-copy): the slice_scatter / index_copy
    #     converters emit an IKVCacheUpdateLayer whose output is aliased in-place
    #     to the cache input. We drop the copy_ and rely on that aliasing.
    #   - Copy-back: any other mutation (a non-KV buffer such as a convolution-
    #     state ring-buffer, OR a slice_scatter / index_copy the converter cannot
    #     turn into an IKVCacheUpdateLayer) has no aliasing, so its new value is
    #     re-attached as an ordinary BUFFER_MUTATION output that ExecuTorch copies
    #     back after the delegate runs.
    # `_kv_write_will_alias` decides between them by reusing the converters' own
    # eligibility predicates (not the op target alone), so a non-aliasable
    # slice_scatter / index_copy falls to copy-back rather than being dropped.
    # index_put has no aliasing converter, so it always falls to copy-back too.
    # (new_value_node, buffer_name, input_binding_name)
    copyback: List[Tuple[torch.fx.Node, str, str]] = []
    # Input-binding names of writes predicted to alias (KV). compile() asserts each
    # actually appears in the engine's aliased_io, and that no copy-back binding
    # does, turning either misprediction into a loud error instead of a silently
    # dropped write-back or a module that raises on every call. The binding name
    # (buf_*) is the stable key: it survives the buffer renaming inline does later,
    # and it is exactly what aliased_io records on the input side.
    predicted_kv_bindings: set[str] = set()
    # Names of the write nodes marked _trt_no_kv_alias below. Read back after
    # lowering by :func:`assert_no_kv_alias_markers_survived`.
    no_kv_alias_writes: List[str] = []

    for copy_node, get_attr_node in mutation_pairs:
        buffer_name = get_attr_node.target
        # A get_attr target is fully qualified, so a buffer owned by a submodule
        # arrives as "layers.0.self_attn.kv_cache.k_cache". getattr does not walk a
        # dotted path, so it reports every nested buffer as missing; get_buffer
        # resolves through the submodules.
        try:
            buffer_tensor = gm.get_buffer(buffer_name)
        except AttributeError:
            logger.warning(
                "lift_mutated_buffers: get_attr target %s not found on gm; skipping",
                buffer_name,
            )
            continue
        if not isinstance(buffer_tensor, torch.Tensor):
            logger.debug(
                "lift_mutated_buffers: attribute %s is not a Tensor; skipping",
                buffer_name,
            )
            continue

        if buffer_name in seen_buffers:
            # Same buffer mutated more than once — already lifted; just remove
            # this copy_ node and rely on the existing placeholder.
            replacement = seen_buffers[buffer_name]
        else:
            # Build a unique placeholder name from the buffer name.
            placeholder_name = "buf_" + buffer_name.replace(".", "_")
            base = placeholder_name
            suffix = 0
            existing = {n.name for n in gm.graph.nodes}
            while placeholder_name in existing:
                suffix += 1
                placeholder_name = f"{base}_{suffix}"

            if insert_after is not None:
                with gm.graph.inserting_after(insert_after):
                    new_ph = gm.graph.placeholder(placeholder_name)
            else:
                # No existing placeholders — insert at graph start.
                with gm.graph.inserting_before(next(iter(gm.graph.nodes))):
                    new_ph = gm.graph.placeholder(placeholder_name)
            new_ph.meta["val"] = get_attr_node.meta.get(
                "val",
                torch.empty_like(buffer_tensor, device="meta"),
            )
            new_ph.meta["_lifted_buffer"] = buffer_name
            insert_after = new_ph
            seen_buffers[buffer_name] = new_ph
            replacement = new_ph
            lifted.append((placeholder_name, buffer_name, buffer_tensor))

        # Re-route every use of the original get_attr (other than the copy_ itself)
        # to the new placeholder.
        get_attr_node.replace_all_uses_with(replacement)

        # KV writes rely on engine-level aliasing, so the trailing copy_ is
        # redundant and dropped. Other (non-KV) mutations have no aliasing:
        # record their new value so we can re-attach it as a copy-back
        # BUFFER_MUTATION output below.
        #
        # Drop the (now input-target) copy_ first, so the classification reads the
        # graph the lowering passes and the partitioner will read. The copy_ is the
        # last reader of the buffer placeholder in the ordinary write-only shape, and
        # whether it is still there decides whether a clone of the cache survives
        # into conversion.
        new_value = copy_node.args[1] if len(copy_node.args) > 1 else None
        gm.graph.erase_node(copy_node)
        if isinstance(new_value, torch.fx.Node):
            will_alias = _kv_write_will_alias(
                new_value, tuple(buffer_tensor.shape), settings
            )
            if will_alias and not new_value.users:
                # The copy_ was this write's only consumer, so it is now dead and
                # dead-code elimination will take it before any engine sees it --
                # leaving the prediction unfulfilled and the write-back lost. Copy-back
                # revives it, because the value is re-attached as a graph output, but
                # then the converter would alias a buffer that is also copy-back and
                # the module would raise on every call. Marking the write forbids the
                # aliasing, which is what makes the copy-back route available at all.
                # Both eligibility checks read the marker: impl.slice_scatter off
                # ctx.current_node, aten_ops_converters._index_copy_kv_eligible off
                # the node it is handed.
                new_value.meta["_trt_no_kv_alias"] = True
                no_kv_alias_writes.append(new_value.name)
                will_alias = False
            if will_alias:
                predicted_kv_bindings.add(replacement.name)
            else:
                copyback.append((new_value, buffer_name, replacement.name))
                # A clone of this buffer holds its old value past the point where
                # forward writes the new one, so lowering must keep it.
                replacement.meta["_copyback_buffer"] = True

        # Erase the now-unused get_attr.
        if not get_attr_node.users:
            gm.graph.erase_node(get_attr_node)

    # Re-attach non-KV mutation new-values as graph outputs so they survive
    # compilation (otherwise, with the copy_ gone, they are dead and eliminated).
    # Appended in order; recorded so create_trt_exp_program / _declare can tag the
    # corresponding engine outputs as BUFFER_MUTATION (copy-back) downstream.
    copyback_buffers: List[str] = []
    copyback_bindings: List[str] = []
    if copyback:
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        out_args = list(output_node.args[0])
        out_args.extend(nv for nv, _, _ in copyback)
        output_node.args = (tuple(out_args),)
        copyback_buffers = [buf for _, buf, _ in copyback]
        copyback_bindings = sorted({binding for _, _, binding in copyback})

    gm.graph.lint()

    if not lifted:
        gm.meta["_copyback_mutation_buffers"] = copyback_buffers
        gm.meta["_copyback_bindings"] = copyback_bindings
        gm.meta["_predicted_kv_bindings"] = sorted(predicted_kv_bindings)
        gm.meta["_no_kv_alias_writes"] = no_kv_alias_writes
        return gm, []

    # ExportedProgram.module() produces a GraphModule whose forward is
    # generated by a ``_PyTreeCodeGen`` baked into the graph: the body
    # unpacks args through a stored pytree spec, ignoring any added
    # placeholders. Rebuild the gm with the default ``CodeGen`` so the
    # forward signature reflects the placeholder set as written.
    # First remove the call to ``_guards_fn`` (generated for the original
    # arity; would fail after lifting).
    gm = erase_export_guards(gm)

    # Reset codegen to the plain CodeGen so the forward args = placeholders.
    gm.graph.set_codegen(torch.fx.graph.CodeGen())
    gm.graph.lint()

    new_gm = torch.fx.GraphModule(gm, gm.graph)
    for attr in ("_in_spec", "_out_spec"):
        if hasattr(new_gm, attr):
            try:
                delattr(new_gm, attr)
            except AttributeError:
                pass
    new_gm.recompile()
    new_gm.meta["_copyback_mutation_buffers"] = copyback_buffers
    new_gm.meta["_copyback_bindings"] = copyback_bindings
    new_gm.meta["_predicted_kv_bindings"] = sorted(predicted_kv_bindings)
    new_gm.meta["_no_kv_alias_writes"] = no_kv_alias_writes

    logger.debug(
        "Lifted %d mutated buffer(s) to placeholders: %s; copy-back buffers: %s",
        len(lifted),
        [(p, b) for p, b, _ in lifted],
        copyback_buffers,
    )

    return new_gm, lifted


class _HiddenCopybackOutputs(torch.fx.graph.CodeGen):  # type: ignore[misc]
    """Generate a ``forward`` that writes the trailing copy-back values into their
    buffers instead of returning them.

    The values stay on the graph's output node, so everything that reads the graph
    rather than calling the module still sees them: the exporters, dead-code
    elimination, and the ``torch.fx.Interpreter`` a non-strict ``torch.export`` runs.
    ``torch_tensorrt.save`` declares each one a BUFFER_MUTATION; a plain non-strict
    ``torch.export.export`` returns them as extra user outputs.
    """

    def __init__(
        self,
        buffers: Sequence[str],
        copied: Sequence[bool],
        helper: str,
        returned_copied: Sequence[bool] = (),
    ) -> None:
        super().__init__()
        self.buffers = list(buffers)
        self.copied = list(copied)
        self.returned_copied = list(returned_copied)
        # The generated code names nothing else from outside: an input may be called
        # anything, even getattr.
        self.helper = helper

    def additional_globals(self) -> List[Tuple[str, Any]]:
        return [(self.helper, write_copyback_buffers)]

    def generate_output(self, output_args: Any, *args: Any, **kwargs: Any) -> str:
        write_back = ""
        if self.buffers and isinstance(output_args, (list, tuple)):
            split = len(output_args) - len(self.buffers)
            repr_fn = kwargs.get("repr_fn") or repr
            # A flagged returned value is copied, under its own name, before the
            # writes.
            rebound = list(
                dict.fromkeys(
                    repr_fn(value)
                    for value, copied in zip(output_args[:split], self.returned_copied)
                    if copied
                )
            )
            names = "".join(f"{name!r}, " for name in self.buffers)
            values = "".join(
                f"{repr_fn(value)}"
                f"{'.clone()' if copied and repr_fn(value) not in rebound else ''}, "
                for value, copied in zip(output_args[split:], self.copied)
            )
            write_back = "".join(f"{name} = {name}.clone()\n" for name in rebound)
            write_back += f"{self.helper}(self, ({names}), ({values}))\n"
            output_args = output_args[:split]
            # descs, when the caller passes it, is one entry per untruncated output.
            kwargs.pop("descs", None)
        return write_back + str(super().generate_output(output_args, *args, **kwargs))


class _BufferProxyTracer(torch.fx.Tracer):  # type: ignore[misc]
    """``torch.load`` rebuilds a pickled GraphModule by tracing its ``forward``.
    Buffers read as proxies, so a written value computed from them, such as a
    step counter, is recorded rather than run once at load."""

    proxy_buffer_attributes = True


def write_copyback_buffers(
    module: torch.nn.Module, names: Sequence[str], values: Sequence[torch.Tensor]
) -> None:
    """Copy each value into ``module``'s buffer named at its position.

    ``torch.load`` rebuilds a pickled module by tracing its ``forward``, which
    records these copies as graph nodes, so the rebuilt module writes its buffers
    and an export of it declares the writes. A value that comes from neither the
    inputs nor the buffers, as from ``torch.rand``, would run only once, at load,
    so that raises.
    """
    # The buffers hold state, not a differentiable output, so a value with
    # autograd history must not attach that history to them.
    with torch.no_grad():
        for name, value in zip(names, values):
            buffer = getattr(module, name)
            if isinstance(buffer, torch.fx.Proxy) and not isinstance(
                value, torch.fx.Proxy
            ):
                raise RuntimeError(
                    f"Cannot rebuild the write into buffer '{name}' by tracing this "
                    "module: its value depends on no input or buffer (for example "
                    "torch.rand), so the rebuilt module would not write it. Save it "
                    "with torch_tensorrt.save instead of pickling it."
                )
            # Tracing, by torch.jit.trace or torch.load, does not record no_grad, so
            # the detach is spelled out. The ATen op records a node that graph
            # cleanup keeps, as it does any op that writes its input.
            torch.ops.aten.copy_.default(buffer, value.detach())


def _may_share_a_buffer(
    gm: torch.fx.GraphModule,
    value: torch.fx.Node,
    buffers: Set[str],
) -> bool:
    """Whether ``value`` can be one of ``buffers``, or a view of one, when it is read.

    The walk follows what can hand back its input's memory: an ATen op whose schema
    returns an alias, an item of a tuple, and the matching output of a submodule left
    to PyTorch, into whose graph it goes. It stops at an engine and at an ATen op
    that returns new memory. Anything else, such as a Python function or an opaque
    submodule, is followed through all its inputs, so its value may be copied when
    it did not need to be.
    """
    from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

    # Each entry is a node, its module, and the calls that entered that module.
    pending: List[Tuple[Any, Any, Tuple[Any, ...]]] = [(value, gm, ())]
    seen = set()

    def enter(sub: Any, outputs: Any, frames: Tuple[Any, ...]) -> None:
        torch.fx.node.map_arg(outputs, lambda n: pending.append((n, sub, frames)))

    while pending:
        node, module, frames = pending.pop()
        if not isinstance(node, torch.fx.Node) or (node, module, frames) in seen:
            continue
        seen.add((node, module, frames))
        if node.op == "placeholder":
            if frames:
                outer, call = frames[-1]
                inputs = [n for n in module.graph.nodes if n.op == "placeholder"]
                index = inputs.index(node)
                if index < len(call.args):
                    pending.append((call.args[index], outer, frames[:-1]))
        elif node.op == "get_attr":
            # A partition reaches a written buffer only through a placeholder, since
            # buffers are lifted before partitioning.
            if not frames and node.target in buffers:
                return True
        elif node.op == "call_module":
            sub = module.get_submodule(node.target)
            if isinstance(sub, TorchTensorRTModule):
                continue
            if isinstance(sub, torch.fx.GraphModule):
                out = next(iter(sub.graph.find_nodes(op="output")))
                enter(sub, out.args[0], frames + ((module, node),))
            else:
                pending.extend((n, module, frames) for n in node.all_input_nodes)
        elif node.op == "call_function" and node.target is operator.getitem:
            base, index = node.args[0], node.args[1]
            sub = (
                module.get_submodule(base.target)
                if isinstance(base, torch.fx.Node) and base.op == "call_module"
                else None
            )
            if isinstance(sub, torch.fx.GraphModule) and not isinstance(
                sub, TorchTensorRTModule
            ):
                out = next(iter(sub.graph.find_nodes(op="output"))).args[0]
                if isinstance(out, (tuple, list)) and isinstance(index, int):
                    enter(sub, out[index], frames + ((module, base),))
                    continue
            pending.append((base, module, frames))
        elif node.op == "call_function" and isinstance(
            node.target, torch._ops.OpOverload
        ):
            if any(r.alias_info is not None for r in node.target._schema.returns):
                pending.extend((n, module, frames) for n in node.all_input_nodes)
        else:
            pending.extend((n, module, frames) for n in node.all_input_nodes)
    return False


def lifted_buffer_attrs(
    names: Iterable[str], model_names: Iterable[str] = ()
) -> Dict[str, str]:
    """The attribute each lifted buffer, by its name in the model, is registered
    under on the compiled module.

    ``register_buffer`` takes no ``.``, so a nested name is flattened, and a
    flattened name that another lifted buffer or a top-level name of the model
    (``model_names``, the model's state names) already uses gets a suffix, in sorted
    order. A flat name stays as it is, so ``state_dict`` keys do not change. The
    attributes follow from the model's names alone, so they are found again on a
    module that was saved and loaded.
    """
    names = set(names)
    attrs = {name: name for name in names if "." not in name}
    taken = set(attrs.values()) | {name.split(".")[0] for name in model_names}
    for name in sorted(names - set(attrs)):
        attr = base = "lifted_buf_" + name.replace(".", "_")
        suffix = 0
        while attr in taken:
            suffix += 1
            attr = f"{base}_{suffix}"
        taken.add(attr)
        attrs[name] = attr
    return attrs


def copy_into_lifted_buffers(
    module: torch.nn.Module, state: Dict[str, torch.Tensor], lifted: Iterable[str]
) -> None:
    """Copy each value in ``state``, the model's state keyed by its names, into the
    buffer of the compiled ``module`` that holds it. ``lifted`` names the buffers
    the model writes, the ones compile() registered on the module."""
    for name, attr in lifted_buffer_attrs(lifted, state).items():
        buffer = module._buffers.get(attr)
        value = state.get(name)
        if buffer is not None and value is not None and buffer.shape == value.shape:
            buffer.copy_(value)


@contextlib.contextmanager
def buffers_restored(module: torch.nn.Module) -> Iterator[None]:
    """Put ``module``'s buffers back as they were when the block exits.

    Calling a compiled module advances the buffers it writes, so code that runs it
    only to trace it or to check its output would otherwise leave it a call ahead.
    The copies are kept on the CPU, so a large cache does not need twice the device
    memory. Only a buffer whose bytes changed is written back: one the block never
    wrote can be read-only. An expanded buffer is never written back: it cannot be
    written, and only a write to the buffer it expands can change it. A sparse or
    quantized buffer has no byte view, and a meta buffer has no data, so they are
    left alone.
    """
    saved = {
        name: buffer.detach().to("cpu", copy=True)
        for name, buffer in module.named_buffers()
        if buffer.layout == torch.strided
        and not buffer.is_quantized
        and not buffer.is_meta
    }
    try:
        yield
    finally:
        # Inference mode accepts the write whether or not a buffer was made in it.
        with torch.inference_mode():
            for name, buffer in module.named_buffers():
                old = saved.get(name)
                if old is None or any(
                    n > 1 and st == 0 for n, st in zip(buffer.shape, buffer.stride())
                ):
                    continue
                if not torch.equal(_bytes(buffer), _bytes(old)):
                    buffer.copy_(old)


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    # Bytes rather than values, so a NaN compares equal to itself. A conjugate or
    # negative view, a strided or an expanded tensor cannot be viewed as bytes as it is.
    return (
        tensor.detach()
        .resolve_conj()
        .resolve_neg()
        .to("cpu")
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
    )


def hide_copyback_outputs(gm: torch.fx.GraphModule, buffers: Sequence[str]) -> None:
    """Make ``gm``'s ``forward`` copy its last ``len(buffers)`` graph outputs into
    those buffers, in order, and return the outputs before them.

    They remain outputs of the graph, so the exporters still find them and declare
    them BUFFER_MUTATIONs. Every value is read before any buffer is written: a value
    that can share memory with one of the buffers, as when one buffer's new contents
    are another's old ones or a view of its own, is copied first. So is a returned
    value that can share a buffer written with other contents.
    """
    if not buffers:
        return
    output_node = next(n for n in gm.graph.nodes if n.op == "output")
    names = set(buffers)
    copied = [
        isinstance(value, torch.fx.Node) and _may_share_a_buffer(gm, value, names)
        for value in output_node.args[0]
    ]
    split = len(copied) - len(buffers)
    # Export writes a buffer's new value and then reads the outputs that alias it
    # from the buffer, so a returned view of a buffer written with a view of itself
    # is read after the writes. A view of a buffer that gets other contents is read
    # before them.
    overwritten = {
        name
        for name, value in zip(buffers, output_node.args[0][split:])
        if not (
            isinstance(value, torch.fx.Node) and _may_share_a_buffer(gm, value, {name})
        )
    }
    returned = [
        flag and _may_share_a_buffer(gm, value, overwritten)
        for value, flag in zip(output_node.args[0][:split], copied[:split])
    ]
    used = {node.name for node in gm.graph.nodes}
    helper = "write_copyback_buffers"
    while helper in used:
        helper += "_"
    gm.graph.set_codegen(
        _HiddenCopybackOutputs(buffers, copied[split:], helper, returned)
    )
    gm.graph._tracer_cls = gm._tracer_cls = _BufferProxyTracer
    gm.recompile()


def inline_lifted_buffers_into_gm(
    gm: torch.fx.GraphModule,
    lifted_buffers: List[Tuple[str, str, torch.Tensor]],
    model_names: Iterable[str] = (),
) -> torch.fx.GraphModule:
    """Inline lifted buffers as ``get_attr`` reads on the compiled GraphModule.

    After ``lift_mutated_buffers`` + ``compile_module``, ``gm`` is a
    ``torch.fx.GraphModule`` whose top-level ``forward`` takes the user's
    inputs *plus* the lifted buffers as placeholders. To make the result
    look like a normal ``nn.Module`` (and to make it serializable via
    ``torch_tensorrt.save`` / ``torch.export``) we:

    1. Register each lifted buffer as a ``register_buffer`` on ``gm``.
    2. Replace each buffer-placeholder node with a ``get_attr`` node that
       reads from ``gm.<buffer_name>``.
    3. Recompile.

    The resulting GraphModule's ``forward`` takes only the user's inputs;
    the buffers are threaded through internally via the get_attr nodes.
    The engine still sees the buffers as input bindings (and writes through
    them via aliased I/O); the buffer storage lives on ``gm`` so subsequent
    calls reuse the mutated state.

    This transform is a no-op if ``lifted_buffers`` is empty (returns
    ``gm`` unchanged).
    """
    if not lifted_buffers:
        return gm

    placeholder_to_buf: Dict[str, str] = {
        ph_name: buf_name for ph_name, buf_name, _ in lifted_buffers
    }
    # Clone so the gm owns its own storage.
    tensors: Dict[str, torch.Tensor] = {}
    for _ph_name, buf_name, tensor in lifted_buffers:
        tensors.setdefault(buf_name, tensor)
    buf_to_attr = lifted_buffer_attrs(tensors, model_names)
    for buf_name, tensor in tensors.items():
        attr_name = buf_to_attr[buf_name]
        if hasattr(gm, attr_name):
            if "." in buf_name:
                raise RuntimeError(
                    f"Cannot register buffer '{buf_name}' as '{attr_name}': the "
                    "compiled module already has an attribute of that name."
                )
            continue
        # forward writes copy-back buffers in place, which an inference tensor
        # refuses outside inference mode.
        with torch.inference_mode(False):
            gm.register_buffer(attr_name, tensor.clone())

    # Copy-back mutation targets were recorded (in gm.meta) under the buffers'
    # original names; nested ones were just flattened to lifted_buf_* above. Remap
    # the recorded targets through the same mapping so the downstream exporter
    # names a buffer that still exists -- otherwise the ExportedProgram verifier
    # rejects the BUFFER_MUTATION ("output ... does not point to a buffer that
    # exists").
    copyback = gm.meta.get("_copyback_mutation_buffers")
    if copyback:
        gm.meta["_copyback_mutation_buffers"] = [
            buf_to_attr.get(name, name) for name in copyback
        ]

    # Find placeholders we need to replace. Insert get_attr nodes BEFORE
    # removing the placeholders so the graph remains valid throughout.
    placeholders_to_remove = []
    for node in list(gm.graph.nodes):
        if node.op != "placeholder":
            continue
        if node.name not in placeholder_to_buf:
            continue
        buf_name = placeholder_to_buf[node.name]
        with gm.graph.inserting_after(node):
            get_attr_node = gm.graph.get_attr(buf_to_attr[buf_name])
        # Carry over fake-tensor metadata so downstream passes see the right
        # shape/dtype.
        if "val" in node.meta:
            get_attr_node.meta["val"] = node.meta["val"]
        node.replace_all_uses_with(get_attr_node)
        placeholders_to_remove.append(node)

    for node in placeholders_to_remove:
        gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()
    logger.debug(
        "Inlined %d lifted buffer(s) into gm as get_attr reads: %s",
        len(lifted_buffers),
        [b for _, b, _ in lifted_buffers],
    )
    return gm
