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
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)


def _kv_write_will_alias(value_node: object, cache_shape: Tuple[int, ...]) -> bool:
    """Whether the converter will emit an ``IKVCacheUpdateLayer`` (with in-place
    aliased I/O) for this mutated buffer's new-value node.

    Eligibility depends on the write dim, static shapes, and the cache being a
    direct network input, so this reuses the converters' own eligibility
    predicates. A ``slice_scatter`` / ``index_copy`` that is not eligible is
    lowered to a non-aliasing scatter, so its write-back is kept as a
    BUFFER_MUTATION output (copy-back) rather than dropped. Imports are local to
    avoid a lowering<->conversion import cycle.
    """
    if not (isinstance(value_node, torch.fx.Node) and value_node.op == "call_function"):
        return False
    args = value_node.args
    # The KV layer aliases the cache only if it is a direct network input; after
    # lifting, the mutated buffer is a placeholder the write op reads from.
    if not (
        args and isinstance(args[0], torch.fx.Node) and args[0].op == "placeholder"
    ):
        return False

    if value_node.target is torch.ops.aten.index_copy.default:
        from torch_tensorrt.dynamo.conversion.aten_ops_converters import (
            _index_copy_kv_eligible,
        )

        return _index_copy_kv_eligible(value_node)

    if value_node.target is torch.ops.aten.slice_scatter.default:
        from torch_tensorrt.dynamo.conversion.impl.slice_scatter import _kv_eligible

        dim = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        start = args[3] if len(args) > 3 and isinstance(args[3], int) else 0
        # Prefer the source's extent along `dim` for the write length; fall back
        # to end-start from the slice bounds.
        update_len = None
        src = args[1] if len(args) > 1 else None
        if isinstance(src, torch.fx.Node):
            src_val = src.meta.get("val")
            if src_val is not None and dim < len(src_val.shape):
                s = src_val.shape[dim]
                update_len = s if isinstance(s, int) else None
        if update_len is None:
            end = (
                args[4]
                if len(args) > 4 and isinstance(args[4], int)
                else (cache_shape[dim] if dim < len(cache_shape) else 0)
            )
            update_len = max(end - start, 0)
        eligible, _reason = _kv_eligible(tuple(cache_shape), dim, start, update_len)
        return eligible

    return False


def assert_predicted_kv_aliased(
    gm: torch.fx.GraphModule, predicted_kv_bindings: List[str]
) -> None:
    """Ground-truth check for the KV predictions :func:`_kv_write_will_alias` made.

    Each write ``lift_mutated_buffers`` classified as KV-aliased had its ``copy_``
    dropped in the expectation that the engine would alias it in place. If the
    converter did not actually emit an ``IKVCacheUpdateLayer`` for it, that
    write-back is silently lost. So assert every predicted-KV input binding
    appears in a compiled engine's ``aliased_io``, and raise loudly otherwise.
    Keyed on the ``buf_*`` binding name, which is stable across the later buffer
    rename and is what ``aliased_io`` records on the input side.
    """
    if not predicted_kv_bindings:
        return
    aliased_in: set = set()
    for _name, sub in gm.named_children():
        amap = getattr(sub, "aliased_io", None)
        if amap:
            for v in amap.values():
                aliased_in.add(v[0] if isinstance(v, (tuple, list)) else v)
    missing = [b for b in predicted_kv_bindings if b not in aliased_in]
    if missing:
        raise RuntimeError(
            "lift_mutated_buffers classified these buffer writes as KV-cache "
            "(engine-aliased) and dropped their copy_, but the compiled engine "
            f"did not alias them (absent from aliased_io): {missing}. Their "
            "write-back would be silently dropped."
        )


def lift_mutated_buffers(
    gm: torch.fx.GraphModule,
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
    buffer, or a ``slice_scatter`` / ``index_copy`` that fails eligibility and is
    lowered to a non-aliasing scatter -- has no engine aliasing, so its new value
    is re-appended as a graph output ("copy-back") and its buffer name recorded,
    in output order, in ``gm.meta['_copyback_mutation_buffers']`` -- the
    downstream exporters (``create_trt_exp_program`` /
    ``_declare_aliased_kv_mutations_on_ep``) read that list to reclassify those
    outputs as BUFFER_MUTATIONs.
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
    copyback: List[Tuple[torch.fx.Node, str]] = []  # (new_value_node, buffer_name)
    # Input-binding names of writes predicted to alias (KV). compile() asserts each
    # actually appears in the engine's aliased_io, turning a mis-prediction into a
    # loud error instead of a silently dropped write-back. The binding name (buf_*)
    # is the stable key: it survives the buffer renaming inline does later, and it
    # is exactly what aliased_io records on the input side.
    predicted_kv_bindings: set[str] = set()

    for copy_node, get_attr_node in mutation_pairs:
        buffer_name = get_attr_node.target
        if not hasattr(gm, buffer_name):
            logger.warning(
                "lift_mutated_buffers: get_attr target %s not found on gm; skipping",
                buffer_name,
            )
            continue
        buffer_tensor = getattr(gm, buffer_name)
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
        # BUFFER_MUTATION output below, then drop the (now input-target) copy_.
        new_value = copy_node.args[1] if len(copy_node.args) > 1 else None
        if isinstance(new_value, torch.fx.Node):
            if _kv_write_will_alias(new_value, tuple(buffer_tensor.shape)):
                predicted_kv_bindings.add(replacement.name)
            else:
                copyback.append((new_value, buffer_name))
        gm.graph.erase_node(copy_node)

        # Erase the now-unused get_attr.
        if not get_attr_node.users:
            gm.graph.erase_node(get_attr_node)

    # Re-attach non-KV mutation new-values as graph outputs so they survive
    # compilation (otherwise, with the copy_ gone, they are dead and eliminated).
    # Appended in order; recorded so create_trt_exp_program / _declare can tag the
    # corresponding engine outputs as BUFFER_MUTATION (copy-back) downstream.
    copyback_buffers: List[str] = []
    if copyback:
        output_node = next(n for n in gm.graph.nodes if n.op == "output")
        out_args = list(output_node.args[0])
        out_args.extend(nv for nv, _ in copyback)
        output_node.args = (tuple(out_args),)
        copyback_buffers = [buf for _, buf in copyback]

    gm.graph.lint()

    if not lifted:
        gm.meta["_copyback_mutation_buffers"] = copyback_buffers
        gm.meta["_predicted_kv_bindings"] = sorted(predicted_kv_bindings)
        return gm, []

    # ExportedProgram.module() produces a GraphModule whose forward is
    # generated by a ``_PyTreeCodeGen`` baked into the graph: the body
    # unpacks args through a stored pytree spec, ignoring any added
    # placeholders. Rebuild the gm with the default ``CodeGen`` so the
    # forward signature reflects the placeholder set as written.
    # First remove the call to ``_guards_fn`` (generated for the original
    # arity; would fail after lifting).
    for node in list(gm.graph.nodes):
        if (
            node.op == "call_module"
            and isinstance(node.target, str)
            and node.target == "_guards_fn"
        ):
            gm.graph.erase_node(node)
            break

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
    new_gm.meta["_predicted_kv_bindings"] = sorted(predicted_kv_bindings)

    logger.debug(
        "Lifted %d mutated buffer(s) to placeholders: %s; copy-back buffers: %s",
        len(lifted),
        [(p, b) for p, b, _ in lifted],
        copyback_buffers,
    )

    return new_gm, lifted


def inline_lifted_buffers_into_gm(
    gm: torch.fx.GraphModule,
    lifted_buffers: List[Tuple[str, str, torch.Tensor]],
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
    # Register buffers as module state, mapping each original buffer name to the
    # attribute name it is registered under. ``nn.Module.register_buffer``
    # rejects names containing "." and ``get_attr`` on a dotted target would
    # traverse submodules that no longer exist on this flattened GraphModule
    # (e.g. HF's ``model.layers.0.k_cache``). Flat names keep their original
    # name to preserve ``state_dict`` keys; nested names are sanitized to a
    # unique flat attribute. Clone so the gm owns its own storage.
    buf_to_attr: Dict[str, str] = {}
    for _ph_name, buf_name, tensor in lifted_buffers:
        if buf_name in buf_to_attr:
            continue
        if "." in buf_name:
            attr_name = "lifted_buf_" + buf_name.replace(".", "_")
            base = attr_name
            suffix = 0
            while hasattr(gm, attr_name):
                suffix += 1
                attr_name = f"{base}_{suffix}"
        else:
            attr_name = buf_name
        if not hasattr(gm, attr_name):
            gm.register_buffer(attr_name, tensor.clone())
        buf_to_attr[buf_name] = attr_name

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
