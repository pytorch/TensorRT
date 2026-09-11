"""Let a TensorRT engine update an aliased mutable buffer in place.

An engine with aliased I/O (a KV cache) writes its aliased output *through* the
aliased input's pointer, so running the engine over the buffer already is the
update. Nothing in the ExecuTorch pipeline knows that, so by default the buffer
makes a full round trip on every execution:

* ``PropagateDevicePass`` wraps every delegate input in ``et_copy._h2d_copy``,
  so the delegate is handed a per-call staging copy rather than the caller's
  buffer. The engine's in-place write lands in that copy.
* the aliased output is threaded back out as a delegate output, and ExecuTorch
  copies it into the buffer afterwards to make the update stick.

For a cache-sized buffer that is two copies per execution of something the
engine could have written directly. This module removes both, in the same
spirit as ``partitioner._keep_mutated_buffers_above_delegate``: let the upstream
pass run, then correct its output for the case Torch-TensorRT owns.

The two halves are inseparable and run at different times:

* :func:`rewire_aliased_mutations_to_buffers`, on the exported program before
  partitioning, drops the copy-back by declaring that the buffer *is* the
  mutation's result. The aliased output then has no user and disappears from the
  partition. :func:`order_copyback_mutations_first` then repairs, on the Edge
  program, the mutation-spec pairing that declaration disturbs downstream --
  a crossing this is one cause of and not the only one, which is why
  ``export()`` runs that repair over every method rather than the rewired ones.
* :func:`unstage_aliased_buffers_pass`, as a ``to_out_var_pass``, drops the
  staging so the engine writes the caller's buffer rather than a copy.

Applying only the first would leave the engine writing a discarded staging copy
with nothing to copy back -- the buffer would simply never update. So neither
pass is public on its own: the rewiring is reached only through
``export(..., zero_copy_kv=True)``, and the un-staging only through
:func:`zero_copy_backend_config`. That, plus :func:`check_zero_copy_kv` -- which
reads a finalized program back and refuses one in which the engine does not end
up writing the caller's buffer -- is what this module exports.
"""

import json
import logging
import operator
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Set

import torch
from executorch.exir.pass_base import PassBase
from torch.fx import Node

if TYPE_CHECKING:
    from executorch.exir import ExecutorchBackendConfig

logger = logging.getLogger(__name__)


def _aliased_inputs_by_output_index(
    exported_program: Any, engine_node: Node
) -> Dict[int, Node]:
    """Map each aliased output index of one engine to the input it writes in place.

    Reads the engine's own ``aliased_io`` rather than inferring aliasing from the
    graph. The graph cannot tell the difference: an aliased KV mutation and a
    copy-back mutation are both a ``getitem`` off the engine node whose buffer is
    also an engine input, and rewiring a copy-back would silently drop a real
    update. An entry whose aliased *input* does not resolve -- the name is not one
    of the engine's input bindings, or its index is past the delegate's argument
    list -- is skipped rather than reported:
    ``_declare_aliased_kv_mutations_on_ep`` warns on both of those for the same
    engine, and neither leaves a mutation to rewire.
    """
    from torch_tensorrt.dynamo.runtime._serialized_engine_layout import (
        ALIASED_IO_IDX,
        INPUT_BINDING_NAMES_IDX,
        OUTPUT_BINDING_NAMES_IDX,
        deserialize_binding_names,
    )
    from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
        deserialize_aliased_io,
    )
    from torch_tensorrt.executorch._export_utils import _resolve_engine_info
    from torch_tensorrt.executorch.backend import _get_str

    # Only aliased_io and the binding names are read, never the engine itself.
    engine_info = _resolve_engine_info(
        exported_program, engine_node, metadata_only=True
    )
    aliased_io = deserialize_aliased_io(_get_str(engine_info, ALIASED_IO_IDX))
    if not aliased_io:
        return {}
    input_names = deserialize_binding_names(
        _get_str(engine_info, INPUT_BINDING_NAMES_IDX)
    )
    output_names = deserialize_binding_names(
        _get_str(engine_info, OUTPUT_BINDING_NAMES_IDX)
    )
    input_nodes = list(engine_node.args[0])

    aliased: Dict[int, Node] = {}
    for output_index, output_name in enumerate(output_names):
        entry = aliased_io.get(output_name)
        if entry is None:
            continue
        input_name = entry[0]
        if input_name not in input_names:
            continue
        input_index = input_names.index(input_name)
        if input_index >= len(input_nodes):
            continue
        aliased[output_index] = input_nodes[input_index]
    return aliased


def _engine_output_binding_names(exported_program: Any, engine_node: Node) -> List[str]:
    """Return one engine's output binding names, in binding (index) order.

    Resolved metadata-only: reading the record without that costs a full
    re-serialization of the engine through ``TRTEngine.__getstate__``, and only
    the binding names are wanted here. Callers that read this repeatedly for the
    same engine memoize it themselves -- ``_resolve_engine_info`` holds no cache.
    """
    from torch_tensorrt.dynamo.runtime._serialized_engine_layout import (
        OUTPUT_BINDING_NAMES_IDX,
        deserialize_binding_names,
    )
    from torch_tensorrt.executorch._export_utils import _resolve_engine_info
    from torch_tensorrt.executorch.backend import _get_str

    engine_info = _resolve_engine_info(
        exported_program, engine_node, metadata_only=True
    )
    names: List[str] = deserialize_binding_names(
        _get_str(engine_info, OUTPUT_BINDING_NAMES_IDX)
    )
    return names


class _AliasedMutation(NamedTuple):
    """One BUFFER_MUTATION an engine satisfies by writing the buffer in place."""

    placeholder: Node  # the buffer, as a graph input
    aliased_output: Node  # getitem(engine, i) currently standing in for it
    engine: Node  # the execute_engine call that performs the write


def _aliased_buffer_mutations(
    exported_program: Any,
) -> Dict[int, _AliasedMutation]:
    """Find the BUFFER_MUTATIONs an engine performs in place.

    Returns ``{index into graph_signature.output_specs: _AliasedMutation}``.
    A mutation qualifies only when its value is ``getitem(engine_node, i)`` and
    the engine declares output ``i`` as aliased onto that very buffer, so a
    buffer mutated by an op outside the engine, or copied back out of one, is
    left alone.
    """
    from torch.export.graph_signature import OutputKind

    graph_module = exported_program.graph_module
    signature = exported_program.graph_signature
    execute_engine = torch.ops.tensorrt.execute_engine.default

    buffer_placeholders = {
        fqn: node
        for node in graph_module.graph.nodes
        if node.op == "placeholder"
        and (fqn := signature.inputs_to_buffers.get(node.name)) is not None
    }
    output_args = list(graph_module.graph.output_node().args[0])
    aliased_by_engine: Dict[Node, Dict[int, Node]] = {}

    mutations: Dict[int, _AliasedMutation] = {}
    for spec_index, spec in enumerate(signature.output_specs):
        if spec.kind != OutputKind.BUFFER_MUTATION or spec_index >= len(output_args):
            continue
        placeholder = buffer_placeholders.get(spec.target)
        if placeholder is None:
            continue
        value = output_args[spec_index]
        if (
            not isinstance(value, Node)
            or value.op != "call_function"
            or value.target is not operator.getitem
        ):
            continue
        engine_node = value.args[0]
        if (
            not isinstance(engine_node, Node)
            or engine_node.op != "call_function"
            or engine_node.target is not execute_engine
        ):
            continue
        if engine_node not in aliased_by_engine:
            aliased_by_engine[engine_node] = _aliased_inputs_by_output_index(
                exported_program, engine_node
            )
        if aliased_by_engine[engine_node].get(value.args[1]) is placeholder:
            mutations[spec_index] = _AliasedMutation(
                placeholder=placeholder, aliased_output=value, engine=engine_node
            )
    return mutations


def rewire_aliased_mutations_to_buffers(exported_program: Any) -> List[str]:
    """Declare that an aliased buffer *is* its own mutation result.

    Export declares an aliased KV mutation as a ``getitem`` off the engine node:
    the engine's aliased output, surfaced as a value. ExecuTorch implements that
    mutation by copying the value back into the buffer, which is the copy this
    removes. Repointing the mutation at the buffer placeholder leaves nothing to
    copy, and with no other user the ``getitem`` dies -- so the aliased output
    also leaves the partition and the delegate never receives an argument for it.

    This must run before partitioning, because it is the partition boundary that
    freezes which outputs the delegate has. It must also run after export has
    declared the aliased mutations, since it works from those declarations; each
    placeholder it rewires is marked for
    :func:`unstage_aliased_buffers_pass`, which cannot re-derive the aliasing
    once lowering has turned the engine into an opaque blob.

    On its own this is not correct: ExecuTorch still stages the buffer, so the
    engine's in-place write would land in per-call scratch and, with the
    copy-back gone, be lost. It is only correct paired with the un-staging pass.

    Returns the engine output binding names of the aliased outputs it elided,
    one per rewired mutation. Only these names may later be exempted from the
    backend's output-binding check -- every *other* aliased output (a user alias
    on a plain, non-buffer input, which export never rewired) must still be a
    delegate output, so a delegate that dropped one of those as well is caught
    rather than silently writing that update into scratch. An engine that mixes
    the two kinds does not lower at all: ``TensorRTBackend.preprocess`` refuses
    it, because the runtime reads elision off a single argument count and so
    cannot express eliding only part of one engine's aliased outputs.
    """
    from torch.export.graph_signature import (
        ExportGraphSignature,
        OutputKind,
        OutputSpec,
        TensorArgument,
    )

    graph_module = exported_program.graph_module
    signature = exported_program.graph_signature
    mutations = _aliased_buffer_mutations(exported_program)
    if not mutations:
        logger.debug("no aliased buffer mutations to rewire")
        return []

    engines_with_elided_outputs: Set[Node] = set()
    output_names_by_engine: Dict[Node, List[str]] = {}
    elided_output_names: List[str] = []
    output_node = graph_module.graph.output_node()
    output_args = list(output_node.args[0])
    output_specs = list(signature.output_specs)
    for spec_index, mutation in mutations.items():
        # Marked on the node rather than read back off the engine because the
        # un-staging pass runs after lowering, where the engine's aliased_io is no
        # longer reachable from the graph: it has become an opaque delegate blob.
        mutation.placeholder.meta["_torch_tensorrt_aliased_buffer"] = True
        output_args[spec_index] = mutation.placeholder
        output_specs[spec_index] = OutputSpec(
            OutputKind.BUFFER_MUTATION,
            TensorArgument(name=mutation.placeholder.name),
            output_specs[spec_index].target,
        )
        engines_with_elided_outputs.add(mutation.engine)
        names = output_names_by_engine.get(mutation.engine)
        if names is None:
            names = _engine_output_binding_names(exported_program, mutation.engine)
            output_names_by_engine[mutation.engine] = names
        output_index = mutation.aliased_output.args[1]
        if 0 <= output_index < len(names):
            elided_output_names.append(names[output_index])

    output_node.args = (tuple(output_args),)
    graph_module.graph.eliminate_dead_code()
    # Leaving an engine with no output would leave its delegate with no outputs.
    # Nothing downstream reports that shape: the runtime infers elision from a
    # single argument count, which a zero-output delegate satisfies, and a
    # delegate nothing reads is a pure node that a later graph-wide dead-code
    # elimination can erase, taking the computation with it. This raise is what
    # stops that. It reads the graph after the elimination above rather than
    # before, so that an output kept alive only by a chain that is itself dead
    # does not count: the elimination erases such a chain however long it is,
    # and what survives it is what the delegate will really have. The engine
    # node itself survives even with no users -- PyTorch defaults an operator
    # taking a ScriptObject argument to an ORDERED effect
    # (torch._library.effects), and execute_engine takes the engine as one, so
    # FX reads it as impure where the delegate is not.
    for engine in engines_with_elided_outputs:
        if not engine.users:
            raise RuntimeError(
                "TensorRT zero-copy KV: eliding the aliased buffers engine node "
                f"'{engine.name}' writes in place leaves it with no output any "
                "node reads, so the delegate would have no outputs at all. This "
                "shape is not supported; export this method without "
                "zero_copy_kv."
            )
    graph_module.graph.lint()
    graph_module.recompile()
    # The signature is replaced in place rather than by rebuilding the program:
    # the graph has already been edited in place, and every other field would be
    # copied across unchanged.
    exported_program._graph_signature = ExportGraphSignature(
        input_specs=list(signature.input_specs), output_specs=output_specs
    )
    logger.debug(
        "rewired %d aliased mutation(s) to their buffers, eliding outputs %s",
        len(mutations),
        elided_output_names,
    )
    return elided_output_names


def _mutation_targets_a_lifted_input(signature: Any) -> Set[str]:
    """The mutation targets ``insert_write_back_for_buffers_pass`` can resolve.

    Mirrors the ``lifted_inputs`` map that pass builds: a buffer, constant,
    parameter or custom object contributes its ``target``, a user input its
    argument name. A mutation whose target is not in here gets no copy either,
    so it belongs with the copy-free ones.
    """
    from torch.export.graph_signature import InputKind, TensorArgument

    lifted: Set[str] = set()
    for spec in signature.input_specs:
        if spec.kind in (
            InputKind.BUFFER,
            InputKind.CONSTANT_TENSOR,
            InputKind.PARAMETER,
            InputKind.CUSTOM_OBJ,
        ):
            if spec.target is not None:
                lifted.add(spec.target)
        elif spec.kind is InputKind.USER_INPUT and isinstance(spec.arg, TensorArgument):
            lifted.add(spec.arg.name)
    return lifted


def order_copyback_mutations_first(exported_program: Any) -> int:
    """Reorder one Edge method's mutations so ExecuTorch pairs them up correctly.

    ExecuTorch finalizes a mutation by inserting a ``copy_`` for it, but only
    when its target is one of the lifted inputs and its value is not already
    reached, through in-place ops, from a placeholder of the mutation's own kind
    -- any buffer placeholder for a buffer mutation, not necessarily the one it
    targets. It moves those copies to the front of the output tuple, leaves
    everything else behind them in order, and then walks the mutation specs
    reassigning each one's argument *by position* over the result
    (``insert_write_back_for_buffers_pass``).
    :func:`rewire_aliased_mutations_to_buffers` makes a mutation's value its own
    placeholder, so a rewired cache gets no copy and drops out of that leading
    run -- and in a method that also has a copy-back buffer, every mutation spec
    from the first copy-free one on then comes out of finalization naming a
    different buffer's value. The specs ahead of it are unaffected, since the
    copies keep their order among themselves. The ``.pte`` is written correctly
    either way, because the emitter and the memory planner read only which
    buffers are mutated and not what by. What the pairing decides is the
    finalized signature, which anyone inspecting the program reads, and
    ExecuTorch's eager call path, which walks ``buffers_to_mutate`` writing the
    graph's leading results into the state dict in that order and so updates
    each buffer from another one's value.

    Zero-copy is not the only way in, which is why ``export()`` runs this over
    every method rather than only the rewired ones. A plain ``nn.Module`` that
    writes one buffer from another buffer -- whose mutation value is then that
    other buffer's placeholder -- and a second buffer from a user input comes out
    of stock ``to_edge().to_executorch()``, with no TensorRT anywhere, with the
    first buffer's mutation spec naming the copy that writes the second.

    Putting the mutations that still get a copy first restores the
    correspondence. Which ones those are is decided by asking upstream's own
    predicate (``_inplace_lineage``, imported rather than reimplemented) rather
    than by asking which mutations this module rewired, so the answer cannot
    drift from the one the write-back pass will give, and any mutation the graph
    already presents as in-place is covered whatever put it there.

    Only what the graph presents *here* is covered, though. ``run_reinplace_pass``
    and ``reinplace_extra_ops`` are supported ``ExecutorchBackendConfig`` fields
    whose pass runs inside ``to_executorch``, after this and immediately before
    the write-back: a mutation it rewrites is ordinary when this reads it and
    in-place when the write-back does, so that pair comes out crossed anyway and
    this reports nothing moved. Reordering cannot reach it from here. The same
    crossing reproduces on a model using no zero-copy at all, so it is a pass
    ordering upstream owns rather than one this creates.

    This runs on the *Edge* program rather than beside the rewiring, because
    ``to_edge_transform_and_lower`` re-derives the whole graph signature -- the
    order it hands back is the buffers' own order, whatever order it was given.
    Nothing between here and the write-back pass re-derives it again.

    Returns the number of mutation slots whose value changed, which is zero when
    the order already holds.
    """
    from executorch.exir.passes.insert_write_back_for_buffers_pass import (
        _inplace_lineage,
    )
    from torch.export.graph_signature import ExportGraphSignature, OutputKind

    signature = exported_program.graph_signature
    specs = list(signature.output_specs)
    output_node = exported_program.graph_module.graph.output_node()
    args = list(output_node.args[0])
    slots = [
        index
        for index, spec in enumerate(specs)
        if spec.kind in (OutputKind.BUFFER_MUTATION, OutputKind.USER_INPUT_MUTATION)
        and index < len(args)
    ]
    if not slots:
        return 0
    lifted = _mutation_targets_a_lifted_input(signature)

    def gets_a_copy(index: int) -> bool:
        value = args[index]
        if not isinstance(value, Node):
            # Upstream reads a non-Node value as needing a copy, and then raises
            # walking it. Grouping it with the copies keeps this reorder from
            # being what raises first.
            return True
        if specs[index].target not in lifted:
            return False
        return not _inplace_lineage(value, signature, specs[index].kind)

    copied = {index for index in slots if gets_a_copy(index)}
    source = [index for index in slots if index in copied] + [
        index for index in slots if index not in copied
    ]
    if source == slots:
        return 0

    new_args, new_specs = list(args), list(specs)
    for slot, index in zip(slots, source):
        new_args[slot] = args[index]
        new_specs[slot] = specs[index]
    output_node.args = (tuple(new_args),)
    exported_program.graph_module.recompile()
    exported_program._graph_signature = ExportGraphSignature(
        input_specs=list(signature.input_specs), output_specs=new_specs
    )
    moved = sum(1 for slot, index in zip(slots, source) if slot != index)
    logger.debug("moved %d mutation(s) so the copy-back ones come first", moved)
    return moved


def _is_tensorrt_delegate(graph_module: torch.fx.GraphModule, node: Node) -> bool:
    """True when ``node`` is a call_delegate dispatching to the TensorRT backend.

    Only a TensorRT engine promises the aliased-binding write; another backend's
    delegate may legitimately need the staging copy.
    """
    from executorch.exir.delegate import executorch_call_delegate
    from torch_tensorrt.executorch.backend import TensorRTBackend

    if node.op != "call_function" or node.target is not executorch_call_delegate:
        return False
    lowered = node.args[0] if node.args else None
    if not isinstance(lowered, Node) or lowered.op != "get_attr":
        return False
    module = getattr(graph_module, lowered.target, None)
    return bool(getattr(module, "backend_id", None) == TensorRTBackend.__name__)


def _zero_copy_compile_spec(graph_module: torch.fx.GraphModule, node: Node) -> Any:
    """One delegate's zero-copy KV compile spec, or ``None`` if it carries none."""
    from torch_tensorrt.executorch.backend import ZERO_COPY_KV_COMPILE_SPEC_KEY

    lowered = node.args[0] if node.args else None
    if not isinstance(lowered, Node) or lowered.op != "get_attr":
        return None
    module = getattr(graph_module, lowered.target, None)
    for spec in getattr(module, "compile_specs", None) or []:
        if getattr(spec, "key", None) == ZERO_COPY_KV_COMPILE_SPEC_KEY:
            return spec
    return None


def _delegate_elided_output_names(
    graph_module: torch.fx.GraphModule, node: Node
) -> Set[str]:
    """The aliased output binding names one delegate's zero-copy spec claims.

    ``TensorRTPartitioner`` writes one name per aliased output it elided on that
    delegate's own engine, and an aliased output is elided exactly when a buffer
    mutation writes it in place, so the size of this set is how many marked
    buffers the delegate has to take. Empty when the delegate carries no such
    spec, when it carries one whose value does not decode into a list of names,
    and when that value decodes into an empty list -- the last two are shapes a
    spec built by hand produces, since the partitioner writes one JSON name per
    aliased output it elided. Callers read empty as "cannot tell", not as "none".

    ``backend._elided_output_names`` reads the same key on the same spec and
    takes the same shapes of value. It differs in what it does with the rest: it
    raises naming the key, because an undecodable spec leaves it unable to say
    which outputs the delegate was allowed to drop, while here it only weakens
    the two cross-checks that read it, both of which then fall back to demanding
    at least one buffer. The one value the two read differently is bytes that
    are not valid UTF-8: it replaces the bad units and reads a name out of them,
    ``json.loads`` refuses them here, and here that is another "cannot tell".
    """
    spec = _zero_copy_compile_spec(graph_module, node)
    if spec is None:
        return set()
    value = getattr(spec, "value", None)
    if not isinstance(value, (str, bytes, bytearray)):
        return set()
    try:
        names = json.loads(value)
    except ValueError:
        return set()
    return {str(name) for name in names} if isinstance(names, list) else set()


def _delegate_declares_zero_copy(
    graph_module: torch.fx.GraphModule, node: Node
) -> bool:
    """True when a TensorRT delegate carries the zero-copy KV compile spec.

    ``TensorRTPartitioner`` stamps this spec per partition, onto only the delegate
    whose own engine had an aliased output elided (derived per engine in
    ``TensorRTPartitioner._partition_elided_output_names``), so a delegate that
    declares it must have had a buffer un-staged here. A method that lowers to
    several TensorRT delegates therefore marks only the KV one, never the plain
    compute engines beside it -- which is what keeps this cross-check from
    demanding an aliased buffer from a delegate that never had one. A delegate
    that declares it but ends up taking fewer marked buffers than it elided
    aliased outputs has lost a KV update -- the mark that would have driven the
    un-staging did not survive, or another delegate took the buffer -- and is
    caught in :func:`_unstage_aliased_buffers` before planning and again in
    :func:`check_zero_copy_kv` after it.
    """
    return _zero_copy_compile_spec(graph_module, node) is not None


def _device_placement_is_safe(
    graph_module: torch.fx.GraphModule,
    source: Node,
    h2d_copy: Any,
    target_device: Any,
    target_device_index: Any,
    already_removed: Any = (),
) -> bool:
    """True when every other consumer of ``source`` survives its device placement.

    A placeholder's device is shared by every user, so it can only be planned in
    the delegate's device memory when nothing else *reads* it from somewhere
    else. ExecuTorch guards the same hazard, more strictly and only under its
    opt-in ``skip_h2d_for_method_inputs``: it demands the placeholder have
    exactly one user. The rule here is looser because some users survive
    unaffected and are allowed: the graph ``output`` node -- the buffer is its
    own BUFFER_MUTATION result, which is exactly what zero-copy sets up and
    which carries no device of its own -- an ``_h2d_copy`` to the same GPU that
    this pass removes, because every one of its users is a TensorRT delegate
    whose argument the pass rewires to the buffer, and a TensorRT delegate
    already taking the buffer itself, which is the shape this pass leaves behind
    and so what a second marked delegate, or a second run, finds.

    ``already_removed`` is the last of them: the staging copies this run has
    detached, which are still in the graph because they are erased only once the
    walk has succeeded. A detached one has no users left, which is the shape a
    *foreign* dead copy has too, and that one is refused -- nothing erases it,
    so the emitter keeps it and it reads device memory as a host source. Ours
    are named here rather than inferred, so the two cannot be confused.

    A staging copy that outlives the pass is *not* allowed, even on the same
    GPU. It would go on reading the buffer as its source once the buffer is in
    device memory, and ``_h2d_copy_out`` requires a host source: the portable
    kernel checks it and fails ``InvalidArgument``.

    The index is compared as well as the type, because ``spec.device`` is only
    ``CUDA``/``CPU``: two engines resolved to ``cuda:0`` and ``cuda:1`` stage the
    same buffer to different GPUs, and un-staging both would leave whichever ran
    last owning the buffer while the other engine writes an address on the wrong
    device.

    This is asked whether or not the buffer's spec already names the target
    device. Where it does, the pass changes no device and the surviving copy is
    already reading device memory as a host source -- a program that was broken
    before this pass touched it -- but the pass is about to hand that buffer to
    an engine on the strength of the same post-condition, so it is refused here
    rather than left to fail on first execution.
    """
    for user in source.users:
        if user.op == "output":
            continue
        if not isinstance(user, Node) or user.op != "call_function":
            return False
        if user in already_removed or _is_tensorrt_delegate(graph_module, user):
            continue
        if user.target is not h2d_copy:
            return False
        spec = user.meta.get("spec")
        if spec is None or spec.device != target_device:
            return False
        if spec.device_index != target_device_index:
            return False
        if not user.users or not all(
            _is_tensorrt_delegate(graph_module, copy_user) for copy_user in user.users
        ):
            return False
    return True


def _unstage_aliased_buffers(
    graph_module: torch.fx.GraphModule, *, device_memory_planning: bool = True
) -> int:
    """Route TensorRT delegate inputs from their staging copy back to the buffer.

    An input is un-staged only when it is an ``_h2d_copy`` of a placeholder
    carrying the mark left by :func:`rewire_aliased_mutations_to_buffers`. Every
    other input keeps its staging, including a mutable buffer the engine does
    not write in place.

    The placeholder's spec takes over the staging copy's device, which is what
    asks memory planning for the delegate's device arena rather than a host one
    (asks, not settles -- see ``device_memory_planning`` below). That is what
    makes handing the buffer straight to the engine valid at all: a host-arena
    pointer is not something the engine can write. It is refused when the buffer
    has a consumer this pass leaves behind that does not survive being handed
    the buffer from there (see :func:`_device_placement_is_safe`) -- asked on
    both routes below and whether or not the spec already names that device,
    since it is the placement and not the change of device that such a consumer
    does not survive. A buffer already reaching its delegate directly is in the
    same position as one this pass moves there, and is the shape this pass's own
    first run leaves for its second.

    What the pass has to establish is the *post-condition*: every marked buffer
    is a direct argument of a TensorRT delegate *declaring zero-copy KV* -- one
    whose own engine elided an aliased output -- *and* ends up planned in device
    memory. Removing a staging copy is only the usual way of getting there, not
    the goal, and a marked buffer that already satisfies both is left alone and
    counts as satisfied -- which is what running this pass a second time over a
    program it has already un-staged finds.

    The zero-copy declaration is what narrows the delegates that count, on both
    routes, because the mark says an engine writes the buffer in place and only a
    stamped delegate's engine did. An unstamped TensorRT delegate taking the
    buffer proves nothing about the stamped one, whose write would still go to a
    staging copy that is discarded. :func:`check_zero_copy_kv` narrows the same
    way over the finalized program, so the two halves of this post-condition
    accept and refuse the same graphs.

    Being a direct argument is not on its own enough, so it is not on its own
    accepted. Two things decide where the buffer is planned, and the second is
    not visible in the graph: the spec's own device, and whether memory planning
    reads spec devices at all. ``enable_non_cpu_memory_planning=False`` on the
    ``ExecutorchBackendConfig`` plans every tensor into the one host arena
    whatever its spec says, so the engine is handed a host pointer for a buffer
    it must write on the device -- and it does that to every marked buffer, the
    ones this pass un-stages as much as the ones that already reach their
    delegate directly. That is why it is checked once for the whole graph, up
    front, rather than on one of the two branches below.
    ``device_memory_planning`` carries the configuration in; the pass
    :func:`unstage_aliased_buffers_pass` builds reads it off the finalization
    config when it runs, resolved as :func:`_config_plans_on_devices` describes,
    so a configuration whose planner never receives the flag is not refused on
    it.

    A failure here is a lost KV update, so it is raised rather than logged:
    export has already removed the copy-back, so a marked buffer left staged has
    the engine write per-call scratch that is then discarded and the buffer never
    updates. It raises when memory planning is host-only, when either end of the
    move -- the staging copy or the buffer -- has no spec, when the staging copy
    is not on CUDA, when a direct argument has no spec of its own
    or that spec is not on CUDA, when the placement is unsafe, and -- so a
    discovery miss cannot pass silently -- after the loop when the post-condition
    does not hold for some marked buffer, cross-checked against each delegate's
    own ``zero_copy_kv`` spec: a TensorRT delegate that takes fewer marked
    buffers than the aliased outputs that spec says it elided is broken, and that
    refusal names the delegate and lists what it does take.

    Returns the number of delegate inputs un-staged, which is zero for a program
    that already satisfied the post-condition.
    """
    from executorch.exir.schema import DeviceType
    from torch_tensorrt.executorch.backend import ZERO_COPY_KV_COMPILE_SPEC_KEY

    marked_placeholders = [
        node
        for node in graph_module.graph.nodes
        if node.op == "placeholder" and node.meta.get("_torch_tensorrt_aliased_buffer")
    ]
    if marked_placeholders and not device_memory_planning:
        names = ", ".join(repr(node.name) for node in marked_placeholders)
        raise RuntimeError(
            "TensorRT zero-copy KV: buffer(s) "
            f"{names} are marked for in-place update, but memory planning is "
            "configured with enable_non_cpu_memory_planning=False, which puts "
            "every tensor in the one host arena whatever its TensorSpec says. "
            "The engine would be handed a host pointer it cannot write, and "
            "export has already removed the copy-back, so the update would be "
            "lost. Finalize over a configuration that leaves "
            "enable_non_cpu_memory_planning on, so a CUDA TensorSpec is given a "
            "CUDA arena."
        )

    h2d_copy = torch.ops.et_copy._h2d_copy.default
    unstaged = 0
    satisfied_placeholders: Set[Node] = set()
    # A dict rather than a list: the walk asks whether a staging copy has
    # already been detached, and the erase below wants each one once.
    orphaned_stagings: Dict[Node, None] = {}
    zero_copy_delegates: List[Node] = []
    # Distinct buffers, not argument slots: one buffer occupying two of a
    # delegate's slots is one cache written in place, and counting the slots
    # would let it satisfy a spec naming two elided outputs. The finalized-program
    # check counts the same way, so the two cannot disagree about one graph.
    satisfied_per_delegate: Dict[Node, Set[Node]] = {}

    for node in list(graph_module.graph.nodes):
        if not _is_tensorrt_delegate(graph_module, node):
            continue
        declares_zero_copy = _delegate_declares_zero_copy(graph_module, node)
        if declares_zero_copy:
            zero_copy_delegates.append(node)
            satisfied_per_delegate[node] = set()
        new_args = list(node.args)
        for i, arg in enumerate(node.args[1:], start=1):
            if not isinstance(arg, Node):
                continue
            if arg.op == "placeholder":
                if arg.meta.get("_torch_tensorrt_aliased_buffer"):
                    direct_spec = arg.meta.get("spec")
                    placement = ""
                    remedy = ""
                    if direct_spec is None:
                        placement = "it carries no TensorSpec"
                        remedy = (
                            "The specs exist only while this runs as the "
                            "ExecutorchBackendConfig to_out_var_pass, which is "
                            "where torch_tensorrt.executorch."
                            "zero_copy_backend_config() installs it."
                        )
                    elif direct_spec.device != DeviceType.CUDA:
                        placement = f"its TensorSpec asks for {direct_spec.device!r}"
                        remedy = (
                            "Give the buffer to a TensorRT delegate on CUDA, or "
                            "export this method without zero_copy_kv."
                        )
                    if placement:
                        raise RuntimeError(
                            "TensorRT zero-copy KV: buffer "
                            f"'{arg.name}' reaches a TensorRT delegate directly, "
                            f"with no staging copy to remove, but {placement}, so "
                            "it is not planned in device memory and the engine is "
                            "handed a host pointer it cannot write. The engine "
                            "writes this buffer in place and its copy-back has "
                            "already been removed, so the update would be lost. "
                            f"{remedy}"
                        )
                    if not _device_placement_is_safe(
                        graph_module,
                        arg,
                        h2d_copy,
                        direct_spec.device,
                        direct_spec.device_index,
                        orphaned_stagings,
                    ):
                        raise RuntimeError(
                            "TensorRT zero-copy KV: buffer "
                            f"'{arg.name}' reaches a TensorRT delegate directly "
                            "and is read by a consumer that does not survive it "
                            "being planned in this engine's device memory -- a "
                            "staging copy left behind reads it as a host source "
                            "and fails InvalidArgument. Export this method "
                            "without zero_copy_kv, or stop sharing the aliased "
                            "buffer."
                        )
                    if declares_zero_copy:
                        satisfied_placeholders.add(arg)
                        satisfied_per_delegate[node].add(arg)
                continue
            if arg.target is not h2d_copy:
                continue
            source = arg.args[0]
            if not isinstance(source, Node) or source.op != "placeholder":
                continue
            if not source.meta.get("_torch_tensorrt_aliased_buffer"):
                continue  # not written in place; it needs its staging copy
            staged_spec = arg.meta.get("spec")
            source_spec = source.meta.get("spec")
            if staged_spec is None or source_spec is None:
                missing = (
                    f"the staging copy '{arg.name}'"
                    if staged_spec is None
                    else f"the buffer placeholder '{source.name}'"
                )
                raise RuntimeError(
                    f"TensorRT zero-copy KV: no TensorSpec on {missing}, so buffer "
                    f"'{source.name}' cannot be moved to the delegate's device. The "
                    "TensorRT engine writes this buffer in place and its copy-back "
                    "has already been removed, so the update would be lost. This "
                    "pass has to run as the ExecutorchBackendConfig to_out_var_pass, "
                    "which is where the specs exist; "
                    "torch_tensorrt.executorch.zero_copy_backend_config installs it "
                    "there."
                )
            # spec.device is an exir schema DeviceType, not a torch.device.
            if staged_spec.device != DeviceType.CUDA:
                raise RuntimeError(
                    "TensorRT zero-copy KV: the staging copy of buffer "
                    f"'{source.name}' targets device {staged_spec.device!r}, not "
                    "CUDA, so moving the buffer there would put it where the "
                    "TensorRT engine cannot write it. The engine writes this "
                    "buffer in place and its copy-back has already been removed, "
                    "so the update would be lost."
                )
            if not _device_placement_is_safe(
                graph_module,
                source,
                h2d_copy,
                staged_spec.device,
                staged_spec.device_index,
                orphaned_stagings,
            ):
                raise RuntimeError(
                    "TensorRT zero-copy KV: buffer "
                    f"'{source.name}' is read by a consumer this pass leaves in "
                    "place that does not survive the buffer being planned in this "
                    "engine's device memory -- a staging copy left behind reads it "
                    "as a host source and fails InvalidArgument. Export this "
                    "method without zero_copy_kv, or stop sharing the aliased "
                    "buffer."
                )
            source_spec.device = staged_spec.device
            source_spec.device_index = staged_spec.device_index
            new_args[i] = source
            unstaged += 1
            orphaned_stagings[arg] = None
            if declares_zero_copy:
                satisfied_placeholders.add(source)
                satisfied_per_delegate[node].add(source)
        node.args = tuple(new_args)

    marked_but_unsatisfied = [
        node for node in marked_placeholders if node not in satisfied_placeholders
    ]
    if marked_but_unsatisfied:
        names = ", ".join(repr(node.name) for node in marked_but_unsatisfied)
        raise RuntimeError(
            "TensorRT zero-copy KV: buffer(s) "
            f"{names} were marked for in-place update but no TensorRT delegate "
            f"declaring zero-copy KV (compile spec "
            f"'{ZERO_COPY_KV_COMPILE_SPEC_KEY}') takes them, either directly or "
            "through a staging copy this pass could remove. Export removed their "
            "copy-back on the promise that a TensorRT engine writes them in "
            "place, so as this program stands nothing updates them. An unstamped "
            "TensorRT delegate taking the buffer does not count: the stamp is "
            "what records that an engine elided its aliased output for it. "
            "Export this method without zero_copy_kv, or keep the aliased buffer "
            "on the delegate whose engine elided it."
        )
    for delegate in zero_copy_delegates:
        # One marked buffer per aliased output the spec says this delegate
        # elided. Demanding only one would accept a delegate that lost all but
        # one of its marks, whose remaining caches are still wired through their
        # staging copies. A spec that names none -- listing none, or not
        # decoding -- cannot say how many to expect, so it falls back to
        # demanding at least one.
        elided = _delegate_elided_output_names(graph_module, delegate)
        satisfied = len(satisfied_per_delegate[delegate])
        if satisfied >= max(len(elided), 1):
            continue
        delegate_inputs = [
            arg.name for arg in delegate.args[1:] if isinstance(arg, Node)
        ]
        expectation = (
            "takes no buffer marked for in-place update"
            if not elided
            else (
                f"elided the aliased output(s) {sorted(elided)} but takes only "
                f"{satisfied} of the {len(elided)} buffers that implies"
            )
        )
        raise RuntimeError(
            "TensorRT zero-copy KV: delegate "
            f"'{delegate.name}' declares zero-copy KV "
            f"(compile spec '{ZERO_COPY_KV_COMPILE_SPEC_KEY}') but "
            f"{expectation} (inputs: {delegate_inputs}). "
            "Export elided its aliased outputs, so the engine now writes "
            "per-call scratch that is discarded and those caches never update."
        )

    if unstaged:
        # Erase only the stagings we orphaned. A graph-wide eliminate_dead_code()
        # in a to_out_var_pass could delete another backend's unused delegate.
        for staging in orphaned_stagings:
            if not staging.users:
                graph_module.graph.erase_node(staging)
        graph_module.graph.lint()
        graph_module.recompile()
    return unstaged


def _config_plans_on_devices(config: "ExecutorchBackendConfig") -> bool:
    """Whether ``enable_non_cpu_memory_planning`` is anything this config decides by.

    ``to_executorch`` does not hand the flag to the memory planner it is given.
    It *assigns* it, and only onto a planner that already has an attribute of
    that name (``exir/program/_program.py``), which in practice means
    ``MemoryPlanningPass`` or a subclass. A caller who brings their own planner
    -- which the user guide tells people to do for a cache shared between
    prefill and decode -- never receives the flag, and what that planner does
    with the specs is its own business: ``False`` does not put the caches in the
    host arena, and ``True`` does not keep them out of it. So the flag answers
    nothing for such a config and ``True`` is returned, meaning only "not a
    ground to refuse on"; :func:`check_zero_copy_kv` reads the arena that
    planner actually chose, which is the answer this cannot give.

    ``memory_planning_pass`` may also be a per-method dict, and this pass sees
    one graph module without its method name, so a dict is read as deciding
    nothing unless every planner in it takes the flag. A method the dict omits
    gets ExecuTorch's default planner, which does.
    """
    planner = config.memory_planning_pass
    planners = list(planner.values()) if isinstance(planner, dict) else [planner]
    if not all(hasattr(p, "enable_non_cpu_memory_planning") for p in planners):
        return True
    return bool(config.enable_non_cpu_memory_planning)


class _UnstageThenToOutVar(PassBase):  # type: ignore[misc]
    """The ``to_out_var_pass`` :func:`unstage_aliased_buffers_pass` builds.

    A module-level named type rather than one built inside the builder, so that
    a config can be *recognised*: a class made afresh on every call gives a
    different type object each time, and two configs this module produced would
    then carry passes with nothing in common to match on. As it is,
    ``isinstance(config.to_out_var_pass, _UnstageThenToOutVar)`` tells a config
    that un-stages from one that does not.

    ``inner`` is the ``to_out_var_pass`` that would otherwise have run.
    ``device_memory_planning`` is what an unbound pass reads; once
    ``finalization_config`` is set, the flag is resolved off that config on
    every call instead, as :func:`_config_plans_on_devices` describes.
    """

    def __init__(self, inner: Any, device_memory_planning: bool) -> None:
        self.inner = inner
        self.device_memory_planning = device_memory_planning
        self.finalization_config: Optional["ExecutorchBackendConfig"] = None

    def call(self, graph_module: torch.fx.GraphModule) -> Any:
        config = self.finalization_config
        planning = (
            self.device_memory_planning
            if config is None
            else _config_plans_on_devices(config)
        )
        unstaged = _unstage_aliased_buffers(
            graph_module, device_memory_planning=planning
        )
        logger.debug("un-staged %d aliased delegate buffer(s)", unstaged)
        return self.inner(graph_module)


def unstage_aliased_buffers_pass(
    inner_pass: Optional[Any] = None, *, device_memory_planning: bool = True
) -> Any:
    """Build a ``to_out_var_pass`` that un-stages aliased buffers, then delegates.

    ``to_out_var_pass`` is the last hook that runs after ``PropagateDevicePass``
    and before memory planning -- the window in which the staging copies exist
    and the buffers' placement is not yet fixed. (``sym_shape_eval_pass`` is a
    caller-supplied hook in that window too, but it runs first.)

    ``inner_pass`` is the ``to_out_var_pass`` that would otherwise have run; it
    runs after the un-staging. Omit it for ExecuTorch's default.

    ``device_memory_planning`` is the ``enable_non_cpu_memory_planning`` the
    program will be finalized with. Nothing in the graph records it, and it is
    half of what decides whether a marked buffer ends up somewhere the engine can
    write, so the pass has to be told: see :func:`_unstage_aliased_buffers`.
    It is what a pass built here and left unbound uses. Set
    ``finalization_config`` on the returned pass to the
    ``ExecutorchBackendConfig`` the program will be finalized with and the pass
    reads the flag off that config instead, on every call, as
    :func:`_config_plans_on_devices` resolves it -- which is what memory planning
    will do with it a few passes later, and is nothing at all when the config
    carries a planner that does not take the flag.
    ``ExecutorchBackendConfig`` is a plain mutable dataclass, so the field can be
    set again on the very config being finalized after the pass was built; a pass
    reading a value captured here would then accept a program the finalizer plans
    into the host arena. :func:`zero_copy_backend_config` binds the attribute for
    that reason, and passes no ``device_memory_planning`` at all, since the
    binding would override it on every call.
    """
    from executorch.exir import ExecutorchBackendConfig

    inner = (
        inner_pass
        if inner_pass is not None
        else ExecutorchBackendConfig().to_out_var_pass
    )
    return _UnstageThenToOutVar(inner, device_memory_planning)


def _device_planned_arenas(
    graph_module: torch.fx.GraphModule,
) -> Optional[Dict[int, Any]]:
    """The finalized program's CUDA arenas, ``mem_id -> device index``, or ``None``.

    Memory planning partitions the specs by device, gives each device its own
    arena, and records the non-CPU ones on the graph module as
    ``non_const_buffer_device``. ``None`` means the program records no arena
    devices at all, which does *not* mean the host: ``apply_algo`` is the only
    thing in ExecuTorch that writes the key, and ``to_executorch`` accepts any
    callable as ``memory_planning_pass``, so a caller-supplied planner -- which
    the user guide tells people to bring for a cache shared between prefill and
    decode -- can plan onto a device and still leave the key unwritten.

    The index is carried, not only the type, because the record is what the
    runtime allocates from: an arena recorded for ``cuda:1`` holding a cache the
    engine writes on ``cuda:0`` is a pointer on the wrong GPU, which the device
    *type* alone cannot tell from a correct program. ``apply_algo`` writes one
    entry per arena, so a ``mem_id`` names at most one device.
    """
    from executorch.exir.schema import DeviceType

    entries = graph_module.meta.get("non_const_buffer_device")
    if not entries:
        return None
    return {
        entry.buffer_idx: getattr(entry, "device_index", None)
        for entry in entries
        if getattr(entry, "device_type", None) == DeviceType.CUDA
    }


def _host_planned_arenas(graph_module: torch.fx.GraphModule) -> Set[int]:
    """The ``mem_id``s of arenas that hold at least one host tensor.

    Planning gives each device its own arena, so an arena holding a tensor whose
    spec is CPU is a host arena -- an argument from the graph rather than from
    the planner's records, which is what makes it usable when those records are
    absent. It is what separates the two shapes that both record no arena
    devices: ``enable_non_cpu_memory_planning=False`` puts every tensor in one
    bucket, so a CUDA-spec cache ends up sharing an arena with the host tensors,
    while a caller-supplied device-aware planner keeps them apart.

    A method with no CPU tensor at all is the residual: nothing here can then
    tell a device arena from a host one, and if the program records no arena
    devices either, the buffer is accepted.
    """
    from executorch.exir.schema import DeviceType

    host: Set[int] = set()
    for node in graph_module.graph.nodes:
        specs = node.meta.get("spec")
        for spec in specs if isinstance(specs, (list, tuple)) else [specs]:
            if (
                spec is not None
                and getattr(spec, "device", None) == DeviceType.CPU
                and getattr(spec, "mem_id", None) is not None
            ):
                host.add(spec.mem_id)
    return host


def _is_host_planned(
    node: Node, device_arenas: Optional[Dict[int, Any]], host_arenas: Set[int]
) -> bool:
    """True when memory planning put ``node`` somewhere the engine cannot write.

    Two independent grounds, because either record may be the only one there.
    Sharing an arena with a host tensor settles it whatever the program records;
    otherwise, when the program does record its arena devices, an arena missing
    from that record is not a CUDA one.

    Asked only for a buffer that has a ``mem_id``. One that does not was never
    planned, which is not a placement this can read and is refused as its own
    thing by the caller.
    """
    mem_id = node.meta["spec"].mem_id
    return mem_id in host_arenas or (
        device_arenas is not None and mem_id not in device_arenas
    )


def _planned_on_another_gpu(
    node: Node, device_arenas: Optional[Dict[int, Any]]
) -> Optional[str]:
    """How ``node``'s arena and its own spec disagree about which GPU, if they do.

    Being in a CUDA arena is not enough: the engine writes the cache through the
    pointer the runtime allocates out of that arena, so an arena the program
    records for another GPU is an address the engine cannot write, exactly as a
    host one is. Only a disagreement between two recorded indices is read as
    one -- either side left unrecorded says nothing, and the arena's own device
    type has already been established by the caller.
    """
    if device_arenas is None:
        return None
    spec = node.meta.get("spec")
    mem_id = getattr(spec, "mem_id", None)
    arena_index = None if mem_id is None else device_arenas.get(mem_id)
    spec_index = getattr(spec, "device_index", None)
    if arena_index is None or spec_index is None or arena_index == spec_index:
        return None
    return (
        f"'{node.name}' asks for cuda:{spec_index} and was planned in an arena "
        f"the program records as cuda:{arena_index}"
    )


def _name_detail(names_by_method: Dict[str, List[str]]) -> str:
    return ", ".join(
        f"'{name}' in method '{method}'"
        for method, names in names_by_method.items()
        for name in names
    )


def check_zero_copy_kv(program: Any) -> None:
    """Raise unless a finalized program really updates its KV buffers in place.

    ``program`` is what ``to_executorch()`` returns. Both halves of zero-copy do
    nothing quietly when they find nothing to do: ``zero_copy_kv=True`` warns and
    carries on when the model holds no aliased buffer mutation, and the pass
    :func:`unstage_aliased_buffers_pass` builds, handed a program with nothing
    marked, un-stages nothing and returns. Neither of those is wrong output -- nothing
    removed a copy-back, so the ``.pte`` stages its cache and updates it like any
    other -- but the optimization the caller asked for is silently not there, and
    a caller who reads the successful ``save`` as proof it is gets neither an
    error nor the speedup. The wrong-output case is the one below: a rewiring
    that did happen and then lost its mark.

    Six shapes are refused: a marked buffer that is not a direct argument of a
    TensorRT delegate carrying the zero-copy compile spec, a stamped delegate
    that takes fewer marked buffers than its own spec says it elided aliased
    outputs -- including one in a method with no marked buffer at all, which is
    the lost-mark case -- a marked buffer that reaches such a delegate directly
    but is planned in a host arena, one whose placement the program records
    nowhere, one planned in a device arena the program records for another GPU,
    and a program carrying neither a marked buffer nor a stamped delegate in any
    of its methods. The first three are
    what finalizing without :func:`zero_copy_backend_config` leaves behind --
    all but the lost-mark case folded into the second, which no finalization
    choice produces. That config's pass gets to all three earlier, off the
    configuration and the graph: it removes the staging copy the first two come
    from, and refuses outright when there is none to remove or when the
    configuration plans nothing onto a device. What it cannot see is the arena
    planning then chose, which is what this reads.

    The spec is what narrows the delegates that count. The mark is put on a
    buffer because one TensorRT engine writes it in place, and only a delegate
    whose own engine elided an aliased output is stamped, so another backend's
    delegate taking the buffer says nothing about whether the engine did, and
    neither does an unrelated TensorRT engine that happens to read it. Either
    would stand in for the delegate whose write was elided while that one still
    reads a staging copy whose contents are discarded. Being stamped is not
    enough on its own either: in a method that lowers to two of them, one
    stamped delegate holding both caches leaves every marked buffer reaching
    *some* stamped delegate while the other's write is still thrown away. So
    each stamped delegate is also counted against its own spec, the same
    cross-check :func:`_unstage_aliased_buffers` makes, against the same spec and
    with the same fallback: a spec that names none -- listing none, or not
    decoding -- cannot say how many to expect, so it demands at least one. What
    that count cannot separate is an exact swap -- two stamped delegates each
    taking one marked buffer, each the other's. The mark records only that some
    engine writes the buffer in place, never which one, and the spec lists
    engine output binding names rather than buffers, so there is nothing left to
    match on; that pass has the same blind spot for the same reason.

    Placement is read from where memory planning actually put the buffer -- the
    ``mem_id`` on its ``TensorSpec`` -- rather than from the spec's own device,
    which does not settle it: ``PropagateDevicePass`` writes CUDA onto the spec
    of a buffer that reaches a CUDA delegate directly, and
    ``enable_non_cpu_memory_planning=False`` then plans that same buffer into the
    one host arena, which is the shape whose every ``execute()`` fails on the
    runtime's alias-target guard. Two independent grounds answer whether that
    ``mem_id`` names a host arena. An arena that also holds one of the program's
    host tensors is one whatever the program records (see
    :func:`_host_planned_arenas`); failing that, an arena missing from the CUDA
    ones the program *does* record in ``non_const_buffer_device`` is one too.

    A placement the program does not record at all is a refusal of its own
    rather than either of those, because it is not an accusation about the
    planner's choice -- a buffer with no ``mem_id`` was not planned, and a
    program with no ``non_const_buffer_device`` entries says nothing about any
    of its arenas. It is refused because of what the runtime makes of it:
    ``MethodMeta::memory_planned_buffer_device`` answers ``CPU`` for an arena
    the ``.pte`` records nothing for, so a runner that honours it --
    ``examples/executorch_reference_runner`` does -- backs that arena with host
    memory and the engine then fails the alias-target guard on every call. Only
    ``apply_algo`` writes that record, so a caller-supplied
    ``memory_planning_pass`` that does not go through it leaves a ``.pte`` in
    exactly that state whatever it planned; a planner for a zero-copy cache has
    to leave the record behind, which means going through ``apply_algo`` with
    ``enable_non_cpu_memory_planning=True``. That parameter defaults to
    ``False``, and with it off ``apply_algo`` plans every spec into one CPU
    bucket and writes no record either.

    Where the record does name a GPU it is read as one: an arena recorded for
    ``cuda:1`` holding a cache whose own spec asks for ``cuda:0`` is an address
    on the wrong device, which fails the same way a host one does, so the index
    is compared and not only the type. An unrecorded index on either side says
    nothing and is accepted, since the arena's device type is already settled by
    then.

    Every method is read, not only ``forward``. ``export()`` rewires each method
    on its own, so a check that stopped at ``forward`` would pass a program whose
    decode had degenerated to staged -- and on the prefill/decode pair the user
    guide's zero-copy example exports it would not get that far, since a
    multi-method program need not have a ``forward`` at all. Within a method the
    marks and the stamped delegates are enumerated independently and a method is
    passed over only when it carries neither, because the disagreement between
    those two records is the whole subject: starting from the marks and looking
    the delegates up leaves anything recorded only on the delegate side outside
    the walk. The last refusal is about the program rather than about one method,
    matching the warning ``export()`` emits: a method with no aliased buffer
    mutation of its own is not an error, so a model that rewires only its decode
    step is accepted.

    This reads the graph and the finalized specs, so it says what the program
    does rather than what the passes recorded. It still says nothing about
    whether the engine's write itself is correct.

    ``save(..., zero_copy_kv=True)`` runs this for you, before it writes the
    file. The two-step path does not, and this is the limitation of that path:
    ``export(..., zero_copy_kv=True)`` hands back an ExecuTorch
    ``EdgeProgramManager``, whose ``to_executorch`` is ExecuTorch's own and
    finalizes whatever config it is given. So finalizing a zero-copy export
    yourself owes two calls, and nothing enforces either -- pass
    :func:`zero_copy_backend_config` as the config, and hand the program that
    comes back to this before writing the ``.pte``. Skipping the config is the
    consequential one: it writes a ``.pte`` whose caches never update, which is
    what the first three refusals above are.

    Arguments:
        program (executorch.exir.ExecutorchProgramManager): The finalized program
            ``to_executorch()`` returned. Every method it holds is read.

    Returns:
        None: a program that passes is left exactly as it was.

    Raises:
        RuntimeError: If the program does not update its KV buffers in place,
            naming the buffers or delegates and the method each is in.
    """
    method_names = sorted(program.methods)
    staged_by_method: Dict[str, List[str]] = {}
    short_by_method: Dict[str, List[str]] = {}
    unrecorded_by_method: Dict[str, List[str]] = {}
    host_planned_by_method: Dict[str, List[str]] = {}
    wrong_gpu_by_method: Dict[str, List[str]] = {}
    marked_anywhere = False
    for method_name in method_names:
        graph_module = program.exported_program(method_name).graph_module
        marked = [
            node
            for node in graph_module.graph.nodes
            if node.op == "placeholder"
            and node.meta.get("_torch_tensorrt_aliased_buffer")
        ]
        zero_copy_delegates = [
            node
            for node in graph_module.graph.nodes
            if _is_tensorrt_delegate(graph_module, node)
            and _delegate_declares_zero_copy(graph_module, node)
        ]
        # Both records have to be read before a method can be passed over. A
        # method carrying neither is one zero-copy never touched; a method
        # carrying a stamped delegate and no mark is the disagreement this
        # exists to catch, which skipping on the marks alone would hide.
        if not marked and not zero_copy_delegates:
            continue
        if marked:
            marked_anywhere = True
        zero_copy_delegate_args = {
            arg for node in zero_copy_delegates for arg in node.args[1:]
        }
        staged = [node.name for node in marked if node not in zero_copy_delegate_args]
        if staged:
            staged_by_method[method_name] = staged
        marked_nodes = set(marked)
        short = []
        for delegate in zero_copy_delegates:
            elided = _delegate_elided_output_names(graph_module, delegate)
            # Distinct buffers rather than argument slots, as the un-staging
            # pass counts them: one buffer in two slots is one cache.
            taken = len(
                {
                    arg
                    for arg in delegate.args[1:]
                    if isinstance(arg, Node) and arg in marked_nodes
                }
            )
            if taken >= max(len(elided), 1):
                continue
            if elided:
                short.append(
                    f"'{delegate.name}' takes {taken} marked buffer(s), not the "
                    f"{len(elided)} its elided aliased output(s) {sorted(elided)} "
                    "imply"
                )
            else:
                short.append(
                    f"'{delegate.name}' names no aliased output this can count, so "
                    "it must take at least one marked buffer and takes none"
                )
        if short:
            short_by_method[method_name] = short
        device_arenas = _device_planned_arenas(graph_module)
        host_arenas = _host_planned_arenas(graph_module)
        reaching = [node for node in marked if node in zero_copy_delegate_args]
        unrecorded: List[str] = []
        host_planned: List[str] = []
        wrong_gpu: List[str] = []
        # One classification per buffer. The two unrecorded shapes are not
        # evidence about the host arena and get a refusal of their own, but the
        # positive host-arena ground is read first where it applies, since
        # host-only planning both puts the cache among the host tensors and
        # writes no arena record, and naming the arena it is actually in tells
        # the caller more than saying nothing was recorded.
        for node in reaching:
            mem_id = getattr(node.meta.get("spec"), "mem_id", None)
            if mem_id is None:
                unrecorded.append(
                    f"'{node.name}' carries no mem_id, so memory planning left "
                    "it unplanned and nothing in the program says where it lives"
                )
            elif _is_host_planned(node, device_arenas, host_arenas):
                host_planned.append(node.name)
            elif device_arenas is None:
                unrecorded.append(
                    f"'{node.name}' is planned in arena {mem_id}, and the "
                    "program records no CUDA arena at all"
                )
            else:
                detail = _planned_on_another_gpu(node, device_arenas)
                if detail is not None:
                    wrong_gpu.append(detail)
        if unrecorded:
            unrecorded_by_method[method_name] = unrecorded
        if host_planned:
            host_planned_by_method[method_name] = host_planned
        if wrong_gpu:
            wrong_gpu_by_method[method_name] = wrong_gpu
    if staged_by_method:
        raise RuntimeError(
            f"TensorRT zero-copy KV: buffer(s) {_name_detail(staged_by_method)} "
            "are marked for in-place update but do not reach the TensorRT "
            "delegate that elided them directly, so the engine writes a staging "
            "copy that is discarded and the cache never updates. Export removed "
            "their copy-back, so nothing else would restore it. Finalize with "
            "torch_tensorrt.executorch.zero_copy_backend_config()."
        )
    if short_by_method:
        raise RuntimeError(
            "TensorRT zero-copy KV: delegate(s) "
            + "; ".join(
                f"{detail} in method '{method}'"
                for method, details in short_by_method.items()
                for detail in details
            )
            + ". Every marked buffer does reach a delegate declaring zero-copy "
            "KV, so either another one in the same method is holding this "
            "engine's cache or a mark was lost; either way this engine still "
            "reads a staging copy that is discarded, and export removed the "
            "copy-back. Export this method without zero_copy_kv, or keep each "
            "aliased buffer on the delegate whose engine elided it."
        )
    # Ordered after the still-staged and short-count refusals because both of
    # those fire on a disagreement between the marks and the stamped delegates,
    # and this one would answer such a program with "it was probably not exported
    # with zero_copy_kv=True" while its own compile specs say it was.
    if not marked_anywhere:
        raise RuntimeError(
            "TensorRT zero-copy KV: no buffer in this program is marked for "
            f"in-place update, in any of its methods ({', '.join(method_names)}), "
            "so it stages its caches like any other .pte. Either it was not "
            "exported with zero_copy_kv=True, or it was and no aliased buffer "
            "mutation was found -- export logs a warning for that case."
        )
    if host_planned_by_method:
        raise RuntimeError(
            f"TensorRT zero-copy KV: buffer(s) "
            f"{_name_detail(host_planned_by_method)} reach their TensorRT "
            "delegate directly but memory planning put them in an arena that "
            "also holds the program's host tensors, or in one it does not record "
            "as CUDA, so the engine is handed a host pointer it cannot write and "
            "every execute() fails on the runtime's alias-target guard. Finalize "
            "with torch_tensorrt.executorch.zero_copy_backend_config() over a "
            "configuration that leaves enable_non_cpu_memory_planning on. If it "
            "is already on, the memory_planning_pass in use is what put this "
            "buffer among the host tensors, and it has to give the delegate's "
            "device an arena of its own."
        )
    if unrecorded_by_method:
        raise RuntimeError(
            "TensorRT zero-copy KV: buffer(s) "
            + "; ".join(
                f"{detail} in method '{method}'"
                for method, details in unrecorded_by_method.items()
                for detail in details
            )
            + ". These buffers reach their TensorRT delegate directly, so the "
            "engine writes them through the pointer the runtime allocates for "
            "them, and the .pte has to say that pointer is device memory. "
            "MethodMeta::memory_planned_buffer_device answers CPU for an arena "
            "the program records nothing for, so a runner that honours it backs "
            "the arena with host memory and every execute() fails on the "
            "alias-target guard. The memory_planning_pass in use has to plan "
            "these buffers and leave the record behind, which means going "
            "through ExecuTorch's apply_algo with "
            "enable_non_cpu_memory_planning=True -- that parameter defaults to "
            "False, and with it off apply_algo plans every spec into one CPU "
            "bucket and writes no record either."
        )
    if wrong_gpu_by_method:
        raise RuntimeError(
            "TensorRT zero-copy KV: buffer(s) "
            + "; ".join(
                f"{detail} in method '{method}'"
                for method, details in wrong_gpu_by_method.items()
                for detail in details
            )
            + ". The engine writes the cache through the pointer the runtime "
            "allocates out of that arena, so it would write the wrong GPU. The "
            "memory_planning_pass in use has to give each delegate's own device "
            "an arena, and put every buffer its engine writes in place in that "
            "device's one."
        )


def _refuse_skip_h2d(config: "ExecutorchBackendConfig") -> None:
    """Raise if the config asks ExecuTorch to un-stage method inputs as well.

    Both places the option can be written are read. ``propagate_device_config``
    is one ``PropagateDeviceConfig`` or a dict of them keyed by method, and
    within either, ``skip_h2d_for_method_inputs`` is a bool or a second
    per-method dict.

    What is refused is every value that pass reads as on, which is every truthy
    one rather than only ``True``. ``PropagateDevicePass`` is handed the field
    whole and only tests it for truth (in ``_insert_h2d_copies``), never
    resolving it per method, so it reads any non-empty dict as on for every
    method -- one whose entries are all ``False`` included. Refusing only the
    true entries would carry such a dict through and hand back a config that
    raises a layer down, which is the failure this exists to prevent. ``False``
    and the empty dict are what that pass reads as off, and both are carried.
    """
    propagate = getattr(config, "propagate_device_config", None)
    per_method = (
        sorted(propagate.items())
        if isinstance(propagate, dict)
        else [("every method", propagate)]
    )
    asked_for = [
        method
        for method, entry in per_method
        if getattr(entry, "skip_h2d_for_method_inputs", False)
    ]
    if not asked_for:
        return
    raise ValueError(
        "TensorRT zero-copy KV: this configuration sets "
        f"skip_h2d_for_method_inputs for {', '.join(asked_for)}, which cannot be "
        "combined with zero-copy KV. PropagateDevicePass refuses to un-stage a "
        "method input whose placeholder does not have exactly one user, and a "
        "buffer zero-copy rewired has two -- the TensorRT delegate, and the graph "
        "output it is its own mutation result for -- so finalization raises "
        "there. Every value that pass reads as on is refused, not only True: it "
        "tests skip_h2d_for_method_inputs for truth without ever resolving it "
        "per method, so setting it to a dict whose entries are all False still "
        "turns it on for every method. Zero-copy already un-stages the aliased "
        "buffers; leave skip_h2d_for_method_inputs at False or unset -- both of "
        "which are carried -- and pass the method's own inputs on the host."
    )


def zero_copy_backend_config(
    config: Optional["ExecutorchBackendConfig"] = None,
) -> "ExecutorchBackendConfig":
    """Build the ``ExecutorchBackendConfig`` a zero-copy KV program needs.

    This is the second half of ``export(..., zero_copy_kv=True)``. Export has
    already removed ExecuTorch's copy-back of the aliased buffers; this installs
    the pass that removes their staging, so the engine writes the caller's
    buffer instead of a scratch copy that is thrown away.

    The feature is split across two calls because ``to_executorch()`` belongs to
    ExecuTorch, not to Torch-TensorRT: ``export()`` hands back an
    ``EdgeProgramManager`` at the Edge boundary and never sees the config the
    program is finalized with.

    ``config`` is your own configuration -- every field is preserved, and a
    ``to_out_var_pass`` you already set runs after the un-staging. Omit it to
    start from ExecuTorch's defaults. Two fields are not merely carried:

    * ``enable_non_cpu_memory_planning`` is *read*. Zero-copy needs the caches
      planned in device memory, so ``False`` -- which plans every tensor into
      the one host arena -- has the pass refuse each cache it finds rather than
      write a ``.pte`` whose every ``execute()`` fails. It is read off the
      config returned here, at the moment the pass runs, so setting the field
      *on that object* afterwards is honoured: the pass and the finalizer then
      cannot disagree about it. There are two cases the field decides nothing
      in, and that the pass therefore refuses nothing on. One is building a
      *new* config out of this one with ``dataclasses.replace``: the field is a
      bool, copied by value, while the pass is copied by reference and goes on
      reading the config returned here, so call this function again on the
      derived config and the pass it builds reads that one. The other is a
      ``memory_planning_pass`` of your own that does not already carry an
      attribute of that name -- ``to_executorch`` assigns the flag onto the
      planner rather than passing it, and only onto a planner that has it, so
      for any other the field reaches nothing and where the caches land is that
      planner's own business. Neither is left to the runtime to discover:
      :func:`check_zero_copy_kv` reads the arena memory planning actually chose
      and refuses the program either mistake produces -- but only where
      something runs it, which on the two-step path is you. The one placement it
      cannot settle is a method holding no host tensor to give the
      shared arena away and a program that records its arena devices; a planner
      that records nothing is refused rather than passed over, as
      :func:`check_zero_copy_kv` describes.
    * ``propagate_device_config.skip_h2d_for_method_inputs`` is *refused*,
      wherever it is written -- in the single ``PropagateDeviceConfig`` or in a
      per-method dict of them -- and on every value ``PropagateDevicePass``
      reads as on rather than only on ``True``: it tests the field for truth
      without ever resolving it per method, so any non-empty dict is on for
      every method, one of ``False`` included. It is ExecuTorch's own un-staging
      of method inputs, and it requires each placeholder it un-stages to have
      exactly one user. A rewired cache always has two -- the delegate, and the
      graph output it is its own mutation result for -- so
      ``PropagateDevicePass`` raises on every zero-copy graph. Returning the
      option unchanged would hand back a config that cannot finalize at all;
      this says so here instead, where the caller can act on it.

    .. warning::
        Finalizing a ``zero_copy_kv=True`` program *without* this config leaves
        the engine writing a per-call staging copy that is then discarded, so
        the buffer never updates -- for a KV cache, wrong output rather than a
        crash. **Nothing stops that on the two-step path.** ``to_executorch``
        belongs to ExecuTorch and finalizes whatever config it is handed, so
        pairing the two calls is yours to get right, and so is handing the
        finalized program to :func:`check_zero_copy_kv` afterwards -- which is
        what catches the placements this config cannot settle.
        ``torch_tensorrt.save(..., zero_copy_kv=True)`` owns both ends and does
        both for you.

        ``save(..., zero_copy_kv=True)`` installs this pass itself, so handing
        it the result of this function as ``backend_config`` applies the pass
        twice. That is redundant rather than an error -- the second run finds
        the buffers already wired straight to their delegates and changes
        nothing -- but the two entry points are alternatives: use one or the
        other.

    Arguments:
        config (Optional[executorch.exir.ExecutorchBackendConfig]): The
            configuration to compose onto. Omit it to start from ExecuTorch's
            defaults.

    Returns:
        executorch.exir.ExecutorchBackendConfig: A new config, every field of
        the given one preserved, whose ``to_out_var_pass`` un-stages the aliased
        buffers before running the ``to_out_var_pass`` that was there.

    Raises:
        ValueError: If the configuration sets
            ``propagate_device_config.skip_h2d_for_method_inputs``, which cannot
            be combined with zero-copy KV.
    """
    from dataclasses import replace

    from executorch.exir import ExecutorchBackendConfig

    base = config if config is not None else ExecutorchBackendConfig()
    _refuse_skip_h2d(base)
    # No device_memory_planning here: setting finalization_config below overrides
    # it for every call, so a value passed would never be read.
    unstage = unstage_aliased_buffers_pass(base.to_out_var_pass)
    wrapped = replace(base, to_out_var_pass=unstage)
    unstage.finalization_config = wrapped
    return wrapped
