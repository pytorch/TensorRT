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
  partition.
* :func:`unstage_aliased_buffers_pass`, as a ``to_out_var_pass``, drops the
  staging so the engine writes the caller's buffer rather than a copy.

Applying only the first would leave the engine writing a discarded staging copy
with nothing to copy back -- the buffer would simply never update. So neither
pass is public on its own: the rewiring is reached only through
``export(..., zero_copy_kv=True)``, and the un-staging only through
:func:`zero_copy_backend_config`. That, plus :func:`check_zero_copy_kv` -- which
reads a finalized program back and refuses one where the pairing did not happen
-- is what this module exports.
"""

import logging
import operator
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Set

import torch
from torch.fx import Node

if TYPE_CHECKING:
    from executorch.exir import ExecutorchBackendConfig

_LOGGER = logging.getLogger(__name__)


def _engine_info_str(engine_info: List[Any], index: int) -> str:
    if index < 0 or index >= len(engine_info) or engine_info[index] is None:
        return ""
    value = engine_info[index]
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def _aliased_inputs_by_output_index(
    exported_program: Any, engine_node: Node
) -> Dict[int, Node]:
    """Map each aliased output index of one engine to the input it writes in place.

    Reads the engine's own ``aliased_io`` rather than inferring aliasing from the
    graph. The graph cannot tell the difference: an aliased KV mutation and a
    copy-back mutation are both a ``getitem`` off the engine node whose buffer is
    also an engine input, and rewiring a copy-back would silently drop a real
    update. An output binding absent from the delegate's inputs is skipped
    rather than reported -- ``_declare_aliased_kv_mutations_on_ep`` has already
    warned about that same engine, and there is no mutation to rewire either way.
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

    # Only aliased_io and the binding names are read, never the engine itself.
    engine_info = _resolve_engine_info(
        exported_program, engine_node, metadata_only=True
    )
    aliased_io = deserialize_aliased_io(_engine_info_str(engine_info, ALIASED_IO_IDX))
    if not aliased_io:
        return {}
    input_names = deserialize_binding_names(
        _engine_info_str(engine_info, INPUT_BINDING_NAMES_IDX)
    )
    output_names = deserialize_binding_names(
        _engine_info_str(engine_info, OUTPUT_BINDING_NAMES_IDX)
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

    engine_info = _resolve_engine_info(
        exported_program, engine_node, metadata_only=True
    )
    names: List[str] = deserialize_binding_names(
        _engine_info_str(engine_info, OUTPUT_BINDING_NAMES_IDX)
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


def order_rewired_mutations_last(exported_program: Any) -> int:
    """Move an Edge program's rewired mutations behind its un-rewired ones.

    ExecuTorch's ``insert_write_back_for_buffers_pass`` renames the mutation
    output specs by walking them with one counter, indexing a list of the
    ``copy_`` nodes it created followed by every output it copied nothing for, in
    graph order. That pairs every spec with its own value exactly when every
    mutation the pass skips comes after every one it copies: a skip with a
    copy-producing mutation behind it shifts that mutation onto another buffer's
    value. A rewired mutation is one the pass creates no copy for -- its value
    already *is* the buffer placeholder -- so ordering every rewired mutation
    behind every un-rewired one meets that condition while the rewired ones are
    the only skips, and keeping each group's relative order lines the copies up
    with the prefix one for one and the rewired values with the tail. A method
    holding only rewired mutations, or only un-rewired ones, is already in that
    order and is left alone.

    The write-back also skips a mutation whose value is a chain of in-place ops
    rooted at the buffer. That one is not rewired, so this pass leaves it in the
    prefix, where it crosses the specs only if a copy-producing mutation follows
    it there. Edge programs are functional, so this pass sees none;
    ``to_executorch()`` runs ``reinplace_pass`` before the write-back when
    ``run_reinplace_pass`` or ``reinplace_extra_ops`` is set, which can create
    one, and :func:`zero_copy_backend_config` preserves both fields rather than
    overriding them. When one turns up with a copy-producing mutation behind it,
    the upstream off-by-one is back, reached by a route this pass cannot see and
    after it has already run.

    This runs on the Edge program rather than beside the rewiring, because
    ``to_edge_transform_and_lower`` calls ``run_decompositions``, which rebuilds
    the output specs in its own order and drops any permutation applied before
    it. The Edge program is the last one ``export()`` holds, and
    ``to_executorch()`` reaches the write-back pass without re-deriving the order
    again.

    Both mutation kinds are moved as one block because that counter walks both.
    The permutation stays inside the mutation block -- a spec is only ever
    exchanged with another mutation spec -- so no ``USER_OUTPUT`` changes
    position; moving one would renumber the program's real outputs. Spec and
    graph-output arg move together, or the program stops verifying.

    A mutation counts as rewired when its value is a placeholder carrying the
    mark :func:`rewire_aliased_mutations_to_buffers` left, which is the same
    property the un-staging pass keys off later.

    Returns the number of mutations in the block it rewrote, or 0 when the method
    held only one kind and there was no reordering to be done. A block that holds
    both kinds is rewritten, and its length reported, even when the permutation
    turns out to be the identity.
    """
    from torch.export.graph_signature import (
        ExportGraphSignature,
        OutputKind,
        OutputSpec,
    )

    graph_module = exported_program.graph_module
    signature = exported_program.graph_signature
    output_node = graph_module.graph.output_node()
    output_args = list(output_node.args[0])
    output_specs: List[OutputSpec] = list(signature.output_specs)

    mutation_kinds = (OutputKind.BUFFER_MUTATION, OutputKind.USER_INPUT_MUTATION)
    slots = [
        index
        for index, spec in enumerate(output_specs[: len(output_args)])
        if spec.kind in mutation_kinds
    ]

    def is_rewired(index: int) -> bool:
        value = output_args[index]
        return (
            isinstance(value, Node)
            and value.op == "placeholder"
            and bool(value.meta.get("_torch_tensorrt_aliased_buffer"))
        )

    rewired = [index for index in slots if is_rewired(index)]
    if not rewired or len(rewired) == len(slots):
        return 0

    # Stable, so both groups keep their relative order; False sorts before True.
    sources = sorted(slots, key=lambda index: index in set(rewired))
    specs = [output_specs[index] for index in sources]
    args = [output_args[index] for index in sources]
    for slot, spec, arg in zip(slots, specs, args):
        output_specs[slot] = spec
        output_args[slot] = arg

    output_node.args = (tuple(output_args),)
    graph_module.recompile()
    exported_program._graph_signature = ExportGraphSignature(
        input_specs=list(signature.input_specs), output_specs=output_specs
    )
    _LOGGER.debug(
        "ordered %d rewired mutation(s) behind %d un-rewired one(s)",
        len(rewired),
        len(slots) - len(rewired),
    )
    return len(slots)


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

    The order the mutations end up in matters as well, but not here: see
    :func:`order_rewired_mutations_last`, which runs on the Edge program because
    lowering rebuilds the output specs.

    On its own this is not correct: ExecuTorch still stages the buffer, so the
    engine's in-place write would land in per-call scratch and, with the
    copy-back gone, be lost. It is only correct paired with the un-staging pass.

    Returns the engine output binding names of the aliased outputs it elided,
    one per rewired mutation. Only these names may later be exempted from the
    backend's output-binding check -- every *other* aliased output (a user alias
    on a plain, non-buffer input, which export never rewired) must still be a
    delegate output, so an engine mixing the two is caught rather than silently
    dropping the un-rewired one's update into scratch.
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
        _LOGGER.debug("no aliased buffer mutations to rewire")
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
    # Leaving an engine with no output would leave its delegate with no outputs.
    # Nothing downstream reports that shape: the runtime infers elision from a
    # single argument count, which a zero-output delegate satisfies, and a
    # delegate nothing reads is a pure node that a later graph-wide dead-code
    # elimination can erase, taking the computation with it. This raise, and not
    # the eliminate_dead_code() below, is what stops that. That DCE leaves the
    # engine node itself in place -- PyTorch defaults an operator taking a
    # ScriptObject argument to an ORDERED effect (torch._library.effects), and
    # execute_engine takes the engine as one, so FX reads it as impure where the
    # delegate is not -- but it does erase the engine's own dead users, so a
    # user with no users of its own is not an output the engine still has.
    for engine in engines_with_elided_outputs:
        if not any(user.users for user in engine.users):
            raise RuntimeError(
                "TensorRT zero-copy KV: eliding the aliased buffers engine node "
                f"'{engine.name}' writes in place leaves it with no output any "
                "node reads, so the delegate would have no outputs at all. This "
                "shape is not supported; export this method without "
                "zero_copy_kv."
            )
    graph_module.graph.eliminate_dead_code()
    graph_module.graph.lint()
    graph_module.recompile()
    # The signature is replaced in place rather than by rebuilding the program:
    # the graph has already been edited in place, and every other field would be
    # copied across unchanged.
    exported_program._graph_signature = ExportGraphSignature(
        input_specs=list(signature.input_specs), output_specs=output_specs
    )
    _LOGGER.debug(
        "rewired %d aliased mutation(s) to their buffers, eliding outputs %s",
        len(mutations),
        elided_output_names,
    )
    return elided_output_names


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
    that declares it but un-staged nothing is a lost KV update -- the mark that
    would have driven the un-staging did not survive to this pass -- and is caught
    in :func:`_unstage_aliased_buffers`.
    """
    from torch_tensorrt.executorch.backend import ZERO_COPY_KV_COMPILE_SPEC_KEY

    lowered = node.args[0] if node.args else None
    if not isinstance(lowered, Node) or lowered.op != "get_attr":
        return False
    module = getattr(graph_module, lowered.target, None)
    return any(
        getattr(spec, "key", None) == ZERO_COPY_KV_COMPILE_SPEC_KEY
        for spec in (getattr(module, "compile_specs", None) or [])
    )


def _device_move_is_safe(
    source: Node, h2d_copy: Any, target_device: Any, target_device_index: Any
) -> bool:
    """True when moving ``source``'s spec device disturbs no other consumer.

    A placeholder's device is shared by every user, so it can only be retargeted
    to the delegate's device when nothing *reads* it on another one. ExecuTorch
    guards the same hazard, more strictly and only under its opt-in
    ``skip_h2d_for_method_inputs``: it demands the placeholder have exactly one
    user. The rule here is looser because two kinds of user impose no such
    constraint and are allowed: the graph ``output`` node -- the buffer is its
    own BUFFER_MUTATION result, which is exactly what zero-copy sets up and
    which carries no device of its own -- and another ``_h2d_copy`` staging to
    the same GPU, superseded by the un-staging when it feeds a TensorRT
    delegate, and otherwise left in place, still reading a buffer that now lives
    on the GPU it was copying to. Any other reader (a compute op, or a staging
    to a different device) makes the move unsafe.

    The index is compared as well as the type, because ``spec.device`` is only
    ``CUDA``/``CPU``: two engines resolved to ``cuda:0`` and ``cuda:1`` stage the
    same buffer to different GPUs, and un-staging both would leave whichever ran
    last owning the buffer while the other engine writes an address on the wrong
    device.
    """
    for user in source.users:
        if user.op == "output":
            continue
        if not (
            isinstance(user, Node)
            and user.op == "call_function"
            and user.target is h2d_copy
        ):
            return False
        spec = user.meta.get("spec")
        if spec is None or spec.device != target_device:
            return False
        if spec.device_index != target_device_index:
            return False
    return True


def _unstage_aliased_buffers(graph_module: torch.fx.GraphModule) -> int:
    """Route TensorRT delegate inputs from their staging copy back to the buffer.

    A delegate input qualifies only when it is an ``_h2d_copy`` of a placeholder
    carrying the mark left by :func:`rewire_aliased_mutations_to_buffers`. Every
    other input keeps its staging, including a mutable buffer the engine does
    not write in place.

    The placeholder's spec takes over the staging copy's device, so memory
    planning puts the buffer in the delegate's device arena rather than a host
    one. That is what makes handing the buffer straight to the engine valid at
    all: a host-arena pointer is not something the engine can write. That move is
    refused when the buffer has another consumer (see
    :func:`_device_move_is_safe`), which would otherwise have its device silently
    changed too.

    A failure here is a lost KV update -- unless the program has already been
    through this pass, the one case where nothing is lost -- so it is raised
    rather than logged: export has already removed the copy-back, so a marked
    buffer left staged has the engine write per-call scratch that is then
    discarded and the buffer never updates. It raises when the staging copy has
    no spec or is not on CUDA, when the device move is unsafe, and -- so a
    discovery miss cannot pass silently -- after the loop when any marked buffer
    was never un-staged, cross-checked against each delegate's own
    ``zero_copy_kv`` spec: a TensorRT delegate that declares zero-copy but
    un-staged nothing is broken and names the buffer.

    Returns the number of delegate inputs un-staged.
    """
    from executorch.exir.schema import DeviceType
    from torch_tensorrt.executorch.backend import ZERO_COPY_KV_COMPILE_SPEC_KEY

    h2d_copy = torch.ops.et_copy._h2d_copy.default
    unstaged = 0
    unstaged_placeholders: Set[Node] = set()
    orphaned_stagings: List[Node] = []
    zero_copy_delegates: List[Node] = []
    unstaged_per_delegate: Dict[Node, int] = {}

    for node in list(graph_module.graph.nodes):
        if not _is_tensorrt_delegate(graph_module, node):
            continue
        declares_zero_copy = _delegate_declares_zero_copy(graph_module, node)
        if declares_zero_copy:
            zero_copy_delegates.append(node)
            unstaged_per_delegate[node] = 0
        new_args = list(node.args)
        for i, arg in enumerate(node.args[1:], start=1):
            if not isinstance(arg, Node) or arg.target is not h2d_copy:
                continue
            source = arg.args[0]
            if not isinstance(source, Node) or source.op != "placeholder":
                continue
            if not source.meta.get("_torch_tensorrt_aliased_buffer"):
                continue  # not written in place; it needs its staging copy
            staged_spec = arg.meta.get("spec")
            source_spec = source.meta.get("spec")
            if staged_spec is None or source_spec is None:
                raise RuntimeError(
                    "TensorRT zero-copy KV: no TensorSpec on the staging copy of "
                    f"buffer '{source.name}', so it cannot be moved to the "
                    "delegate's device. The TensorRT engine writes this buffer in "
                    "place and its copy-back has already been removed, so the "
                    "update would be lost. This pass has to run as the "
                    "ExecutorchBackendConfig to_out_var_pass, which is where the "
                    "specs exist; torch_tensorrt.executorch.zero_copy_backend_config "
                    "installs it there."
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
            already_placed = (source_spec.device, source_spec.device_index) == (
                staged_spec.device,
                staged_spec.device_index,
            )
            if not already_placed and not _device_move_is_safe(
                source, h2d_copy, staged_spec.device, staged_spec.device_index
            ):
                raise RuntimeError(
                    "TensorRT zero-copy KV: buffer "
                    f"'{source.name}' is read by a consumer the move would "
                    "disturb -- an op outside the delegate, or a staging copy "
                    "bound for a different GPU -- so placing it on this engine's "
                    "device would silently change that consumer's device too. "
                    "Export this method without zero_copy_kv, or stop sharing the "
                    "aliased buffer."
                )
            source_spec.device = staged_spec.device
            source_spec.device_index = staged_spec.device_index
            new_args[i] = source
            unstaged += 1
            unstaged_placeholders.add(source)
            orphaned_stagings.append(arg)
            if declares_zero_copy:
                unstaged_per_delegate[node] += 1
        node.args = tuple(new_args)

    marked_but_unstaged = [
        node
        for node in graph_module.graph.nodes
        if node.op == "placeholder"
        and node.meta.get("_torch_tensorrt_aliased_buffer")
        and node not in unstaged_placeholders
    ]
    if marked_but_unstaged:
        names = ", ".join(repr(node.name) for node in marked_but_unstaged)
        raise RuntimeError(
            "TensorRT zero-copy KV: buffer(s) "
            f"{names} were marked for in-place update but no TensorRT delegate "
            "staging was found to un-stage. Either they never reached a "
            "TensorRT delegate, which is a broken zero-copy program -- export "
            "removed their copy-back, so leaving them staged has the engine "
            "write per-call scratch that is discarded and the buffer never "
            "updates -- or this pass has already run over the program and they "
            "are wired straight to the engine already, which happens whenever "
            "it is installed twice: nesting zero_copy_backend_config, "
            "finalizing the same program twice, or passing "
            "save(zero_copy_kv=True) a config that already carries the pass. "
            "Install it once."
        )
    for delegate in zero_copy_delegates:
        if unstaged_per_delegate[delegate] == 0:
            staged_inputs = [
                arg.args[0].name
                for arg in delegate.args[1:]
                if isinstance(arg, Node)
                and arg.target is h2d_copy
                and isinstance(arg.args[0], Node)
            ]
            raise RuntimeError(
                "TensorRT zero-copy KV: delegate "
                f"'{delegate.name}' declares zero-copy KV "
                f"(compile spec '{ZERO_COPY_KV_COMPILE_SPEC_KEY}') but no aliased "
                f"buffer was un-staged for it (staged inputs: {staged_inputs}). Export "
                "elided its aliased outputs, so the engine now writes per-call "
                "scratch that is discarded and the cache never updates."
            )

    if unstaged:
        # Erase only the stagings we orphaned. A graph-wide eliminate_dead_code()
        # in a to_out_var_pass could delete another backend's unused delegate.
        for staging in dict.fromkeys(orphaned_stagings):
            if not staging.users:
                graph_module.graph.erase_node(staging)
        graph_module.graph.lint()
        graph_module.recompile()
    return unstaged


def unstage_aliased_buffers_pass(inner_pass: Optional[Any] = None) -> Any:
    """Build a ``to_out_var_pass`` that un-stages aliased buffers, then delegates.

    ``to_out_var_pass`` is the last hook that runs after ``PropagateDevicePass``
    and before memory planning -- the window in which the staging copies exist
    and the buffers' placement is not yet fixed. (``sym_shape_eval_pass`` is a
    caller-supplied hook in that window too, but it runs first.)

    ``inner_pass`` is the ``to_out_var_pass`` that would otherwise have run; it
    runs after the un-staging. Omit it for ExecuTorch's default.
    """
    from executorch.exir import ExecutorchBackendConfig
    from executorch.exir.pass_base import PassBase

    inner = (
        inner_pass
        if inner_pass is not None
        else ExecutorchBackendConfig().to_out_var_pass
    )

    class _UnstageThenToOutVar(PassBase):  # type: ignore[misc]
        def call(self, graph_module: torch.fx.GraphModule) -> Any:
            unstaged = _unstage_aliased_buffers(graph_module)
            _LOGGER.debug("un-staged %d aliased delegate buffer(s)", unstaged)
            return inner(graph_module)

    return _UnstageThenToOutVar()


def check_zero_copy_kv(program: Any) -> None:
    """Raise unless a finalized program really updates its KV buffers in place.

    ``program`` is what ``to_executorch()`` returns. Both halves of zero-copy do
    nothing quietly when they find nothing to do: ``zero_copy_kv=True`` warns and
    carries on when the model holds no aliased buffer mutation, and
    :func:`unstage_aliased_buffers_pass`, handed a program with nothing marked,
    un-stages nothing and returns. Either way the ``.pte`` that comes out runs
    and stages its cache like any other, which for a KV cache is wrong output
    rather than a crash.

    Two shapes are refused: a marked buffer that is not a direct argument of any
    delegate -- it still reaches one through an ``_h2d_copy`` staging, or reaches
    none -- and a program with no marked buffer in any method. The first is what
    finalizing without :func:`zero_copy_backend_config` leaves behind; when that
    config *is* installed, ``_unstage_aliased_buffers`` has already raised on the
    same condition.

    Every method is read, not only ``forward``. ``export()`` rewires each method
    on its own, so a check that stopped at ``forward`` would pass a program whose
    decode had degenerated to staged -- and on the prefill/decode pair the user
    guide's zero-copy example exports it would not get that far, since a
    multi-method program need not have a ``forward`` at all. The second refusal is
    about the program rather than about one method, matching the warning
    ``export()`` emits: a method with no aliased buffer mutation of its own is
    not an error, so a model that rewires only its decode step is accepted.

    This reads the graph, so it says what the program does rather than what the
    passes recorded. It says nothing about whether the engine's write is correct,
    only that the buffer it writes is the caller's.
    """
    from executorch.exir.delegate import executorch_call_delegate

    method_names = sorted(program.methods)
    staged_by_method: Dict[str, List[str]] = {}
    marked_anywhere = False
    for method_name in method_names:
        graph_module = program.exported_program(method_name).graph_module
        marked = [
            node
            for node in graph_module.graph.nodes
            if node.op == "placeholder"
            and node.meta.get("_torch_tensorrt_aliased_buffer")
        ]
        if not marked:
            continue
        marked_anywhere = True
        delegate_args = {
            arg
            for node in graph_module.graph.nodes
            if node.op == "call_function" and node.target is executorch_call_delegate
            for arg in node.args[1:]
        }
        staged = [node.name for node in marked if node not in delegate_args]
        if staged:
            staged_by_method[method_name] = staged
    if not marked_anywhere:
        raise RuntimeError(
            "TensorRT zero-copy KV: no buffer in this program is marked for "
            f"in-place update, in any of its methods ({', '.join(method_names)}), "
            "so it stages its caches like any other .pte. Either it was not "
            "exported with zero_copy_kv=True, or it was and no aliased buffer "
            "mutation was found -- export logs a warning for that case."
        )
    if staged_by_method:
        detail = ", ".join(
            f"'{name}' in method '{method}'"
            for method, names in staged_by_method.items()
            for name in names
        )
        raise RuntimeError(
            f"TensorRT zero-copy KV: buffer(s) {detail} are marked for in-place "
            "update but do not reach a delegate directly, so the engine writes a "
            "staging copy that is discarded and the cache never updates. Export "
            "removed their copy-back, so nothing else would restore it. Finalize "
            "with torch_tensorrt.executorch.zero_copy_backend_config()."
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
    start from ExecuTorch's defaults.

    .. warning::
        Finalizing a ``zero_copy_kv=True`` program *without* this config does
        not raise on its own. The engine writes a per-call staging copy that is
        then discarded and the buffer never updates, which for a KV cache is
        wrong output rather than a crash. Hand the finalized program to
        :func:`check_zero_copy_kv` before writing the ``.pte`` and that mistake
        becomes an error; ``torch_tensorrt.save(..., zero_copy_kv=True)`` runs
        the check for you.

        The opposite mistake does raise. ``save(..., zero_copy_kv=True)``
        installs this pass itself, so handing it the result of this function as
        ``backend_config`` applies the pass twice and finalization fails. The
        two entry points are mutually exclusive: use one or the other.
    """
    from dataclasses import replace

    from executorch.exir import ExecutorchBackendConfig

    base = config if config is not None else ExecutorchBackendConfig()
    return replace(
        base, to_out_var_pass=unstage_aliased_buffers_pass(base.to_out_var_pass)
    )
