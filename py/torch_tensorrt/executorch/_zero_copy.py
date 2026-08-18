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
half may be applied alone, and neither is part of the public API on its own: a
caller opts into both together or into neither.
"""

import logging
import operator
from typing import Any, Dict, List, NamedTuple, Optional

import torch
from torch.fx import Node

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
    from torch_tensorrt.executorch.backend import _get_engine_info_for_node

    engine_info = _get_engine_info_for_node(exported_program, engine_node)
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


def rewire_aliased_mutations_to_buffers(exported_program: Any) -> int:
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

    Returns the number of mutations rewired.
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
        return 0

    elided_by_engine: Dict[Node, List[Node]] = {}
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
        elided_by_engine.setdefault(mutation.engine, []).append(mutation.aliased_output)

    output_node.args = (tuple(output_args),)
    # Dropping every output of an engine would leave a delegate with no outputs.
    # Nothing downstream reports that shape: the runtime infers elision from a
    # single argument count, which a zero-output delegate satisfies, and a
    # delegate nothing reads is a pure node that a later graph-wide dead-code
    # elimination can erase, taking the computation with it. Stop here instead.
    # The eliminate_dead_code() below does not erase this engine node: unlike
    # the delegate, an execute_engine node is impure to FX and survives with no
    # users.
    for engine, elided in elided_by_engine.items():
        if all(user in elided and not user.users for user in engine.users):
            raise RuntimeError(
                "TensorRT zero-copy KV: every output of engine node "
                f"'{engine.name}' is an aliased buffer written in place, so "
                "eliding them would leave the delegate with no outputs at all. "
                "This shape is not supported; export this method without "
                "zero_copy_kv."
            )
    graph_module.graph.eliminate_dead_code()
    graph_module.graph.lint()
    graph_module.recompile()
    exported_program._graph_signature = ExportGraphSignature(
        input_specs=list(signature.input_specs), output_specs=output_specs
    )
    _LOGGER.debug("rewired %d aliased mutation(s) to their buffers", len(mutations))
    return len(mutations)


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


def _unstage_aliased_buffers(graph_module: torch.fx.GraphModule) -> int:
    """Route TensorRT delegate inputs from their staging copy back to the buffer.

    A delegate input qualifies only when it is an ``_h2d_copy`` of a placeholder
    carrying the mark left by :func:`rewire_aliased_mutations_to_buffers`. Every
    other input keeps its staging, including a mutable buffer the engine does
    not write in place.

    The placeholder's spec takes over the staging copy's device, so memory
    planning puts the buffer in the delegate's device arena instead of a host
    one -- which is what makes the engine's write land somewhere the caller can
    still see afterwards.

    Raises when a marked buffer cannot be un-staged, because the alternative is
    silence: its copy-back has already been removed, so leaving the staging in
    place means the engine writes a scratch tensor and the buffer never updates.

    Returns the number of delegate inputs un-staged.
    """
    from executorch.exir.schema import DeviceType

    h2d_copy = torch.ops.et_copy._h2d_copy.default
    unstaged = 0
    for node in list(graph_module.graph.nodes):
        if not _is_tensorrt_delegate(graph_module, node):
            continue
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
            source_spec.device = staged_spec.device
            source_spec.device_index = staged_spec.device_index
            new_args[i] = source
            unstaged += 1
        node.args = tuple(new_args)
    if unstaged:
        graph_module.graph.eliminate_dead_code()
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
