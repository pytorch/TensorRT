# ExecuTorch TensorRT backend: serialize engines to a libtorch-free runtime blob.

import json
import operator
from typing import Any, Container, Iterable, List, Optional, Set, final

import torch
import torch.fx
from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)
from torch.export.exported_program import ExportedProgram
from torch_tensorrt.dynamo._exporter import _resolve_lifted_custom_obj
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
    ALIASED_IO_IDX,
    DEVICE_IDX,
    ENGINE_IDX,
    HW_COMPATIBLE_IDX,
    INPUT_BINDING_NAMES_IDX,
    OUTPUT_BINDING_NAMES_IDX,
    REQUIRES_OUTPUT_ALLOCATOR_IDX,
    SERIALIZED_METADATA_IDX,
    TARGET_PLATFORM_IDX,
    deserialize_aliased_io,
)
from torch_tensorrt.executorch.serialization import (
    TensorRTBlobMetadata,
    TensorRTIOBinding,
    serialize_engine,
)

_BINDING_DELIM = "%"

# CompileSpec key by which export() tells this backend which aliased outputs it
# deliberately took out of the delegate. Its value is the JSON list of those
# engine output binding names (see _serialize_elided_output_names), NOT a bare
# flag: export only elides the aliased outputs backed by a registered buffer, so
# the backend must exempt exactly those and still reject a delegate that dropped
# any other binding. It travels on the partitioner's DelegationSpec, the only
# channel from the export call down to preprocess. Without it a delegate short of
# its aliased outputs is a bug, not a zero-copy program, and stays an error.
ZERO_COPY_KV_COMPILE_SPEC_KEY = "zero_copy_kv"


def _schema_name(target: Any) -> str:
    """Return the qualified op schema name for an OpOverload or EdgeOpOverload."""
    if hasattr(target, "_schema"):
        return str(target._schema.name)
    return ""


_ENGINE_OP_SCHEMA_NAMES = (
    "tensorrt::execute_engine",
    "tensorrt::no_op_placeholder_for_execute_engine",
)


def _get_engine_nodes_in(nodes: Any) -> List[Any]:
    """Return the TRT engine nodes in an iterable of FX nodes (graph or partition)."""
    return [
        node
        for node in nodes
        if node.op == "call_function"
        and _schema_name(node.target) in _ENGINE_OP_SCHEMA_NAMES
    ]


def _get_engine_nodes_from_edge_program(edge_program: ExportedProgram) -> List[Any]:
    """Return all TRT engine nodes found in a lowered ExecuTorch partition."""
    return _get_engine_nodes_in(edge_program.graph_module.graph.nodes)


def _get_single_engine_node(edge_program: ExportedProgram) -> Any:
    """Return the partition's one TRT engine node, or raise if not exactly one."""
    engine_nodes = _get_engine_nodes_from_edge_program(edge_program)
    if len(engine_nodes) != 1:
        raise RuntimeError(
            "TensorRT ExecuTorch backend expects exactly 1 engine node per "
            f"partition, found {len(engine_nodes)}."
        )
    return engine_nodes[0]


def _get_engine_info_from_edge_program(edge_program: ExportedProgram) -> List[Any]:
    """Extract engine info (list of strings/bytes) from the partition's TRT node.

    Handles two cases:
    - no_op_placeholder_for_execute_engine: engine info is embedded directly as
      string args (args[1:]) — used when _replace_execute_engine_for_executorch
      converted the graph before to_edge_transform_and_lower.
    - execute_engine: engine info is read off the ScriptObject through
      :func:`get_engine_info_from_state` — fallback for graphs not yet converted.

    Uses schema name comparison (not object identity) so it works for both
    OpOverload and EdgeOpOverload targets.
    """
    return _get_engine_info_for_node(
        edge_program, _get_single_engine_node(edge_program)
    )


def _get_engine_info_for_node(
    edge_program: ExportedProgram, node: torch.fx.Node, *, metadata_only: bool = False
) -> List[Any]:
    # Engine-info extraction for a single TRT node; callable per-partition so a
    # coalesced multi-engine graph can resolve each engine without the
    # whole-program "exactly 1 engine" assumption.
    #
    # metadata_only is forwarded verbatim to get_engine_info_from_state; its docstring
    # states what that flag costs and which slot it leaves unreadable.
    gm = edge_program.graph_module
    name = _schema_name(node.target)

    if name == "tensorrt::no_op_placeholder_for_execute_engine":
        engine_info = list(node.args[1:])
        # ENGINE_IDX slot is either a `get_attr` FX node (when this runs
        # before constant-lifting) or a `placeholder` FX node (after
        # ExecuTorch's lifter rewrote the get_attr into a graph input
        # referencing the buffer). Resolve both shapes to the raw uint8
        # tensor so the rest of the backend can stay engine-format
        # agnostic.
        engine_slot = engine_info[ENGINE_IDX]
        if isinstance(engine_slot, torch.fx.Node):
            engine_tensor = None
            if engine_slot.op == "get_attr":
                engine_tensor = getattr(gm, engine_slot.target, None)
            elif engine_slot.op == "placeholder":
                # The lifter mangles the placeholder name (e.g.
                # "b__trt_engine_0" with a "b_" buffer prefix). The
                # canonical attribute target lives in
                # graph_signature.input_specs[i].target.
                target = engine_slot.target
                sig = getattr(edge_program, "graph_signature", None)
                if sig is not None:
                    for ispec in sig.input_specs:
                        arg = getattr(ispec, "arg", None)
                        if (
                            arg is not None
                            and getattr(arg, "name", None) == engine_slot.name
                        ):
                            target = ispec.target or target
                            break
                state_dict = getattr(edge_program, "state_dict", {}) or {}
                constants = getattr(edge_program, "constants", {}) or {}
                # Explicit None-check: `state_dict.get(target) or ...`
                # would call `bool(tensor)`, which raises
                # "Boolean value of Tensor with more than one element
                # is ambiguous" for any multi-element engine tensor.
                engine_tensor = state_dict.get(target)
                if engine_tensor is None:
                    engine_tensor = constants.get(target)
            else:
                raise RuntimeError(
                    f"no_op_placeholder node '{node.name}': unexpected engine "
                    f"slot op '{engine_slot.op}' (target={engine_slot.target})"
                )
            if engine_tensor is None:
                raise RuntimeError(
                    f"no_op_placeholder node '{node.name}': engine slot "
                    f"'{engine_slot.target}' (op={engine_slot.op}) did not "
                    f"resolve to a tensor in gm, state_dict, or constants"
                )
            engine_info[ENGINE_IDX] = engine_tensor
        return engine_info

    engine_node = node.args[1]
    if engine_node.op == "get_attr":
        engine_obj = getattr(gm, engine_node.target, None)
        if engine_obj is None:
            raise RuntimeError(
                f"execute_engine node '{node.name}': get_attr target "
                f"'{engine_node.target}' not found on graph module"
            )
    elif engine_node.op == "placeholder":
        engine_obj = _resolve_lifted_custom_obj(edge_program, engine_node)
        if engine_obj is None:
            raise RuntimeError(
                f"execute_engine node '{node.name}': placeholder engine "
                f"'{engine_node.name}' did not resolve to a lifted custom-object "
                f"constant (available: "
                f"{sorted(getattr(edge_program, 'constants', {}) or {})})"
            )
    else:
        raise RuntimeError(
            f"execute_engine node '{node.name}': unexpected engine arg op "
            f"'{engine_node.op}'"
        )

    from torch_tensorrt.executorch._export_utils import get_engine_info_from_state

    state: List[Any] = get_engine_info_from_state(
        engine_obj, metadata_only=metadata_only
    )
    return state


def _validate_engine_info(engine_info: List[Any]) -> None:
    if len(engine_info) <= ENGINE_IDX:
        raise RuntimeError(
            "TensorRT ExecuTorch backend received incomplete engine "
            "serialization info."
        )
    if (
        len(engine_info) > REQUIRES_OUTPUT_ALLOCATOR_IDX
        and str(engine_info[REQUIRES_OUTPUT_ALLOCATOR_IDX]) == "1"
    ):
        raise RuntimeError(
            "ExecuTorch export does not support TensorRT engines that require "
            "an output allocator (data-dependent output shapes)."
        )


def _split_binding_names(value: Any) -> List[str]:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return [name for name in str(value or "").split(_BINDING_DELIM) if name]


def _parse_device_id(value: Any) -> int:
    parts = str(value or "").split(_BINDING_DELIM)
    try:
        return int(parts[0])
    except (IndexError, ValueError):
        return 0


def _reorder_input_names_for_executorch(
    edge_program: ExportedProgram, engine_node: Any, input_names: List[str]
) -> List[str]:
    """Reorder TRT binding names into executorch_call_delegate argument order.

    The runtime binds positionally (``execute()`` arg ``i`` -> input_binding_names
    ``[i]``), but ExecuTorch fusion may permute the delegate placeholders relative
    to the TRT-submodule order that produced ``input_binding_names``. The names
    can't be matched (TRT names are semantic, lowered placeholders are generic
    ``arg_N``), so recover the permutation by node identity: the engine node's
    first arg lists its input nodes in binding order, so sort the names by each
    node's slot among the graph placeholders (its runtime delegate-arg position).

    Only inputs need this. Outputs are also bound positionally, but they are
    ``getitem(engine_node, idx)`` nodes whose index order equals the engine
    output-binding order, and that order survives lowering -- though not because
    the partition is mutation-free. With aliased-I/O (KV-cache) support a TensorRT
    partition *does* produce mutation outputs, and ``arrange_graph_outputs`` does
    move buffer-mutation outputs ahead of user outputs. It stays a no-op here
    because ``_keep_mutated_buffers_above_delegate`` (``partitioner.py``) strips
    the ``delegation_tag`` from mutated buffer placeholders, so they stay out of
    the delegate's state dict and constants; ExecuTorch's ``_get_new_signature``
    then records the mutation as a plain ``USER_OUTPUT`` rather than a
    ``BUFFER_MUTATION`` (it uses the latter only when the delegate itself consumes
    the buffer). The lowered submodule therefore has no mutation specs, so
    ``arrange_graph_outputs`` computes the identity permutation and the getitem
    indices still line up with the engine's output bindings.

    That guarantee is conditional, not structural: if a mutated buffer is ever
    tagged into a delegate its spec becomes ``BUFFER_MUTATION``, the delegate's
    outputs are permuted, and they would need the same node-identity reordering as
    the inputs below. ``_validate_output_binding_order`` checks that correspondence
    on every preprocess, so it would fail loudly rather than mis-bind.
    """
    input_nodes = list(engine_node.args[0])
    if len(input_nodes) != len(input_names):
        raise ValueError(
            "TensorRT ExecuTorch backend: engine has "
            f"{len(input_names)} input binding names but {len(input_nodes)} "
            "engine input nodes; cannot establish a reliable binding order."
        )
    slot = {
        node: i
        for i, node in enumerate(
            n for n in edge_program.graph_module.graph.nodes if n.op == "placeholder"
        )
    }
    missing = [n for n in input_nodes if n not in slot]
    if missing:
        raise ValueError(
            "TensorRT ExecuTorch backend: engine inputs "
            f"{[n.name for n in missing]} are not delegate runtime placeholders; "
            "cannot determine their argument position."
        )
    order = sorted(range(len(input_nodes)), key=lambda i: slot[input_nodes[i]])
    return [input_names[i] for i in order]


def _validate_output_binding_order(
    edge_program: ExportedProgram,
    engine_node: Any,
    output_names: List[str],
    elidable_output_names: Optional[Container[str]] = None,
) -> None:
    """Check the delegate's outputs are the engine's output bindings, in order.

    The runtime binds output ``i`` to ``output_binding_names[i]``, and nothing
    downstream re-derives that correspondence: it holds because the partition's
    outputs are ``getitem(engine_node, i)`` in index order. A pass that reordered
    them -- ``arrange_graph_outputs`` moves buffer mutations ahead of user outputs,
    which only stays a no-op here while the mutated buffers are kept above the
    delegate -- would swap the names silently. Inputs cannot rely on position at
    all and recover their order by node identity in
    ``_reorder_input_names_for_executorch``.

    ``elidable_output_names`` names the bindings the delegate is *allowed* to
    have dropped, which zero-copy KV sets to exactly the aliased outputs export
    rewired to write in place (never the whole aliased_io): the engine's in-place
    write through the aliased input already is the buffer update, so no argument
    is passed for them. Pass ``None`` (the default) when elision was not asked
    for, and the delegate must carry every binding.

    A delegate that dropped its aliased outputs because nothing declared them as
    mutations looks exactly like a zero-copy one, and the runtime reads elision off
    a single argument count, so it cannot tell them apart either. That is also why
    a partial drop stays an error: the count cannot express which bindings went.
    """
    elidable_names = elidable_output_names if elidable_output_names is not None else ()
    all_indices = list(range(len(output_names)))
    unaliased_indices = [
        i for i in all_indices if output_names[i] not in elidable_names
    ]

    output_node = next(
        node for node in edge_program.graph_module.graph.nodes if node.op == "output"
    )
    out_args = list(output_node.args[0])
    # A single-output engine is returned directly rather than through a getitem,
    # and one binding has no order to get wrong. The same holds under elision
    # when exactly one binding is left unaliased.
    if len(out_args) == 1 and out_args[0] is engine_node:
        if len(all_indices) != 1 and len(unaliased_indices) != 1:
            remaining = (
                f", {len(unaliased_indices)} of them after eliding the in-place "
                "outputs"
                if unaliased_indices != all_indices
                else ""
            )
            raise ValueError(
                "TensorRT ExecuTorch backend: the delegate returns the engine node "
                f"directly but the engine declares {len(output_names)} output "
                f"bindings{remaining}; only a single-output engine can be returned "
                "unwrapped."
            )
        return
    indices: List[Any] = []
    for node in out_args:
        if (
            not isinstance(node, torch.fx.Node)
            or node.op != "call_function"
            or node.target is not operator.getitem
            or node.args[0] is not engine_node
        ):
            raise ValueError(
                "TensorRT ExecuTorch backend: delegate output "
                f"{getattr(node, 'name', node)!r} is not a getitem of the engine "
                "node; cannot establish a reliable output binding order."
            )
        indices.append(node.args[1])
    if indices not in (all_indices, unaliased_indices):
        expected = (
            f"{all_indices}, or {unaliased_indices} with the in-place outputs elided"
            if unaliased_indices != all_indices
            else f"{all_indices}"
        )
        # Outputs are missing and nothing exempted them. That is what an export
        # asking for zero_copy_kv looks like when the aliased-buffer mark did not
        # reach the partitioner, so no delegate was stamped and none is exempt --
        # a failure mode with no other symptom, hence naming it here.
        unexempted_drop = elidable_output_names is None and len(indices) < len(
            all_indices
        )
        raise ValueError(
            "TensorRT ExecuTorch backend: delegate outputs map to engine output "
            f"indices {indices}, expected {expected} -- the runtime binds each "
            "output it is given in binding order, so a permuted, incomplete, or "
            "partially elided output list would bind the wrong tensors."
            + (
                " No output was declared elidable for this delegate, so if the "
                "export asked for zero_copy_kv the aliased-buffer mark did not "
                "survive lowering."
                if unexempted_drop
                else ""
            )
        )


def _serialize_elided_output_names(names: Iterable[str]) -> bytes:
    """Encode the elided aliased-output binding names for the compile spec.

    JSON, not the ``%`` / ``@`` delimiters that separate ``engine_info``'s
    binding-name and aliased_io fields, so a binding name containing one of those
    cannot corrupt the record.
    """
    return json.dumps(sorted(set(names))).encode("utf-8")


def _elided_output_names(compile_specs: List[CompileSpec]) -> Optional[Set[str]]:
    """The aliased-output binding names export declared elidable, or ``None``.

    ``None`` when no zero-copy spec is present, which keeps a missing output an
    error: only a caller who asked for zero-copy may drop the aliased outputs,
    and then only exactly the ones export rewired to write in place.
    """
    for spec in compile_specs:
        if getattr(spec, "key", None) != ZERO_COPY_KV_COMPILE_SPEC_KEY:
            continue
        value = spec.value
        if isinstance(value, (bytes, bytearray)):
            value = bytes(value).decode("utf-8")
        return set(json.loads(value))
    return None


def _get_str(engine_info: List[Any], index: int, default: str = "") -> str:
    if index < 0 or index >= len(engine_info):
        return default
    value = engine_info[index]
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


@final
class TensorRTBackend(BackendDetails):  # type: ignore[misc]
    """Backend that serializes TensorRT engines for the native ExecuTorch runtime.

    The partition contains a single execute_engine node; we extract the engine
    and metadata and encode them as a standalone blob. The C++ runtime
    backend parses that blob directly without the legacy Torch-TensorRT C++ runtime.
    """

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        engine_node = _get_single_engine_node(edge_program)
        engine_info = _get_engine_info_for_node(edge_program, engine_node)
        engine_info = list(engine_info)
        _validate_engine_info(engine_info)
        serialized_engine = engine_info[ENGINE_IDX]
        if isinstance(serialized_engine, torch.Tensor):
            # `bytes(storage)` looks equivalent but has two problems. It iterates
            # the storage element by element in Python, costing about two seconds
            # per megabyte, and it returns the whole backing allocation rather than
            # the tensor's own extent, so a view of a larger buffer serializes too
            # many bytes. `memoryview` is not redundant here: on a 0-dim uint8
            # tensor numpy alone goes through `__index__` and yields that many zero
            # bytes instead of the value. `.view(torch.uint8)` keeps `.numpy()` from
            # rejecting a dtype it has no equivalent for.
            engine_bytes = serialized_engine.cpu().contiguous().view(torch.uint8)
            engine_info[ENGINE_IDX] = bytes(memoryview(engine_bytes.numpy()))
        elif not isinstance(serialized_engine, (bytes, bytearray)):
            engine_info[ENGINE_IDX] = bytes(serialized_engine)
        input_names = _reorder_input_names_for_executorch(
            edge_program,
            engine_node,
            _split_binding_names(_get_str(engine_info, INPUT_BINDING_NAMES_IDX)),
        )
        output_names = _split_binding_names(
            _get_str(engine_info, OUTPUT_BINDING_NAMES_IDX)
        )
        _validate_output_binding_order(
            edge_program,
            engine_node,
            output_names,
            _elided_output_names(compile_specs),
        )
        io_bindings = [
            TensorRTIOBinding(name=name, is_input=True) for name in input_names
        ] + [TensorRTIOBinding(name=name, is_input=False) for name in output_names]

        # Carry the KV-cache / user aliasing (out->in, kind) into the blob so the
        # C++ backend binds each aliased output to its aliased input's tensor
        # (in-place) and reflects the update back into the delegate output.
        aliased_io = deserialize_aliased_io(_get_str(engine_info, ALIASED_IO_IDX))

        metadata = TensorRTBlobMetadata(
            io_bindings=io_bindings,
            aliased_io=aliased_io,
            hardware_compatible=_get_str(engine_info, HW_COMPATIBLE_IDX) == "1",
            device_id=_parse_device_id(engine_info[DEVICE_IDX]),
            serialized_metadata=_get_str(engine_info, SERIALIZED_METADATA_IDX),
            target_platform=_get_str(engine_info, TARGET_PLATFORM_IDX),
        )
        blob = serialize_engine(bytes(engine_info[ENGINE_IDX]), metadata)
        return PreprocessResult(processed_bytes=blob)
