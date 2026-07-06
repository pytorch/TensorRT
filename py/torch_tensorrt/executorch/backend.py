# ExecuTorch TensorRT backend: serialize engines to a libtorch-free runtime blob.

from typing import Any, List, final

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
    DEVICE_IDX,
    ENGINE_IDX,
    HW_COMPATIBLE_IDX,
    INPUT_BINDING_NAMES_IDX,
    OUTPUT_BINDING_NAMES_IDX,
    REQUIRES_OUTPUT_ALLOCATOR_IDX,
    SERIALIZED_METADATA_IDX,
    TARGET_PLATFORM_IDX,
)
from torch_tensorrt.executorch.serialization import (
    TensorRTBlobMetadata,
    TensorRTIOBinding,
    serialize_engine,
)

_BINDING_DELIM = "%"


def _schema_name(target: Any) -> str:
    """Return the qualified op schema name for an OpOverload or EdgeOpOverload."""
    if hasattr(target, "_schema"):
        return str(target._schema.name)
    return ""


def _get_engine_nodes_from_edge_program(edge_program: ExportedProgram) -> List[Any]:
    """Return all TRT engine nodes found in a lowered ExecuTorch partition."""
    engine_nodes = []
    for node in edge_program.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        if _schema_name(node.target) in (
            "tensorrt::execute_engine",
            "tensorrt::no_op_placeholder_for_execute_engine",
        ):
            engine_nodes.append(node)
    return engine_nodes


def _get_engine_info_from_edge_program(edge_program: ExportedProgram) -> List[Any]:
    """Extract engine info (list of strings/bytes) from the partition's TRT node.

    Handles two cases:
    - no_op_placeholder_for_execute_engine: engine info is embedded directly as
      string args (args[1:]) — used when _replace_execute_engine_for_executorch
      converted the graph before to_edge_transform_and_lower.
    - execute_engine: engine info is extracted from the ScriptObject via
      __getstate__() — fallback for graphs not yet converted.

    Uses schema name comparison (not object identity) so it works for both
    OpOverload and EdgeOpOverload targets.
    """
    gm = edge_program.graph_module
    engine_nodes = _get_engine_nodes_from_edge_program(edge_program)
    if len(engine_nodes) != 1:
        raise RuntimeError(
            "TensorRT ExecuTorch backend expects exactly 1 engine node per "
            f"partition, found {len(engine_nodes)}."
        )

    node = engine_nodes[0]
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

    state = engine_obj.__getstate__()
    return list(state[0] if isinstance(state, tuple) else state)


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
    and metadata and encode them as a standalone TR01 blob. The C++ runtime
    backend parses that blob directly without the legacy Torch-TensorRT C++ runtime.
    """

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: List[CompileSpec],
    ) -> PreprocessResult:
        engine_info = _get_engine_info_from_edge_program(edge_program)
        engine_info = list(engine_info)
        _validate_engine_info(engine_info)
        serialized_engine = engine_info[ENGINE_IDX]
        if isinstance(serialized_engine, torch.Tensor):
            # Single copy out of the underlying storage. The prior
            # `.numpy().tobytes()` path allocated a fresh bytes buffer
            # on top of the numpy view, which for a >2 GB engine
            # roughly doubled peak memory at this step. `.cpu()` and
            # `.contiguous()` are no-ops when already host-side and
            # contiguous (the common case for the uint8 buffer this
            # backend produces).
            engine_info[ENGINE_IDX] = bytes(
                serialized_engine.cpu().contiguous().untyped_storage()
            )
        elif not isinstance(serialized_engine, (bytes, bytearray)):
            engine_info[ENGINE_IDX] = bytes(serialized_engine)
        input_names = _split_binding_names(
            _get_str(engine_info, INPUT_BINDING_NAMES_IDX)
        )
        output_names = _split_binding_names(
            _get_str(engine_info, OUTPUT_BINDING_NAMES_IDX)
        )
        io_bindings = [
            TensorRTIOBinding(name=name, is_input=True) for name in input_names
        ] + [TensorRTIOBinding(name=name, is_input=False) for name in output_names]

        metadata = TensorRTBlobMetadata(
            io_bindings=io_bindings,
            hardware_compatible=_get_str(engine_info, HW_COMPATIBLE_IDX) == "1",
            device_id=_parse_device_id(engine_info[DEVICE_IDX]),
            serialized_metadata=_get_str(engine_info, SERIALIZED_METADATA_IDX),
            target_platform=_get_str(engine_info, TARGET_PLATFORM_IDX),
        )
        blob = serialize_engine(bytes(engine_info[ENGINE_IDX]), metadata)
        return PreprocessResult(processed_bytes=blob)
