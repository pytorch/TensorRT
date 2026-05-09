# ExecuTorch TensorRT backend: serialize engine to same blob format as TRT runtime.

import base64
from typing import Any, List, final

from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)
from torch.export.exported_program import ExportedProgram
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
    ENGINE_IDX,
    REQUIRES_OUTPUT_ALLOCATOR_IDX,
)
from torch_tensorrt.executorch.serialization import serialize_engine_info


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
        return list(node.args[1:])

    engine_node = node.args[1]
    if engine_node.op == "get_attr":
        engine_obj = getattr(gm, engine_node.target, None)
        if engine_obj is None:
            raise RuntimeError(
                f"execute_engine node '{node.name}': get_attr target "
                f"'{engine_node.target}' not found on graph module"
            )
    elif engine_node.op == "placeholder":
        constants = getattr(edge_program, "constants", {})
        engine_obj = constants.get(engine_node.name) or constants.get(
            engine_node.target
        )
        if engine_obj is None:
            raise RuntimeError(
                f"execute_engine node '{node.name}': placeholder engine "
                f"'{engine_node.name}' not found in edge_program.constants"
            )
    else:
        raise RuntimeError(
            f"execute_engine node '{node.name}': unexpected engine arg op "
            f"'{engine_node.op}'"
        )

    state = engine_obj.__getstate__()
    return list(state[0] if isinstance(state, tuple) else state)


def _validate_engine_info(engine_info: List[Any]) -> None:
    if not engine_info:
        raise RuntimeError(
            "TensorRT ExecuTorch backend received empty engine serialization info."
        )
    if (
        len(engine_info) > REQUIRES_OUTPUT_ALLOCATOR_IDX
        and str(engine_info[REQUIRES_OUTPUT_ALLOCATOR_IDX]) == "1"
    ):
        raise RuntimeError(
            "ExecuTorch export does not support TensorRT engines that require "
            "an output allocator (data-dependent output shapes)."
        )


@final
class TensorRTBackend(BackendDetails):  # type: ignore[misc]
    """Backend that serializes TensorRT engine to the same blob format as the TRT runtime.

    The partition contains a single execute_engine node; we extract the engine
    and metadata and encode them as a vector of strings (same layout as
    core/runtime/runtime.h SerializedInfoIndex) so the same blob works for
    both ExecuTorch and non-ExecuTorch TRT runtime.
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
        if isinstance(serialized_engine, str):
            try:
                engine_info[ENGINE_IDX] = base64.b64decode(
                    serialized_engine.encode("utf-8")
                )
            except Exception as exc:
                raise RuntimeError(
                    "TensorRT ExecuTorch backend failed to decode the serialized "
                    "engine payload."
                ) from exc
        elif not isinstance(serialized_engine, (bytes, bytearray)):
            engine_info[ENGINE_IDX] = bytes(serialized_engine)
        if len(engine_info) > 7 and isinstance(engine_info[7], bytes):
            engine_info[7] = engine_info[7].decode("utf-8", errors="replace")
        blob = serialize_engine_info(engine_info)
        return PreprocessResult(processed_bytes=blob)
