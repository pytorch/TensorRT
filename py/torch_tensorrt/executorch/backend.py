# ExecuTorch TensorRT backend: serialize engine to same blob format as TRT runtime.

import base64
from typing import Any, List, final

from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)
from torch.export.exported_program import ExportedProgram
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import ENGINE_IDX
from torch_tensorrt.executorch.serialization import serialize_engine_info


def _schema_name(target: Any) -> str:
    """Return the qualified op schema name for an OpOverload or EdgeOpOverload."""
    if hasattr(target, "_schema"):
        return str(target._schema.name)
    return ""


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

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue

        name = _schema_name(node.target)

        if name == "tensorrt::no_op_placeholder_for_execute_engine":
            # Engine info is stored directly as string args (indices 0-9).
            return list(node.args[1:])

        if name == "tensorrt::execute_engine":
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
            return list(engine_obj.__getstate__())

    raise RuntimeError(
        "TensorRT ExecuTorch backend: no execute_engine or "
        "no_op_placeholder_for_execute_engine node found in partition."
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
        serialized_engine = engine_info[ENGINE_IDX]
        if isinstance(serialized_engine, str):
            engine_info[ENGINE_IDX] = base64.b64decode(
                serialized_engine.encode("utf-8")
            )
        elif not isinstance(serialized_engine, (bytes, bytearray)):
            engine_info[ENGINE_IDX] = bytes(serialized_engine)
        if len(engine_info) > 7 and isinstance(engine_info[7], bytes):
            engine_info[7] = engine_info[7].decode("utf-8", errors="replace")
        blob = serialize_engine_info(engine_info)
        return PreprocessResult(processed_bytes=blob)
