# ExecuTorch TensorRT backend: serialize engine to same blob format as TRT runtime.

import base64
from typing import Any, List, final

import torch
from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)
from torch.export.exported_program import ExportedProgram
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
    ENGINE_IDX,
    SERIALIZATION_LEN,
)
from torch_tensorrt.executorch.serialization import serialize_engine_info


def _get_engine_info_from_edge_program(edge_program: ExportedProgram) -> List[Any]:
    """Extract engine info (list of strings/bytes) from the partition's execute_engine node."""
    gm = edge_program.graph_module
    execute_engine_op = torch.ops.tensorrt.execute_engine.default

    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target is not execute_engine_op:
            continue
        if len(node.args) < 2:
            continue
        engine_arg = node.args[1]
        if engine_arg.op == "get_attr":
            val = getattr(gm, engine_arg.target, None)
            if val is None:
                raise RuntimeError(
                    f"Engine get_attr({engine_arg.target}) not found on partition module."
                )
            if hasattr(val, "__getstate__"):
                engine_info = val.__getstate__()
            else:
                engine_info = getattr(val, "engine_info", val)
            if (
                isinstance(engine_info, (list, tuple))
                and len(engine_info) >= SERIALIZATION_LEN
            ):
                return list(engine_info)
            raise RuntimeError(
                f"Engine argument get_attr({engine_arg.target}) did not yield engine info list (len >= {SERIALIZATION_LEN})."
            )
        raise RuntimeError(
            "TensorRT ExecuTorch backend expects execute_engine(inputs, engine) "
            "where engine is a get_attr; cannot find engine."
        )
    raise RuntimeError(
        "TensorRT ExecuTorch backend: no execute_engine node found in partition."
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
