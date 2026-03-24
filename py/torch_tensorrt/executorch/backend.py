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
    """Extract engine info (list of strings/bytes) from the partition's no_op_placeholder node.

    Before calling to_edge_transform_and_lower, _save_as_executorch replaces
    execute_engine nodes with no_op_placeholder_for_execute_engine whose args are
    (inputs_tuple, abi_version, name, device, engine_b64, in_names, out_names,
    hw_compat, metadata, platform, requires_oa).  This function reads those flat
    args back out and returns them as a list indexed by SerializedInfoIndex.
    """
    gm = edge_program.graph_module
    no_op = torch.ops.tensorrt.no_op_placeholder_for_execute_engine.default

    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target is not no_op:
            continue
        # args layout: (inputs_tuple, *engine_info_strings)
        # engine_info_strings has SERIALIZATION_LEN - 1 entries (no RESOURCE_ALLOCATION_STRATEGY)
        if len(node.args) < 2:
            raise RuntimeError(
                f"no_op_placeholder node '{node.name}' has too few args: {len(node.args)}"
            )
        engine_info = list(node.args[1:])
        if len(engine_info) < SERIALIZATION_LEN - 1:
            raise RuntimeError(
                f"no_op_placeholder node '{node.name}' has {len(engine_info)} engine "
                f"info args, expected at least {SERIALIZATION_LEN - 1}"
            )
        return engine_info

    raise RuntimeError(
        "TensorRT ExecuTorch backend: no no_op_placeholder_for_execute_engine "
        "node found in partition."
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
