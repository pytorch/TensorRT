# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Helper for saving torch_tensorrt-compiled modules to ExecuTorch (.pte) format.
# Uses the same TensorRT blob format (TR01) as executorch.backends.nvidia.tensorrt
# so the resulting .pte can be loaded by the ExecuTorch TensorRT runtime.

from __future__ import annotations

import json
import logging
import struct
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, cast

import torch

logger = logging.getLogger(__name__)

# TR01 blob format (aligned with executorch backends/nvidia/tensorrt/serialization.py)
_TENSORRT_MAGIC = b"TR01"
_HEADER_SIZE = 32
_HEADER_FORMAT = "<4sIIIQ8s"


def _align_to_16(offset: int) -> int:
    return (offset + 15) & ~15


@dataclass
class _IOBinding:
    name: str
    dtype: str
    shape: List[int]
    is_input: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class _TensorRTBlobMetadata:
    io_bindings: List[_IOBinding] = field(default_factory=list)

    def to_json(self) -> bytes:
        data = {"io_bindings": [b.to_dict() for b in self.io_bindings]}
        return json.dumps(data, separators=(",", ":")).encode("utf-8")


def _serialize_tr01_blob(
    engine_bytes: bytes, metadata: Optional[_TensorRTBlobMetadata] = None
) -> bytes:
    if metadata is None:
        metadata = _TensorRTBlobMetadata()
    metadata_json = metadata.to_json()
    metadata_size = len(metadata_json)
    metadata_offset = _HEADER_SIZE
    engine_offset = _align_to_16(metadata_offset + metadata_size)
    engine_size = len(engine_bytes)
    reserved = b"\x00" * 8
    header = struct.pack(
        _HEADER_FORMAT,
        _TENSORRT_MAGIC,
        metadata_offset,
        metadata_size,
        engine_offset,
        engine_size,
        reserved,
    )
    padding_size = engine_offset - (metadata_offset + metadata_size)
    padding = b"\x00" * padding_size
    return header + metadata_json + padding + engine_bytes


def _get_execute_engine_op() -> Optional[Any]:
    if not hasattr(torch.ops, "tensorrt") or not hasattr(
        torch.ops.tensorrt, "execute_engine"
    ):
        return None
    return torch.ops.tensorrt.execute_engine


def _get_execute_engine_default_op() -> Optional[Any]:
    """Return the .default OpOverload used as the graph node target (not the OpOverloadGroup)."""
    op = _get_execute_engine_op()
    if op is None:
        return None
    if hasattr(op, "default"):
        return op.default
    return op


def _get_engine_idx() -> Optional[int]:
    try:
        from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import ENGINE_IDX

        return cast(int, ENGINE_IDX)
    except Exception:
        return None


def _extract_engine_bytes_from_partition(exported_program: Any) -> Optional[bytes]:
    gm = exported_program.graph_module
    engine_idx = _get_engine_idx()
    if engine_idx is None:
        return None
    op = _get_execute_engine_default_op()
    if op is None:
        return None
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target != op:
            continue
        if len(node.args) < 2:
            continue
        engine_arg = node.args[1]
        engine_obj = None
        if engine_arg.op == "get_attr":
            try:
                engine_obj = getattr(gm, engine_arg.target)
            except AttributeError:
                pass
        if engine_obj is None:
            continue
        if hasattr(engine_obj, "engine"):
            engine_obj = engine_obj.engine
        if hasattr(engine_obj, "__getstate__"):
            state = engine_obj.__getstate__()
            if isinstance(state, (list, tuple)) and len(state) > engine_idx:
                raw = state[engine_idx]
                if isinstance(raw, bytes):
                    return raw
    return None


def _build_metadata_from_partition(exported_program: Any) -> _TensorRTBlobMetadata:
    bindings = []
    gm = exported_program.graph_module
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            val = node.meta.get("val")
            shape = list(getattr(val, "shape", ())) if val is not None else []
            dtype = getattr(val, "dtype", torch.float32)
            dtype_str = str(dtype) if dtype is not None else "float32"
            if hasattr(dtype, "name"):
                dtype_str = getattr(dtype, "name", dtype_str)
            bindings.append(
                _IOBinding(name=node.name, dtype=dtype_str, shape=shape, is_input=True)
            )
    out_node = gm.graph.nodes[-1] if gm.graph.nodes else None
    if out_node and out_node.op == "output" and out_node.args:
        out_val = out_node.args[0]
        if isinstance(out_val, (list, tuple)) and out_val:
            for i, v in enumerate(out_val):
                if hasattr(v, "meta") and "val" in v.meta:
                    val = v.meta["val"]
                    shape = list(getattr(val, "shape", ()))
                    dtype = getattr(val, "dtype", torch.float32)
                    dtype_str = str(dtype) if dtype is not None else "float32"
                    if hasattr(dtype, "name"):
                        dtype_str = getattr(dtype, "name", dtype_str)
                    bindings.append(
                        _IOBinding(
                            name=f"output_{i}",
                            dtype=dtype_str,
                            shape=shape,
                            is_input=False,
                        )
                    )
    return _TensorRTBlobMetadata(io_bindings=bindings)


def _import_executorch() -> Tuple[Any, ...]:
    try:
        from executorch.exir import to_edge_transform_and_lower
        from executorch.exir.backend.backend_details import (
            BackendDetails,
            PreprocessResult,
        )
        from executorch.exir.backend.compile_spec_schema import CompileSpec
        from executorch.exir.backend.partitioner import (
            DelegationSpec,
            Partitioner,
            PartitionResult,
        )
        from executorch.exir.backend.utils import tag_constant_data

        return (
            BackendDetails,
            PreprocessResult,
            CompileSpec,
            DelegationSpec,
            Partitioner,
            PartitionResult,
            tag_constant_data,
            to_edge_transform_and_lower,
        )
    except ImportError as e:
        raise ImportError(
            "ExecuTorch is required for output_format='executorch'. "
            "Install with: pip install executorch"
        ) from e


def _get_backend_and_partitioner_classes() -> Tuple[Any, Any, Any]:
    (
        BackendDetails,
        PreprocessResult,
        CompileSpec,
        DelegationSpec,
        Partitioner,
        PartitionResult,
        tag_constant_data,
        to_edge_transform_and_lower,
    ) = _import_executorch()
    from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
    from torch.fx.passes.operator_support import OperatorSupportBase

    class TensorRTBackend(BackendDetails):  # type: ignore[valid-type,misc]
        """Backend that packages torch_tensorrt engine as TR01 blob; backend_id must be TensorRTBackend for ExecuTorch runtime."""

        @staticmethod
        def preprocess(
            edge_program: Any,
            compile_specs: List[Any],
        ) -> Any:
            engine_bytes = _extract_engine_bytes_from_partition(edge_program)
            if engine_bytes is None:
                raise RuntimeError(
                    "Could not extract TensorRT engine from partition. "
                    "The graph must contain torch.ops.tensorrt.execute_engine."
                )
            metadata = _build_metadata_from_partition(edge_program)
            blob = _serialize_tr01_blob(engine_bytes, metadata=metadata)
            return PreprocessResult(processed_bytes=blob)

    class ExecuteEngineOperatorSupport(OperatorSupportBase):  # type: ignore[misc]
        """Operator support for torch_tensorrt: only torch.ops.tensorrt.execute_engine is supported."""

        def is_node_supported(self, submodules: Any, node: torch.fx.Node) -> bool:
            if node.op != "call_function":
                return False
            op = _get_execute_engine_default_op()
            return op is not None and node.target == op

    class TorchTensorRTPartitioner(Partitioner):  # type: ignore[valid-type,misc]
        """Partitioner for TensorRT backend (torch_tensorrt pre-built engine)."""

        def __init__(self, compile_specs: Optional[List[Any]] = None) -> None:
            super().__init__()
            self.compile_specs = compile_specs or []
            self.delegation_spec = DelegationSpec(
                backend_id=TensorRTBackend.__name__,
                compile_specs=self.compile_specs,
            )

        def partition(self, exported_program: Any) -> Any:
            op = _get_execute_engine_default_op()
            if op is None:
                raise RuntimeError(
                    "torch.ops.tensorrt.execute_engine not found; is torch_tensorrt runtime loaded?"
                )
            gm = exported_program.graph_module
            if not any(
                n.op == "call_function" and n.target == op for n in gm.graph.nodes
            ):
                raise RuntimeError(
                    "Exported program has no torch.ops.tensorrt.execute_engine nodes. "
                    "Save as executorch only works for torch_tensorrt-compiled modules."
                )
            capability_partitioner = CapabilityBasedPartitioner(
                exported_program.graph_module,
                ExecuteEngineOperatorSupport(),
                allows_single_node_partition=True,
            )
            partition_list = capability_partitioner.propose_partitions()
            partition_tags: Dict[str, Any] = {}
            for partition in partition_list:
                tag = f"tensorrt_{partition.id}"
                for node in partition.nodes:
                    node.meta["delegation_tag"] = tag
                partition_tags[tag] = self.delegation_spec
            tag_constant_data(exported_program)
            return PartitionResult(
                tagged_exported_program=exported_program,
                partition_tags=partition_tags,
            )

    return TensorRTBackend, TorchTensorRTPartitioner, to_edge_transform_and_lower


def _execute_engine_fake_for_export(
    args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Optional[List[torch.Tensor]]:
    """Return fake outputs when execute_engine is called with CustomObjArgument (engine placeholder)."""
    if len(args) >= 2 and type(args[1]).__name__ == "CustomObjArgument":
        input_tensors = args[0]
        if (
            input_tensors
            and isinstance(input_tensors, (list, tuple))
            and len(input_tensors) > 0
        ):
            t = input_tensors[0]
            if hasattr(t, "device"):
                return [torch.empty_like(t, device=t.device)]
        return [torch.empty(1)]
    return None


def export_to_executorch(exported_program: Any, file_path: str) -> None:
    """Export an ExportedProgram (from a torch_tensorrt-compiled module) to a .pte file."""
    if _get_execute_engine_op() is None:
        raise RuntimeError(
            "torch.ops.tensorrt.execute_engine not found; is torch_tensorrt runtime loaded?"
        )
    op_default = _get_execute_engine_default_op()
    if op_default is None:
        raise RuntimeError("torch.ops.tensorrt.execute_engine.default not found.")
    TensorRTBackend, TorchTensorRTPartitioner, to_edge_transform_and_lower = (
        _get_backend_and_partitioner_classes()
    )
    partitioner = TorchTensorRTPartitioner()
    orig_call = op_default.__call__

    def patched_call(*args: Any, **kwargs: Any) -> Any:
        fake = _execute_engine_fake_for_export(args, kwargs)
        if fake is not None:
            return fake
        return orig_call(*args, **kwargs)

    patched_call.__name__ = getattr(orig_call, "__name__", "execute_engine")
    op_default.__call__ = patched_call
    try:
        edge_program = to_edge_transform_and_lower(
            exported_program,
            partitioner=[partitioner],
        )
        exec_prog = edge_program.to_executorch()
        with open(file_path, "wb") as f:
            exec_prog.write_to_file(f)
    finally:
        op_default.__call__ = orig_call
