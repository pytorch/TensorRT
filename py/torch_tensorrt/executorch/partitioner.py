# ExecuTorch partitioner: partition by execute_engine nodes.

import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.utils import tag_constant_data
from torch.export import ExportedProgram
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import DEVICE_IDX
from torch_tensorrt.executorch.backend import (
    TensorRTBackend,
    _get_engine_info_from_edge_program,
    _parse_device_id,
)
from torch_tensorrt.executorch.operator_support import TensorRTOperatorSupport

# Key recognized by ExecuTorch's PropagateDevicePass that tags delegate I/O
# TensorSpecs with the target device, which is then serialized into the
# .pte's extra_tensor_info.device_type field.
#
# Prefer the canonical constant when ExecuTorch exposes it (will fail loudly
# at import time if the key is renamed upstream) and fall back to the inlined
# string for older ExecuTorch revisions that don't yet ship the constant.
try:
    from executorch.exir.passes.propagate_device_pass import (
        TARGET_DEVICE_COMPILE_SPEC_KEY as _TARGET_DEVICE_COMPILE_SPEC_KEY,
    )
except ImportError:
    _TARGET_DEVICE_COMPILE_SPEC_KEY = "target_device"

logger = logging.getLogger(__name__)


class TensorRTPartitioner(Partitioner):  # type: ignore[misc]
    """Partitions the graph for TensorRT delegation.

    Only nodes that are torch.ops.tensorrt.execute_engine are supported;
    each such node becomes its own partition so the backend can serialize
    the engine to the same format as the TRT runtime.

    If `compile_specs` does not already contain a ``target_device`` entry,
    one defaulting to ``cuda:0`` is auto-appended (mirroring CudaPartitioner).
    Callers targeting a non-default GPU should pre-populate
    ``compile_specs`` with the desired ``CompileSpec("target_device",
    b"cuda:<index>")`` to override the default.

    Note: ``target_device`` is AOT metadata only -- it drives ExecuTorch's
    PropagateDevicePass tagging at export time. At runtime the C++ backend
    selects the GPU from the device baked into the serialized engine blob,
    not from this value.
    """

    def __init__(
        self,
        compile_specs: Optional[List[CompileSpec]] = None,
    ) -> None:
        super().__init__()
        self.compile_specs = list(compile_specs) if compile_specs else []
        # Mirror CudaPartitioner: a target_device CompileSpec drives ExecuTorch's
        # PropagateDevicePass, which tags delegate I/O TensorSpecs with the device
        # and serializes it into the .pte's extra_tensor_info. When the caller pins
        # it we use that verbatim; otherwise it is derived per export from the
        # engine's real device in partition() (engine nodes are not available here)
        # so a cuda:N engine is not mislabeled cuda:0.
        self._has_explicit_target_device = any(
            s.key == _TARGET_DEVICE_COMPILE_SPEC_KEY for s in self.compile_specs
        )
        self.delegation_spec = DelegationSpec(
            backend_id=TensorRTBackend.__name__,
            compile_specs=self.compile_specs,
        )

    def _resolve_target_device(self, exported_program: ExportedProgram) -> bytes:
        """Best-effort ``target_device`` for the delegate-boundary TensorSpecs.

        Reuses the backend's own engine-info extraction so the device index
        cannot drift from the runtime blob. Any extraction failure -- no single
        engine node (zero or multiple TRT partitions) or an unreadable index --
        falls back to ``cuda:0``; per-partition multi-GPU labeling is left to a
        follow-up.
        """
        try:
            engine_info = _get_engine_info_from_edge_program(exported_program)
            return f"cuda:{_parse_device_id(engine_info[DEVICE_IDX])}".encode()
        except Exception as e:
            # Broad by design: any extraction failure must fall back, not abort
            # the export. Warn so a non-default GPU silently labeled cuda:0 stays
            # diagnosable.
            logger.warning(
                "Could not derive target_device from the TensorRT engine (%s); "
                "falling back to cuda:0. A non-default GPU engine may be "
                'mislabeled -- pin it via CompileSpec("target_device", '
                'b"cuda:<index>").',
                e,
            )
            return b"cuda:0"

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        capability_partitioner = CapabilityBasedPartitioner(
            exported_program.graph_module,
            TensorRTOperatorSupport(),
            allows_single_node_partition=True,
        )
        partition_list = capability_partitioner.propose_partitions()

        if self._has_explicit_target_device:
            delegation_spec = self.delegation_spec
        else:
            delegation_spec = DelegationSpec(
                backend_id=TensorRTBackend.__name__,
                compile_specs=self.compile_specs
                + [
                    CompileSpec(
                        _TARGET_DEVICE_COMPILE_SPEC_KEY,
                        self._resolve_target_device(exported_program),
                    )
                ],
            )

        partition_tags: Dict[str, DelegationSpec] = {}
        for partition in partition_list:
            tag = f"tensorrt_{partition.id}"
            for node in partition.nodes:
                node.meta["delegation_tag"] = tag
            partition_tags[tag] = delegation_spec

        tag_constant_data(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program,
            partition_tags=partition_tags,
        )

    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        return ([], None)
