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
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner, Partition
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import DEVICE_IDX
from torch_tensorrt.executorch.backend import (
    TensorRTBackend,
    _get_engine_info_for_node,
    _get_engine_nodes_in,
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

    def _resolve_target_device_for_partition(
        self, exported_program: ExportedProgram, partition: Partition
    ) -> bytes:
        """Best-effort ``target_device`` for one partition's delegate boundary.

        Derives the device from this partition's own TRT engine node, so a
        coalesced multi-engine graph labels each delegate with its correct GPU
        instead of stamping every partition with a single whole-program value.
        Any extraction failure falls back to ``cuda:0``.
        """
        try:
            engine_nodes = _get_engine_nodes_in(partition.nodes)
            if len(engine_nodes) != 1:
                raise RuntimeError(
                    f"expected exactly 1 engine node in partition "
                    f"{getattr(partition, 'id', '?')}, found {len(engine_nodes)}"
                )
            engine_info = _get_engine_info_for_node(exported_program, engine_nodes[0])
            return f"cuda:{_parse_device_id(engine_info[DEVICE_IDX])}".encode()
        except Exception as e:
            # Broad by design: any extraction failure must fall back, not abort
            # the export. Warn so a non-default GPU silently labeled cuda:0 stays
            # diagnosable.
            logger.warning(
                "Could not derive target_device for partition %s (%s); falling "
                'back to cuda:0. Pin it via CompileSpec("target_device", '
                'b"cuda:<index>").',
                getattr(partition, "id", "?"),
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

        partition_tags: Dict[str, DelegationSpec] = {}
        for partition in partition_list:
            tag = f"tensorrt_{partition.id}"
            for node in partition.nodes:
                node.meta["delegation_tag"] = tag
            if self._has_explicit_target_device:
                partition_tags[tag] = self.delegation_spec
            else:
                partition_tags[tag] = DelegationSpec(
                    backend_id=TensorRTBackend.__name__,
                    compile_specs=self.compile_specs
                    + [
                        CompileSpec(
                            _TARGET_DEVICE_COMPILE_SPEC_KEY,
                            self._resolve_target_device_for_partition(
                                exported_program, partition
                            ),
                        )
                    ],
                )

        tag_constant_data(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program,
            partition_tags=partition_tags,
        )

    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        return ([], None)
