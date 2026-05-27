# ExecuTorch partitioner: partition by execute_engine nodes.

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
from torch_tensorrt.executorch.backend import TensorRTBackend
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
    """

    def __init__(
        self,
        compile_specs: Optional[List[CompileSpec]] = None,
    ) -> None:
        super().__init__()
        self.compile_specs = list(compile_specs) if compile_specs else []
        # Mirror CudaPartitioner: emit a target_device CompileSpec so that
        # ExecuTorch's PropagateDevicePass tags delegate I/O TensorSpecs with
        # the correct device, which is then serialized into the .pte's
        # extra_tensor_info.device_type field.
        if not any(
            s.key == _TARGET_DEVICE_COMPILE_SPEC_KEY for s in self.compile_specs
        ):
            self.compile_specs.append(
                CompileSpec(_TARGET_DEVICE_COMPILE_SPEC_KEY, b"cuda:0")
            )
        self.delegation_spec = DelegationSpec(
            backend_id=TensorRTBackend.__name__,
            compile_specs=self.compile_specs,
        )

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
            partition_tags[tag] = self.delegation_spec

        tag_constant_data(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program,
            partition_tags=partition_tags,
        )

    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        return ([], None)
