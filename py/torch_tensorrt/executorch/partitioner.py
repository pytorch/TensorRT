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


class TensorRTPartitioner(Partitioner):  # type: ignore[misc]
    """Partitions the graph for TensorRT delegation.

    Only nodes that are torch.ops.tensorrt.execute_engine are supported;
    each such node becomes its own partition so the backend can serialize
    the engine to the same format as the TRT runtime.
    """

    def __init__(
        self,
        compile_specs: Optional[List[CompileSpec]] = None,
    ) -> None:
        super().__init__()
        self.compile_specs = compile_specs or []
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
