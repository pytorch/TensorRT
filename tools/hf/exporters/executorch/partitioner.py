"""Partition named Edge-LLM component operators for ExecuTorch."""

from __future__ import annotations

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

from .backend import EdgeLLMBackend
from .operator_support import EdgeLLMOperatorSupport

try:
    from executorch.exir.passes.propagate_device_pass import (
        TARGET_DEVICE_COMPILE_SPEC_KEY as _TARGET_DEVICE_COMPILE_SPEC_KEY,
    )
except ImportError:
    _TARGET_DEVICE_COMPILE_SPEC_KEY = "target_device"


class EdgeLLMPartitioner(Partitioner):  # type: ignore[misc]
    """Creates one EdgeLLMBackend partition per named component operator."""

    def __init__(
        self,
        compile_specs: Optional[List[CompileSpec]] = None,
    ) -> None:
        super().__init__()
        self.compile_specs = list(compile_specs or [])
        if not any(
            spec.key == _TARGET_DEVICE_COMPILE_SPEC_KEY for spec in self.compile_specs
        ):
            self.compile_specs.append(
                CompileSpec(_TARGET_DEVICE_COMPILE_SPEC_KEY, b"cuda:0")
            )
        self.delegation_spec = DelegationSpec(
            backend_id=EdgeLLMBackend.__name__,
            compile_specs=self.compile_specs,
        )

    def partition(self, exported_program: ExportedProgram) -> PartitionResult:
        partitions = CapabilityBasedPartitioner(
            exported_program.graph_module,
            EdgeLLMOperatorSupport(),
            allows_single_node_partition=True,
        ).propose_partitions()

        partition_tags: Dict[str, DelegationSpec] = {}
        for partition in partitions:
            tag = f"edge_llm_{partition.id}"
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
        del ep
        return ([], None)
