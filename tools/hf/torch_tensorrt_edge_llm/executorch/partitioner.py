"""Lower supported standalone ``tensorrt_edge_llm`` runtime operators."""

from __future__ import annotations

import operator
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

from ..serialization import EdgeComponentMetadata
from .backend import _COMPONENT_SCHEMAS, EdgeLLMBackend
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
        self._explicit_target = any(
            spec.key == _TARGET_DEVICE_COMPILE_SPEC_KEY for spec in self.compile_specs
        )
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
        support = EdgeLLMOperatorSupport()
        partition_tags: Dict[str, DelegationSpec] = {}
        for node in exported_program.graph.nodes:
            if not support.is_node_supported({}, node):
                continue
            # Each runtime module owns one engine payload and one native handle.
            # CapabilityBasedPartitioner can merge independent supported nodes,
            # which would make the single-component backend contract ambiguous.
            tag = f"edge_llm_{len(partition_tags)}"
            node.meta["delegation_tag"] = tag
            for user in node.users:
                if user.op == "call_function" and user.target is operator.getitem:
                    user.meta["delegation_tag"] = tag
            delegation = self.delegation_spec
            if not self._explicit_target:
                # ExecuTorch may substitute fake tensors for payload buffers;
                # the JSON operator argument remains available during partitioning.
                index = _COMPONENT_SCHEMAS[node.target._schema.name][1]
                metadata = EdgeComponentMetadata.from_json(node.args[index])
                device_id = metadata.runner_config.get("device_id", 0)
                if type(device_id) is not int or device_id < 0:
                    raise ValueError("Edge component device_id must be non-negative")
                device = f"cuda:{device_id}".encode()
                specs = [
                    spec
                    for spec in self.compile_specs
                    if spec.key != _TARGET_DEVICE_COMPILE_SPEC_KEY
                ]
                specs.append(CompileSpec(_TARGET_DEVICE_COMPILE_SPEC_KEY, device))
                delegation = DelegationSpec(
                    backend_id=EdgeLLMBackend.__name__, compile_specs=specs
                )
            partition_tags[tag] = delegation

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
