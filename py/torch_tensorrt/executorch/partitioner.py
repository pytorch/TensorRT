# ExecuTorch partitioner: partition by execute_engine nodes.

import logging
from typing import Callable, Dict, List, Optional, Set, Tuple

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
from torch_tensorrt.dynamo.runtime._serialized_engine_layout import (
    OUTPUT_BINDING_NAMES_IDX,
    deserialize_binding_names,
)
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import DEVICE_IDX
from torch_tensorrt.executorch.backend import (
    ZERO_COPY_KV_COMPILE_SPEC_KEY,
    TensorRTBackend,
    _get_engine_info_for_node,
    _get_engine_nodes_in,
    _parse_device_id,
    _serialize_elided_output_names,
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

# Compile spec key that carries the TensorRT weight streaming budget into the
# delegate. Must match kWeightStreamingBudgetKey on the C++ side
# (cpp/include/torch_tensorrt/executorch/WeightStreamingBudget.h). The value is a
# non-negative decimal integer of bytes (ASCII). The key is absent for the
# automatic budget, which the delegate applies itself for streamable engines.
# The delegate also reads this same key as a load-time backend option (runtime
# spec), which takes precedence over this baked value when provided at load.
WEIGHT_STREAMING_BUDGET_COMPILE_SPEC_KEY = "weight_streaming_budget"

# The C++ side parses the value into an int64_t, so anything at or above 2**63 has no
# representation there.
WEIGHT_STREAMING_BUDGET_MAX_BYTES = 2**63

logger = logging.getLogger(__name__)


def normalize_weight_streaming_budget_per_engine(
    weight_streaming_budget_per_engine: Optional[int],
) -> Optional[bytes]:
    """Validate a per-engine weight streaming budget and encode it for a CompileSpec.

    ``None``, the default, means automatic: the delegate picks the budget at load time.
    A non-negative integer is an explicit GPU budget in bytes. Returns the ASCII bytes
    to store in the CompileSpec, or ``None`` when no budget was supplied.
    """
    if weight_streaming_budget_per_engine is None:
        return None
    # bool is an int subclass, so reject it explicitly along with non-ints.
    if isinstance(weight_streaming_budget_per_engine, bool) or not isinstance(
        weight_streaming_budget_per_engine, int
    ):
        raise TypeError(
            "weight_streaming_budget_per_engine must be a non-negative int (number of "
            "bytes) or None for automatic, got "
            f"{type(weight_streaming_budget_per_engine).__name__}."
        )
    if not 0 <= weight_streaming_budget_per_engine < WEIGHT_STREAMING_BUDGET_MAX_BYTES:
        raise ValueError(
            "weight_streaming_budget_per_engine must be in [0, 2**63), got "
            f"{weight_streaming_budget_per_engine}."
        )
    return str(weight_streaming_budget_per_engine).encode("ascii")


def _keep_mutated_buffers_above_delegate(exported_program: ExportedProgram) -> None:
    """Undo tag_constant_data freezing a delegate-mutated buffer as constant.

    tag_constant_data detects a mutated buffer only via its *direct* users, so a
    buffer whose mutation is produced inside the delegate (the mutation is a
    getitem off the call_delegate, not a direct user of the buffer placeholder)
    is misclassified as constant data and tagged into the delegate. A TensorRT
    engine is stateless across executions, so an absorbed mutable buffer would be
    a frozen constant (the KV-cache update would be lost). Strip the delegation
    tag from any buffer that is a mutation target so it stays a caller-owned
    mutable buffer owned above the delegate.
    """
    sig = exported_program.graph_signature
    mutated_buffer_targets = set(sig.buffers_to_mutate.values())
    for node in exported_program.graph_module.graph.nodes:
        if (
            node.op == "placeholder"
            and sig.inputs_to_buffers.get(node.name) in mutated_buffer_targets
        ):
            node.meta.pop("delegation_tag", None)


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
        # The zero-copy KV spec is stamped per-partition in partition(), never
        # applied to every partition like the rest of compile_specs. Its presence
        # here only records that this method asked for zero-copy; the actual
        # elided binding names are derived per engine at partition time, so a
        # method that lowers to several TensorRT delegates marks only the one
        # whose aliased outputs were elided. It has to stay out of the shared
        # list: a plain-compute delegate carrying the spec would make the
        # un-staging cross-check demand an aliased buffer it never had.
        self._zero_copy_requested = any(
            s.key == ZERO_COPY_KV_COMPILE_SPEC_KEY for s in self.compile_specs
        )
        self._base_compile_specs = [
            s for s in self.compile_specs if s.key != ZERO_COPY_KV_COMPILE_SPEC_KEY
        ]
        # Mirror CudaPartitioner: a target_device CompileSpec drives ExecuTorch's
        # PropagateDevicePass, which tags delegate I/O TensorSpecs with the device
        # and serializes it into the .pte's extra_tensor_info. When the caller pins
        # it we use that verbatim; otherwise each partition's device is derived from
        # its own engine node in partition() (engine nodes are not available here)
        # so a cuda:N engine is not mislabeled cuda:0.
        self._has_explicit_target_device = any(
            s.key == _TARGET_DEVICE_COMPILE_SPEC_KEY for s in self._base_compile_specs
        )
        # ExecuTorch partitioners conventionally hold a delegation_spec. partition()
        # builds a fresh DelegationSpec per partition and never reads this one; the
        # only reader is _export._declared_method_name, on a partitioner the caller
        # passes to export().
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
            # Only DEVICE_IDX is read, never the engine itself.
            engine_info = _get_engine_info_for_node(
                exported_program, engine_nodes[0], metadata_only=True
            )
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

    def _partition_elided_output_names(
        self, exported_program: ExportedProgram, partition: Partition
    ) -> Set[str]:
        """Engine output binding names this partition's delegate legitimately drops.

        Zero-copy KV elides an engine's aliased output when its aliased input is a
        buffer export rewired to be written in place -- marked on the placeholder
        with ``_torch_tensorrt_aliased_buffer``. This is derived from THIS
        partition's own engine (its ``aliased_io`` paired with the marks on its own
        input placeholders), never from a method-wide name list, so a second engine
        that happens to share an output binding name is never told it may drop that
        binding. That is what lets a real lost output on the plain delegate still
        raise while the KV delegate's genuine elision is exempted.

        Any extraction failure returns an empty set: the delegate then carries every
        binding and a genuinely missing aliased output stays an error in the
        backend's ``_validate_output_binding_order``.
        """
        from torch_tensorrt.executorch._zero_copy import _aliased_inputs_by_output_index

        try:
            engine_nodes = _get_engine_nodes_in(partition.nodes)
            if len(engine_nodes) != 1:
                return set()
            engine = engine_nodes[0]
            aliased = _aliased_inputs_by_output_index(exported_program, engine)
            if not aliased:
                return set()
            # Only OUTPUT_BINDING_NAMES_IDX is read, never the engine itself.
            engine_info = _get_engine_info_for_node(
                exported_program, engine, metadata_only=True
            )
            raw = engine_info[OUTPUT_BINDING_NAMES_IDX]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", "replace")
            output_names = deserialize_binding_names(str(raw or ""))
            elided: Set[str] = set()
            for output_index, input_node in aliased.items():
                if (
                    isinstance(input_node, torch.fx.Node)
                    and input_node.meta.get("_torch_tensorrt_aliased_buffer")
                    and 0 <= output_index < len(output_names)
                ):
                    elided.add(output_names[output_index])
            return elided
        except Exception as e:
            # Broad by design, mirroring _resolve_target_device_for_partition: any
            # extraction failure must not abort the export. It degrades safely --
            # the delegate keeps every binding, so a truly-elided output is caught
            # downstream rather than dropped.
            logger.warning(
                "zero-copy KV: could not resolve elided outputs for partition %s "
                "(%s); the delegate will carry every binding.",
                getattr(partition, "id", "?"),
                e,
            )
            return set()

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
            specs = list(self._base_compile_specs)
            if not self._has_explicit_target_device:
                specs.append(
                    CompileSpec(
                        _TARGET_DEVICE_COMPILE_SPEC_KEY,
                        self._resolve_target_device_for_partition(
                            exported_program, partition
                        ),
                    )
                )
            if self._zero_copy_requested:
                elided = self._partition_elided_output_names(
                    exported_program, partition
                )
                if elided:
                    specs.append(
                        CompileSpec(
                            ZERO_COPY_KV_COMPILE_SPEC_KEY,
                            _serialize_elided_output_names(elided),
                        )
                    )
            partition_tags[tag] = DelegationSpec(
                backend_id=TensorRTBackend.__name__,
                compile_specs=specs,
            )

        tag_constant_data(exported_program)
        _keep_mutated_buffers_above_delegate(exported_program)

        return PartitionResult(
            tagged_exported_program=exported_program,
            partition_tags=partition_tags,
        )

    def ops_to_not_decompose(
        self, ep: ExportedProgram
    ) -> Tuple[List[torch._ops.OpOverload], Optional[Callable[[torch.fx.Node], bool]]]:
        return ([], None)
