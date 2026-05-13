from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.types import TRTNetwork


class AliasKind(str, Enum):
    """Origin of an aliased input/output binding pair.

    KV_CACHE_UPDATE: Aliasing is enforced by TensorRT itself via
        ``IKVCacheUpdateLayer``; the engine reports it through
        ``ICudaEngine.get_aliased_input_tensor``. Shape contract is enforced
        by the layer.

    USER: Aliasing is declared by the Torch-TensorRT compile flow. TRT does
        not enforce it; the runtime must validate shape compatibility and
        bind both input and output to the same device pointer.
    """

    KV_CACHE_UPDATE = "kv_cache_update"
    USER = "user"


class AliasedOutput(NamedTuple):
    """One aliased output recorded during conversion."""

    # The TRT ITensor that should be aliased to an input binding.
    output_tensor: object  # tensorrt.ITensor (avoid hard import here)
    # The TRT input binding name the output should share device memory with.
    input_binding_name: str
    kind: AliasKind


@dataclass
class ConversionContext:
    """Class representing the context for conversion of a particular network

    Args:
        net: TensorRT Network being built
        compilation_settings: Settings selected by the user for compilation
        requires_output_allocator: Boolean flag indicating if the converter creates operators which require an Output Allocator to run (e.g. data dependent operators)
        weight_refit_map: Dictionary mapping weight names to their corresponding np.array
        cpu_weights_reference_holder: Dictionary mapping weight names to their corresponding torch.Tensor
    """

    net: TRTNetwork
    compilation_settings: CompilationSettings = field(
        default_factory=CompilationSettings
    )
    requires_output_allocator: bool = False
    requires_native_multidevice: bool = False
    weight_refit_map: dict[str, torch.Tensor] = field(default_factory=dict)
    cpu_weights_reference_holder: list[torch.Tensor] = field(default_factory=list)
    # Aliased outputs registered by converters during conversion.
    # ``TRTInterpreter`` is responsible for ensuring each output_tensor that
    # isn't already a user output is added to the network outputs, and for
    # carrying the alias mapping forward into ``TRTInterpreterResult``.
    aliased_outputs: list[AliasedOutput] = field(default_factory=list)

    def record_weight(self, name: str, weight: torch.Tensor) -> None:
        """
        Record the weight and name for refitting and CPU reference.
        For the refit map, the key is the weight name that appears in the TRT engine and the value is the weight tensor.
        For the CPU reference holder, we need to hold the reference to the weight tensor until the whole compilation process is complete.

        Args:
            name: Name of the weight
            weight: Weight to record
        """
        self.weight_refit_map[name] = weight
        self.cpu_weights_reference_holder.append(weight)

    def clear_cpu_weights_reference_holder(self) -> None:
        self.cpu_weights_reference_holder.clear()
