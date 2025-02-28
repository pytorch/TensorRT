from dataclasses import dataclass, field

from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.fx.types import TRTNetwork


@dataclass
class ConversionContext:
    """Class representing the context for conversion of a particular network

    Args:
        net: TensorRT Network being built
        compilation_settings: Settings selected by the user for compilation
        requires_output_allocator: Whether the network requires output allocator
    """

    net: TRTNetwork
    compilation_settings: CompilationSettings = field(
        default_factory=CompilationSettings
    )
    requires_output_allocator: bool = False
