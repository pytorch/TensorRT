from dataclasses import dataclass, field
from typing import Union

import numpy as np
import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.types import TRTNetwork


@dataclass
class ConversionContext:
    """Class representing the context for conversion of a particular network

    Args:
        net: TensorRT Network being built
        compilation_settings: Settings selected by the user for compilation
        requires_output_allocator: Boolean flag indicating if the converter creates operators which require an Output Allocator to run (e.g. data dependent operators)
    """

    net: TRTNetwork
    compilation_settings: CompilationSettings = field(
        default_factory=CompilationSettings
    )
    requires_output_allocator: bool = False
    mapping: dict[str, np.array] = field(default_factory=dict)
    weights_reference_holder: dict[str, Union[torch.Tensor, np.array]] = field(
        default_factory=dict
    )
