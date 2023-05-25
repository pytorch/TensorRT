from typing import Sequence, Union
import torch
from torch_tensorrt.fx.trt_module import TRTModule
from torch_tensorrt import TRTModuleNext
from torch_tensorrt.dynamo.backend._settings import CompilationSettings
from torch_tensorrt.fx.fx2trt import (
    InputTensorSpec,
    TRTInterpreter,
)

import tensorrt as trt


def convert_module(
    module: torch.fx.GraphModule,
    inputs: Sequence[torch.Tensor],
    settings: CompilationSettings = CompilationSettings(),
) -> Union[TRTModuleNext, TRTModule]:
    """Convert an FX module to a TRT module
    Args:
        module: FX GraphModule to convert
        inputs: Sequence of Tensors representing inputs to the module
        settings: Compilation settings
    Returns:
        TRTModule or TRTModuleNext
    """
    interp = TRTInterpreter(
        module,
        InputTensorSpec.from_tensors(inputs),
        explicit_batch_dimension=True,
        logger_level=(trt.Logger.VERBOSE if settings.debug else trt.Logger.WARNING),
    )

    r = interp.run(
        max_workspace_size=settings.workspace_size,
        lower_precision=settings.precision,
        profiling_verbosity=(
            trt.ProfilingVerbosity.VERBOSE
            if settings.debug
            else trt.ProfilingVerbosity.LAYER_NAMES_ONLY
        ),
    )

    return TRTModule(*r)
