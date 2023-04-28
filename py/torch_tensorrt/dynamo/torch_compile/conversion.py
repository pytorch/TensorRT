from typing import Sequence, Union
import torch
from torch_tensorrt.fx.trt_module import TRTModule
from torch_tensorrt import TRTModuleNext
from torch_tensorrt.fx.fx2trt import (
    InputTensorSpec,
    TRTInterpreter,
)
from torch_tensorrt.fx.utils import LowerPrecision

import tensorrt as trt


def convert_module(
    module: torch.fx.GraphModule,
    inputs: Sequence[torch.Tensor],
    debug: bool = False,
    workspace_size: int = 20 << 30,
    precision: LowerPrecision = LowerPrecision.FP32,
    explicit_batch_dim: bool = True,
    explicit_precision: bool = True,
) -> Union[TRTModuleNext, TRTModule]:
    """Convert an FX module to a TRT module
    Args:
        module: FX GraphModule to convert
        inputs: Sequence of Tensors representing inputs to the module
        debug: Whether to print out verbose debugging information
        workspace_size: Maximum workspace TRT is allowed to use for the module
        precision: Model Layer precision
    Returns:
        TRTModule or TRTModuleNext
    """
    interp = TRTInterpreter(
        module,
        InputTensorSpec.from_tensors(inputs),
        explicit_batch_dimension=True,
        logger_level=(trt.Logger.VERBOSE if debug else trt.Logger.WARNING),
    )

    r = interp.run(
        max_workspace_size=workspace_size,
        lower_precision=precision,
        profiling_verbosity=(
            trt.ProfilingVerbosity.VERBOSE
            if debug
            else trt.ProfilingVerbosity.LAYER_NAMES_ONLY
        ),
    )

    return TRTModule(*r)
