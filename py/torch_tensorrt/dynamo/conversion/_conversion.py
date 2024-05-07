from __future__ import annotations

import io
import logging
from typing import List, Sequence

import tensorrt as trt
import torch
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
from torch_tensorrt._Device import Device
from torch_tensorrt._enums import dtype
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._TRTInterpreter import (
    TRTInterpreter,
    TRTInterpreterResult,
)
from torch_tensorrt.dynamo.runtime import PythonTorchTensorRTModule, TorchTensorRTModule
from torch_tensorrt.dynamo.utils import get_torch_inputs

logger = logging.getLogger(__name__)


def infer_module_output_dtypes(
    module: torch.fx.GraphModule,
    inputs: Sequence[Input],
    device: Device,
    truncate_double: bool = False,
) -> List[dtype]:
    with maybe_disable_fake_tensor_mode():
        torch_inputs = get_torch_inputs(inputs, device)
        module = module.to(device.to(torch.device))
        module_outputs = module(*torch_inputs)
        if not isinstance(module_outputs, (list, tuple)):
            module_outputs = [module_outputs]

    # Int64 outputs can sometimes be generated from within other operators
    # such as aten.sum - such outputs can be truncated
    output_dtypes = []
    for output in module_outputs:
        if not isinstance(output, torch.Tensor):
            if isinstance(output, str):
                raise ValueError(
                    f"Receieved an output type {type(output)} that's not in the acceptable datatypes (https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)"
                )
            else:
                output_ = torch.tensor(output)
        else:
            output_ = output

        if truncate_double and output_.dtype == dtype.float64:
            output_dtypes.append(dtype.float32)
        else:
            output_dtypes.append(dtype._from(output_.dtype))

    return output_dtypes


def interpret_module_to_result(
    module: torch.fx.GraphModule,
    inputs: Sequence[Input],
    settings: CompilationSettings = CompilationSettings(),
) -> TRTInterpreterResult:
    """Interpret an FX module to a TRTInterpreterResult
    Args:
        module: FX GraphModule to interpret
        inputs: Sequence of Tensors representing inputs to the module
        settings: Compilation settings
    Returns:
        TRTInterpreterResult
    """
    output_dtypes = infer_module_output_dtypes(
        module,
        inputs,
        settings.device,
        truncate_double=settings.truncate_double,
    )

    interpreter = TRTInterpreter(
        module,
        inputs,
        logger_level=(trt.Logger.VERBOSE if settings.debug else trt.Logger.WARNING),
        output_dtypes=output_dtypes,
        compilation_settings=settings,
    )
    interpreter_result = interpreter.run()
    return interpreter_result


def convert_module(
    module: torch.fx.GraphModule,
    inputs: Sequence[Input],
    settings: CompilationSettings = CompilationSettings(),
    name: str = "",
) -> PythonTorchTensorRTModule | TorchTensorRTModule:
    """Convert an FX module to a TRT module
    Args:
        module: FX GraphModule to convert
        inputs: Sequence of Tensors representing inputs to the module
        settings: Compilation settings
        name: TRT engine name
    Returns:
        _PythonTorchTensorRTModule or TorchTensorRTModule
    """
    interpreter_result = interpret_module_to_result(module, inputs, settings)

    if settings.use_python_runtime or not ENABLED_FEATURES.torch_tensorrt_runtime:
        if not settings.use_python_runtime:
            logger.info(
                "Since Torch-TensorRT runtime is not available, using Python Runtime, some features may not be available"
            )
        return PythonTorchTensorRTModule(
            engine=interpreter_result.engine,
            input_names=list(interpreter_result.input_names),
            output_names=list(interpreter_result.output_names),
            target_device=settings.device,
            profiling_enabled=settings.debug,
        )

    else:
        from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

        with io.BytesIO() as engine_bytes:
            engine_bytes.write(interpreter_result.engine)
            engine_str = engine_bytes.getvalue()

        return TorchTensorRTModule(
            serialized_engine=engine_str,
            name=name,
            input_binding_names=list(interpreter_result.input_names),
            output_binding_names=list(interpreter_result.output_names),
            target_device=settings.device,
            hardware_compatible=settings.hardware_compatible,
        )
