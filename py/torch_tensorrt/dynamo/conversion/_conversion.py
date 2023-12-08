from __future__ import annotations

import io
from typing import Sequence

import tensorrt as trt
import torch
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._TRTInterpreter import TRTInterpreter
from torch_tensorrt.dynamo.runtime import PythonTorchTensorRTModule, TorchTensorRTModule
from torch_tensorrt.dynamo.utils import get_torch_inputs


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
    # Specify module output data types to ensure TRT output types agree with
    # that of the equivalent Torch module
    torch_inputs = get_torch_inputs(inputs, settings.device)
    module_outputs = module(*torch_inputs)

    if not isinstance(module_outputs, (list, tuple)):
        module_outputs = [module_outputs]

    # Int64 outputs can sometimes be generated from within other operators
    # such as aten.sum - such outputs can be truncated
    output_dtypes = []
    for output in module_outputs:
        if settings.truncate_long_and_double and output.dtype == torch.float64:
            output_dtypes.append(torch.float32)
        elif settings.truncate_long_and_double and output.dtype == torch.int64:
            output_dtypes.append(torch.int32)
        else:
            output_dtypes.append(output.dtype)

    interpreter = TRTInterpreter(
        module,
        inputs,
        logger_level=(trt.Logger.VERBOSE if settings.debug else trt.Logger.WARNING),
        output_dtypes=output_dtypes,
        compilation_settings=settings,
    )
    interpreter_result = interpreter.run()

    if settings.use_python_runtime:
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
            engine_bytes.write(interpreter_result.engine.serialize())
            engine_str = engine_bytes.getvalue()
        return TorchTensorRTModule(
            serialized_engine=engine_str,
            name=name,
            input_binding_names=list(interpreter_result.input_names),
            output_binding_names=list(interpreter_result.output_names),
            target_device=settings.device,
        )
