from __future__ import annotations

import io
import logging
from typing import Any, List, NamedTuple, Optional, Sequence

import torch
from torch_tensorrt._enums import dtype
from torch_tensorrt._features import ENABLED_FEATURES
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo._engine_cache import BaseEngineCache
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._TRTInterpreter import TRTInterpreter
from torch_tensorrt.dynamo.runtime import PythonTorchTensorRTModule, TorchTensorRTModule
from torch_tensorrt.dynamo.utils import (
    get_cpu_memory_usage,
    get_output_dtypes,
    release_memory,
)

logger = logging.getLogger(__name__)


class SerializedInterpreterResult(NamedTuple):
    serialized_engine: bytes
    input_names: Sequence[str]
    output_names: Sequence[str]
    weight_name_map: Optional[dict[Any, Any]]
    requires_output_allocator: bool


def infer_module_output_dtypes(
    module: torch.fx.GraphModule,
    truncate_double: bool = False,
) -> List[dtype]:
    """
    This function get the output dtypes from node.meta['val'] which was set during dynamo compile_module step
    and truncates them accordingly.
    """
    outputs = [node for node in module.graph.nodes if node.op == "output"]
    outputs = outputs[0].args
    return get_output_dtypes(outputs, truncate_double)  # type: ignore


def interpret_module_to_result(
    module: torch.fx.GraphModule,
    inputs: Sequence[Input],
    settings: CompilationSettings = CompilationSettings(),
    arg_inputs: Optional[Sequence[Input]] = None,
    kwarg_inputs: Optional[dict[str, Any]] = None,
    engine_cache: Optional[BaseEngineCache] = None,
) -> SerializedInterpreterResult:
    """Interpret an FX module to a TRTInterpreterResult
    Args:
        module: FX GraphModule to interpret
        inputs: Sequence of FLATTENED Tensors representing inputs to the module. It should include both
                arg_inputs and kwarg_inputs, if applicable.
        arg_inputs: Sequence of Tensors representing inputs to the module.
        kwarg_inputs: A dictionary of Tensors representing inputs to the module.
        settings: Compilation settings
        engine_cache: Engine cache instance
    Returns:
        TRTInterpreterResult
    """
    output_dtypes = infer_module_output_dtypes(
        module, truncate_double=settings.truncate_double
    )

    interpreter = TRTInterpreter(
        module,
        inputs,
        output_dtypes=output_dtypes,
        compilation_settings=settings,
        engine_cache=engine_cache,
    )

    interpreter_result = interpreter.run()
    # Delete the frozen parameters from the module to release CPU memory
    del interpreter
    for attr in dir(module):
        if attr.startswith("_frozen_param"):
            delattr(module, attr)
    release_memory()
    logger.debug(
        f"CPU memory usage after clearing frozen parameters and building memory in conversion: {get_cpu_memory_usage()} MB"
    )

    serialized_engine = interpreter_result.engine.serialize()
    with io.BytesIO() as engine_bytes:
        engine_bytes.write(serialized_engine)
        serialized_engine = engine_bytes.getvalue()
        logger.debug(
            f"CPU memory usage after serializing engine: {get_cpu_memory_usage()} MB"
        )
    serialized_interpreter_result = SerializedInterpreterResult(
        serialized_engine=serialized_engine,
        input_names=interpreter_result.input_names,
        output_names=interpreter_result.output_names,
        weight_name_map=interpreter_result.weight_name_map,
        requires_output_allocator=interpreter_result.requires_output_allocator,
    )

    return serialized_interpreter_result


def convert_module(
    module: torch.fx.GraphModule,
    inputs: Sequence[Input],
    settings: CompilationSettings = CompilationSettings(),
    name: str = "",
    engine_cache: Optional[BaseEngineCache] = None,
) -> PythonTorchTensorRTModule | TorchTensorRTModule:
    """Convert an FX module to a TRT module
    Args:
        module: FX GraphModule to convert
        inputs: Sequence of Tensors representing inputs to the module
        settings: Compilation settings
        name: TRT engine name
        engine_cache: Engine cache instance
    Returns:
        PythonTorchTensorRTModule or TorchTensorRTModule
    """
    interpreter_result = interpret_module_to_result(
        module, inputs, settings, engine_cache=engine_cache
    )

    rt_cls = PythonTorchTensorRTModule

    if ENABLED_FEATURES.torch_tensorrt_runtime and not settings.use_python_runtime:
        from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

        rt_cls = TorchTensorRTModule

    elif (
        not ENABLED_FEATURES.torch_tensorrt_runtime and not settings.use_python_runtime
    ):
        logger.info(
            "Since Torch-TensorRT runtime is not available, using Python Runtime, some features may not be available"
        )

    return rt_cls(
        serialized_engine=interpreter_result.serialized_engine,
        input_binding_names=list(interpreter_result.input_names),
        output_binding_names=list(interpreter_result.output_names),
        name=name,
        settings=settings,
        weight_name_map=interpreter_result.weight_name_map,
        requires_output_allocator=interpreter_result.requires_output_allocator,
    )
