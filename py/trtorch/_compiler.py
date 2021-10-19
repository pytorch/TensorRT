from typing import List, Dict, Any
import torch
from torch import nn

import trtorch._C
from trtorch._types import EngineCapability
from trtorch._compile_spec import _parse_compile_spec, _parse_device
from trtorch._version import __version__
from trtorch.Device import Device
from types import FunctionType


def compile(module: torch.jit.ScriptModule,
            inputs=[],
            device=Device._current_device(),
            disable_tf32=False,
            sparse_weights=False,
            enabled_precisions=set(),
            refit=False,
            debug=False,
            strict_types=False,
            capability=EngineCapability.default,
            num_min_timing_iters=2,
            num_avg_timing_iters=1,
            workspace_size=0,
            max_batch_size=0,
            calibrator=None,
            truncate_long_and_double=False,
            require_full_compilation=False,
            min_block_size=3,
            torch_executed_ops=[],
            torch_executed_modules=[]) -> torch.jit.ScriptModule:
    """Compile a TorchScript module for NVIDIA GPUs using TensorRT

    Takes a existing TorchScript module and a set of settings to configure the compiler
    and will convert methods to JIT Graphs which call equivalent TensorRT engines

    Converts specifically the forward method of a TorchScript Module

    Arguments:
        module (torch.jit.ScriptModule): Source module, a result of tracing or scripting a PyTorch
            ``torch.nn.Module``

    Keyword Arguments:
        inputs (List[Union(trtorch.Input, torch.Tensor)]): **Required** List of specifications of input shape, dtype and memory layout for inputs to the module. This argument is required. Input Sizes can be specified as torch sizes, tuples or lists. dtypes can be specified using
            torch datatypes or trtorch datatypes and you can use either torch devices or the trtorch device type enum
            to select device type. ::

                input=[
                    trtorch.Input((1, 3, 224, 224)), # Static NCHW input shape for input #1
                    trtorch.Input(
                        min_shape=(1, 224, 224, 3),
                        opt_shape=(1, 512, 512, 3),
                        max_shape=(1, 1024, 1024, 3),
                        dtype=torch.int32
                        format=torch.channel_last
                    ), # Dynamic input shape for input #2
                    torch.randn((1, 3, 224, 244)) # Use an example tensor and let trtorch infer settings
                ]

        device (Union(trtorch.Device, torch.device, dict)): Target device for TensorRT engines to run on ::

            device=trtorch.Device("dla:1", allow_gpu_fallback=True)

        disable_tf32 (bool): Force FP32 layers to use traditional as FP32 format vs the default behavior of rounding the inputs to 10-bit mantissas before multiplying, but accumulates the sum using 23-bit mantissas
        sparse_weights (bool): Enable sparsity for convolution and fully connected layers.
        enabled_precision (Set(Union(torch.dtype, trtorch.dtype))): The set of datatypes that TensorRT can use when selecting kernels
        refit (bool): Enable refitting
        debug (bool): Enable debuggable engine
        strict_types (bool): Kernels should strictly run in a particular operating precision. Enabled precision should only have one type in the set
        capability (trtorch.EngineCapability): Restrict kernel selection to safe gpu kernels or safe dla kernels
        num_min_timing_iters (int): Number of minimization timing iterations used to select kernels
        num_avg_timing_iters (int): Number of averaging timing iterations used to select kernels
        workspace_size (int): Maximum size of workspace given to TensorRT
        max_batch_size (int): Maximum batch size (must be >= 1 to be set, 0 means not set)
        truncate_long_and_double (bool): Truncate weights provided in int64 or double (float64) to int32 and float32
        calibrator (Union(trtorch._C.IInt8Calibrator, tensorrt.IInt8Calibrator)): Calibrator object which will provide data to the PTQ system for INT8 Calibration
        require_full_compilation (bool): Require modules to be compiled end to end or return an error as opposed to returning a hybrid graph where operations that cannot be run in TensorRT are run in PyTorch
        min_block_size (int): The minimum number of contiguous TensorRT convertable operations in order to run a set of operations in TensorRT
        torch_executed_ops (List[str]): List of aten operators that must be run in PyTorch. An error will be thrown if this list is not empty but ``require_full_compilation`` is True
        torch_executed_modules (List[str]): List of modules that must be run in PyTorch. An error will be thrown if this list is not empty but ``require_full_compilation`` is True

    Returns:
        torch.jit.ScriptModule: Compiled TorchScript Module, when run it will execute via TensorRT
    """

    if isinstance(module, torch.jit.ScriptFunction):
        raise TypeError(
            "torch.jit.ScriptFunction currently is not directly supported, wrap the function in a module to compile")

    if require_full_compilation and (len(torch_executed_modules) > 0 or len(torch_executed_ops) > 0):
        raise ValueError(
            "require_full_compilation is enabled however the list of modules and ops to run in torch is not empty. Found: torch_executed_ops: "
            + torch_executed_ops + ", torch_executed_modules: " + torch_executed_modules)

    spec = {
        "inputs": inputs,
        "device": device,
        "disable_tf32":
            disable_tf32,  # Force FP32 layers to use traditional as FP32 format vs the default behavior of rounding the inputs to 10-bit mantissas before multiplying, but accumulates the sum using 23-bit mantissas
        "sparse_weights": sparse_weights,  #Enable sparsity for convolution and fully connected layers.
        "enabled_precisions": enabled_precisions,  # Enabling FP16 kernels
        "refit": refit,  # enable refit
        "debug": debug,  # enable debuggable engine
        "strict_types": strict_types,  # kernels should strictly run in operating precision
        "capability": capability,  # Restrict kernel selection to safe gpu kernels or safe dla kernels
        "num_min_timing_iters": num_min_timing_iters,  # Number of minimization timing iterations used to select kernels
        "num_avg_timing_iters": num_avg_timing_iters,  # Number of averaging timing iterations used to select kernels
        "workspace_size": workspace_size,  # Maximum size of workspace given to TensorRT
        "max_batch_size": max_batch_size,  # Maximum batch size (must be >= 1 to be set, 0 means not set)
        "calibrator": calibrator,
        "truncate_long_and_double": truncate_long_and_double,
        "torch_fallback": {
            "enabled": not require_full_compilation,
            "force_fallback_ops": torch_executed_ops,
            "force_fallback_modules": torch_executed_modules
        }
    }

    compiled_cpp_mod = trtorch._C.compile_graph(module._c, _parse_compile_spec(spec))
    compiled_module = torch.jit._recursive.wrap_cpp_module(compiled_cpp_mod)
    return compiled_module


def convert_method_to_trt_engine(module: torch.jit.ScriptModule,
                                 method_name: str,
                                 inputs=[],
                                 device=Device._current_device(),
                                 disable_tf32=False,
                                 sparse_weights=False,
                                 enabled_precisions=set(),
                                 refit=False,
                                 debug=False,
                                 strict_types=False,
                                 capability=EngineCapability.default,
                                 num_min_timing_iters=2,
                                 num_avg_timing_iters=1,
                                 workspace_size=0,
                                 max_batch_size=0,
                                 truncate_long_and_double=False,
                                 calibrator=None) -> str:
    """Convert a TorchScript module method to a serialized TensorRT engine

    Converts a specified method of a module to a serialized TensorRT engine given a dictionary of conversion settings

    Arguments:
        module (torch.jit.ScriptModule): Source module, a result of tracing or scripting a PyTorch
            ``torch.nn.Module``
        method_name (str): Name of method to convert

    Keyword Args:
        inputs (List[Union(trtorch.Input, torch.Tensor)]): **Required** List of specifications of input shape, dtype and memory layout for inputs to the module. This argument is required. Input Sizes can be specified as torch sizes, tuples or lists. dtypes can be specified using
            torch datatypes or trtorch datatypes and you can use either torch devices or the trtorch device type enum
            to select device type. ::

                input=[
                    trtorch.Input((1, 3, 224, 224)), # Static NCHW input shape for input #1
                    trtorch.Input(
                        min_shape=(1, 224, 224, 3),
                        opt_shape=(1, 512, 512, 3),
                        max_shape=(1, 1024, 1024, 3),
                        dtype=torch.int32
                        format=torch.channel_last
                    ), # Dynamic input shape for input #2
                    torch.randn((1, 3, 224, 244)) # Use an example tensor and let trtorch infer settings
                ]

        device (Union(trtorch.Device, torch.device, dict)): Target device for TensorRT engines to run on ::

            device=trtorch.Device("dla:1", allow_gpu_fallback=True)

        disable_tf32 (bool): Force FP32 layers to use traditional as FP32 format vs the default behavior of rounding the inputs to 10-bit mantissas before multiplying, but accumulates the sum using 23-bit mantissas
        sparse_weights (bool): Enable sparsity for convolution and fully connected layers.
        enabled_precision (Set(Union(torch.dtype, trtorch.dtype))): The set of datatypes that TensorRT can use when selecting kernels
        refit (bool): Enable refitting
        debug (bool): Enable debuggable engine
        strict_types (bool): Kernels should strictly run in a particular operating precision. Enabled precision should only have one type in the set
        capability (trtorch.EngineCapability): Restrict kernel selection to safe gpu kernels or safe dla kernels
        num_min_timing_iters (int): Number of minimization timing iterations used to select kernels
        num_avg_timing_iters (int): Number of averaging timing iterations used to select kernels
        workspace_size (int): Maximum size of workspace given to TensorRT
        max_batch_size (int): Maximum batch size (must be >= 1 to be set, 0 means not set)
        truncate_long_and_double (bool): Truncate weights provided in int64 or double (float64) to int32 and float32
        calibrator (Union(trtorch._C.IInt8Calibrator, tensorrt.IInt8Calibrator)): Calibrator object which will provide data to the PTQ system for INT8 Calibration

    Returns:
        bytes: Serialized TensorRT engine, can either be saved to a file or deserialized via TensorRT APIs
    """
    if isinstance(module, torch.jit.ScriptFunction):
        raise TypeError(
            "torch.jit.ScriptFunctions currently are not directly supported, wrap the function in a module to compile")

    compile_spec = {
        "inputs": inputs,
        "device": device,
        "disable_tf32":
            disable_tf32,  # Force FP32 layers to use traditional as FP32 format vs the default behavior of rounding the inputs to 10-bit mantissas before multiplying, but accumulates the sum using 23-bit mantissas
        "sparse_weights": sparse_weights,  #Enable sparsity for convolution and fully connected layers.
        "enabled_precisions": enabled_precisions,  # Enabling FP16 kernels
        "refit": refit,  # enable refit
        "debug": debug,  # enable debuggable engine
        "strict_types": strict_types,  # kernels should strictly run in operating precision
        "capability": capability,  # Restrict kernel selection to safe gpu kernels or safe dla kernels
        "num_min_timing_iters": num_min_timing_iters,  # Number of minimization timing iterations used to select kernels
        "num_avg_timing_iters": num_avg_timing_iters,  # Number of averaging timing iterations used to select kernels
        "workspace_size": workspace_size,  # Maximum size of workspace given to TensorRT
        "max_batch_size": max_batch_size,  # Maximum batch size (must be >= 1 to be set, 0 means not set)
        "calibrator": calibrator,
        "truncate_long_and_double": truncate_long_and_double
    }

    return trtorch._C.convert_graph_to_trt_engine(module._c, method_name, _parse_compile_spec(compile_spec))


def embed_engine_in_new_module(serialized_engine: bytes, device=Device._current_device()) -> torch.jit.ScriptModule:
    """Takes a pre-built serialized TensorRT engine and embeds it within a TorchScript module

    Takes a pre-built serialied TensorRT engine (as bytes) and embeds it within a TorchScript module.
    Registers the forward method to execute the TensorRT engine with the function signature:

        forward(Tensor[]) -> Tensor[]

    Module can be save with engine embedded with torch.jit.save and moved / loaded according to TRTorch portability rules

    Arguments:
        serialized_engine (bytes): Serialized TensorRT engine from either TRTorch or TensorRT APIs

    Keyword Arguments:
        device (Union(trtorch.Device, torch.device, dict)): Target device to run engine on. Must be compatible with engine provided. Default: Current active device

    Returns:
        torch.jit.ScriptModule: New TorchScript module with engine embedded
    """
    cpp_mod = trtorch._C.embed_engine_in_new_module(serialized_engine, _parse_device(device))
    return torch.jit._recursive.wrap_cpp_module(cpp_mod)


def check_method_op_support(module: torch.jit.ScriptModule, method_name: str) -> bool:
    """Checks to see if a method is fully supported by TRTorch

    Checks if a method of a TorchScript module can be compiled by TRTorch, if not, a list of operators
    that are not supported are printed out and the function returns false, else true.

    Arguments:
        module (torch.jit.ScriptModule): Source module, a result of tracing or scripting a PyTorch
            ``torch.nn.Module``
        method_name (str): Name of method to check

    Returns:
        bool: True if supported Method
    """
    return trtorch._C.check_method_op_support(module._c, method_name)


def dump_build_info():
    """Prints build information about the TRTorch distribution to stdout
    """
    print(get_build_info())


def get_build_info() -> str:
    """Returns a string containing the build information of TRTorch distribution

    Returns:
        str: String containing the build information for TRTorch distribution
    """
    build_info = trtorch._C.get_build_info()
    build_info = "TRTorch Version: " + str(__version__) + '\n' + build_info
    return build_info


def set_device(gpu_id):
    trtorch._C.set_device(gpu_id)
