from typing import List, Dict, Any
from torch_tensorrt import _enums
import torch_tensorrt.ts
from torch_tensorrt import logging
import torch
from torch import fx
from enum import Enum
from torch_tensorrt import fx

class _IRType(Enum):
    """Enum to set the minimum required logging level to print a message to stdout
    """
    ts = 0
    fx = 1


class _ModuleType(Enum):
    """Enum to set the minimum required logging level to print a message to stdout
    """
    nn = 0
    ts = 1
    fx = 2


def _parse_module_type(module: Any) -> _ModuleType:
    if any(isinstance(module, t) for t in [torch.jit.ScriptModule, torch.jit.ScriptFunction]):
        return _ModuleType.ts
    elif isinstance(module, torch.fx.GraphModule):
        return _ModuleType.fx
    elif isinstance(module, torch.nn.Module):
        return _ModuleType.nn
    else:
        raise RuntimeError("Module is an unknown format")


def _get_target_ir(module_type: _ModuleType, ir: str) -> _IRType:
    module_is_tsable = any([module_type == t for t in [_ModuleType.nn, _ModuleType.ts]])
    module_is_fxable = any([module_type == t for t in [_ModuleType.nn, _ModuleType.fx]])

    ir_targets_torchscript = any([ir == opt for opt in ["torchscript", "ts"]])
    ir_targets_fx = ir == "fx"

    if module_is_tsable and ir_targets_torchscript:
        return _IRType.ts
    elif module_is_fxable and ir_targets_fx:
        return _IRType.fx
    else:
        if ir == "default":
            # Options are listed in order of preference
            if module_is_tsable:
                logging.log(logging.Level.Info, "ir was set to default, using TorchScript as ir")
                return _IRType.ts
            elif module_is_fxable:
                raise ValueError("Was given a torch.fx.GraphModule, fx is not currently supported by Torch-TensorRT")
                #logging.log(logging.Level.Info, "ir was set to default, using TorchScript as fx")
                #return _IRType.fx
            else:
                raise ValueError("Module was provided with in an unsupported format")
        else:
            raise ValueError("Unknown ir was requested")


def compile(module: Any, ir="default", inputs=[], enabled_precisions=set([_enums.dtype.float]), **kwargs):
    """Compile a PyTorch module for NVIDIA GPUs using TensorRT

    Takes a existing PyTorch module and a set of settings to configure the compiler
    and using the path specified in ``ir`` lower and compile the module to TensorRT
    returning a PyTorch Module back

    Converts specifically the forward method of a Module

    Arguments:
        module (Union(torch.nn.Module,torch.jit.ScriptModule): Source module

    Keyword Arguments:
        inputs (List[Union(torch_tensorrt.Input, torch.Tensor)]): **Required** List of specifications of input shape, dtype and memory layout for inputs to the module. This argument is required. Input Sizes can be specified as torch sizes, tuples or lists. dtypes can be specified using
            torch datatypes or torch_tensorrt datatypes and you can use either torch devices or the torch_tensorrt device type enum
            to select device type. ::

                input=[
                    torch_tensorrt.Input((1, 3, 224, 224)), # Static NCHW input shape for input #1
                    torch_tensorrt.Input(
                        min_shape=(1, 224, 224, 3),
                        opt_shape=(1, 512, 512, 3),
                        max_shape=(1, 1024, 1024, 3),
                        dtype=torch.int32
                        format=torch.channel_last
                    ), # Dynamic input shape for input #2
                    torch.randn((1, 3, 224, 244)) # Use an example tensor and let torch_tensorrt infer settings
                ]

        enabled_precision (Set(Union(torch.dtype, torch_tensorrt.dtype))): The set of datatypes that TensorRT can use when selecting kernels
        ir (str): The requested strategy to compile. (Options: default - Let Torch-TensorRT decide, ts - TorchScript with scripting path)
        **kwargs: Additional settings for the specific requested strategy (See submodules for more info)

    Returns:
        torch.nn.Module: Compiled Module, when run it will execute via TensorRT
    """
    module_type = _parse_module_type(module)
    target_ir = _get_target_ir(module_type, ir)
    if target_ir == _IRType.ts:
        ts_mod = module
        if module_type == _ModuleType.nn:
            logging.log(
                logging.Level.Info,
                "Module was provided as a torch.nn.Module, trying to script the module with torch.jit.script. In the event of a failure please preconvert your module to TorchScript"
            )
            ts_mod = torch.jit.script(module)
        return torch_tensorrt.ts.compile(ts_mod, inputs=inputs, enabled_precisions=enabled_precisions, **kwargs)
    elif target_ir == _IRType.fx:
        from torch_tensorrt.fx.tracer.acc_tracer import acc_tracer
        from torch_tensorrt.fx import InputTensorSpec
        from torch_tensorrt.fx import TRTInterpreter
        from torch_tensorrt.fx.passes.lower_basic_pass import transform_setitem
        from torch_tensorrt.fx.tools.trt_splitter import TRTSplitter
        from torch_tensorrt.fx.tools.trt_splitter import TRTSplitterSetting
        from torch_tensorrt.fx.trt_module import TRTModule
        from torch_tensorrt.fx.utils import LowerPrecision
        acc_model = acc_tracer.trace(module, inputs)

        splitter_setting = TRTSplitterSetting()
        splitter_setting.use_implicit_batch_dim = False
        splitter = TRTSplitter(acc_model, inputs, settings=splitter_setting)
        splitter.node_support_preview()
        split_mod = splitter()
        num_piece = 0
        for name, _ in split_mod.named_children():
            print(f"graph is split into {name}")
            num_piece += 1

        # if the graph module is split into pieces larger than 8, we consider its perf
        # is not good and fall back to non-TRT
        if num_piece > 8:
            print(
                f"The graph module is split into {num_piece} which is large than the \
                threshold=8. Fall back to non-TRT module."
            )
            return None

        if torch.float16 in enabled_precisions or torch.half in enabled_precisions:
            precision = LowerPrecision.FP16
        else:
            precision = LowerPrecision.FP32

        def get_submod_inputs(mod, submod, inputs):
            acc_inputs = None

            def get_input(self, inputs):
                nonlocal acc_inputs
                acc_inputs = inputs

            handle = submod.register_forward_pre_hook(get_input)
            mod(*inputs)
            handle.remove()
            return acc_inputs

        for name, _ in split_mod.named_children():
            if "_run_on_acc" in name:
                submod = getattr(split_mod, name)
                # Get submodule inputs for fx2trt
                acc_inputs = get_submod_inputs(split_mod, submod, inputs)

                # fx2trt replacement
                interp = TRTInterpreter(
                    submod,
                    InputTensorSpec.from_tensors(acc_inputs),
                    explicit_batch_dimension=True,
                )
                r = interp.run(
                    max_workspace_size=20 << 30,
                    lower_precision=precision,
                    # profiling_verbosity=trt.ProfilingVerbosity.DETAILED, #For profile
                )
                # For profile
                # from fx2trt_oss.fx.tools.trt_profiler_sorted import profile_trt_module
                # profile_trt_module("", trt_mod, acc_inputs)
                trt_mod = TRTModule(*r)

                setattr(split_mod, name, trt_mod)
            else:
                submod = getattr(split_mod, name)
        return split_mod
    else:
        raise RuntimeError("Module is an unknown format or the ir requested is unknown")


def convert_method_to_trt_engine(module: Any,
                                 method_name: str,
                                 ir="default",
                                 inputs=[],
                                 enabled_precisions=set([_enums.dtype.float]),
                                 **kwargs):
    """Convert a TorchScript module method to a serialized TensorRT engine

    Converts a specified method of a module to a serialized TensorRT engine given a dictionary of conversion settings

    Arguments:
        module (Union(torch.nn.Module,torch.jit.ScriptModule): Source module

    Keyword Arguments:
        inputs (List[Union(torch_tensorrt.Input, torch.Tensor)]): **Required** List of specifications of input shape, dtype and memory layout for inputs to the module. This argument is required. Input Sizes can be specified as torch sizes, tuples or lists. dtypes can be specified using
            torch datatypes or torch_tensorrt datatypes and you can use either torch devices or the torch_tensorrt device type enum
            to select device type. ::

                input=[
                    torch_tensorrt.Input((1, 3, 224, 224)), # Static NCHW input shape for input #1
                    torch_tensorrt.Input(
                        min_shape=(1, 224, 224, 3),
                        opt_shape=(1, 512, 512, 3),
                        max_shape=(1, 1024, 1024, 3),
                        dtype=torch.int32
                        format=torch.channel_last
                    ), # Dynamic input shape for input #2
                    torch.randn((1, 3, 224, 244)) # Use an example tensor and let torch_tensorrt infer settings
                ]

        enabled_precision (Set(Union(torch.dtype, torch_tensorrt.dtype))): The set of datatypes that TensorRT can use when selecting kernels
        ir (str): The requested strategy to compile. (Options: default - Let Torch-TensorRT decide, ts - TorchScript with scripting path)
        **kwargs: Additional settings for the specific requested strategy (See submodules for more info)
    Returns:
        bytes: Serialized TensorRT engine, can either be saved to a file or deserialized via TensorRT APIs
    """
    module_type = _parse_module_type(module)
    target_ir = _get_target_ir(module_type, ir)
    if target_ir == _IRType.ts:
        ts_mod = module
        if module_type == _ModuleType.nn:
            logging.log(
                logging.Level.Info,
                "Module was provided as a torch.nn.Module, trying to script the module with torch.jit.script. In the event of a failure please preconvert your module to TorchScript"
            )
            ts_mod = torch.jit.script(module)
        return torch_tensorrt.ts.convert_method_to_trt_engine(ts_mod,
                                                              method_name,
                                                              inputs=inputs,
                                                              enabled_precisions=enabled_precisions,
                                                              **kwargs)
    elif target_ir == _IRType.fx:
        raise RuntimeError("fx is currently not supported")
    else:
        raise RuntimeError("Module is an unknown format or the ir requested is unknown")
