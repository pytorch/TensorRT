

# @manual=//deeplearning/trt/python:py_tensorrt
import tensorrt as trt
import torch

from torch_tensorrt.fx.tracer.acc_tracer import acc_tracer
from torch_tensorrt.fx import InputTensorSpec
from torch_tensorrt.fx import TRTInterpreter
from torch_tensorrt.fx.passes.lower_basic_pass import transform_setitem
from torch_tensorrt.fx.tools.trt_splitter import TRTSplitter
from torch_tensorrt.fx.tools.trt_splitter import TRTSplitterSetting
from torch_tensorrt.fx.trt_module import TRTModule
from torch_tensorrt.fx.utils import LowerPrecision
 

def compile(module: Any, ir="default", inputs=[], enabled_precisions=set([_enums.dtype.float]), **kwargs):
    """Compile a PyTorch module through fx

    Takes a existing PyTorch module and a set of settings to configure the compiler
    and using the path specified in ``ir`` lower and compile the module to TensorRT
    returning a PyTorch Module back

    Converts specifically the forward method of a Module

    Arguments:
        module (torch.nn.Module): Source module

    Keyword Arguments:
        inputs (List[torch.Tensor]): for fixed shape scenario, inputs shapes can not change
        enabled_precision (torch.dtype): The datatype that TensorRT can use when selecting kernels. If torch.float is chosen, the kernel is running with fp32; If torch.float16 is chosen, the kernel is running with fp16 or fp32 which selected by TensorRT
        ir (str): The requested strategy to compile. (default is ts - TorchScript with scripting path, fx is FX based path)
        **kwargs: Additional settings for the specific requested strategy (See submodules for more info)

    Returns:
        torch.nn.Module: Compiled Module, when run it will execute via TensorRT
    """
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
            # from torch_tensorrt.fx.tools.trt_profiler_sorted import profile_trt_module
            # profile_trt_module("", trt_mod, acc_inputs)
            trt_mod = TRTModule(*r)

            setattr(split_mod, name, trt_mod)
        else:
            submod = getattr(split_mod, name)
    return split_mod
