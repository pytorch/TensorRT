# type: ignore[]

import torch
import torch.fx
import torch.nn as nn
from torch_tensorrt.fx.utils import LowerPrecision
import torch_tensorrt.fx.tracer.acc_tracer.acc_tracer as acc_tracer
from torch_tensorrt.fx import InputTensorSpec, TRTInterpreter, TRTModule
from torch_tensorrt.fx.tools.trt_splitter import TRTSplitter

# The purpose of this example is to demonstrate the overall flow of lowering a PyTorch
# model to TensorRT via FX with existing FX based tooling. The general lowering flow
# would be like:
#
# 1. Use splitter to split the model if there're ops in the model that we don't want to
#    lower to TensorRT for some reasons like the ops are not supported in TensorRT or
#    running them on other backends provides better performance.
# 2. Lower the model (or part of the model if splitter is used) to TensorRT via fx2trt.
#
# If we know the model is fully supported by fx2trt then we can skip the splitter.


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = torch.linalg.norm(x, ord=2, dim=1)
        x = self.relu(x)
        return x


inputs = [torch.randn((1, 10), device=torch.device("cuda"))]
model = Model().cuda().eval()

# acc_tracer is a custom fx tracer that maps nodes whose targets are PyTorch operators
# to acc ops.
traced = acc_tracer.trace(model, inputs)

# Splitter will split the model into several submodules. The name of submodules will
# be either `run_on_acc_{}` or `run_on_gpu_{}`. Submodules named `run_on_acc_{}` can
# be fully lowered to TensorRT via fx2trt while submodules named `run_on_gpu_{}` has
# unsupported ops and can't be lowered by fx2trt. We can still run `run_on_gpu_{}`
# submodules on Gpu if ops there have cuda implementation, the naming is a bit
# confusing and we'll improve it.
splitter = TRTSplitter(traced, inputs)

# Preview functionality allows us to see what are the supported ops and unsupported
# ops. We can optionally the dot graph which will color supported ops and unsupported
# ops differently.
splitter.node_support_preview(dump_graph=False)
"""
Supported node types in the model:
acc_ops.linear: ((), {'input': torch.float32, 'weight': torch.float32, 'bias': torch.float32})
acc_ops.relu: ((), {'input': torch.float32})

Unsupported node types in the model:
acc_ops.linalg_norm: ((), {'input': torch.float32})
"""

# Split.
split_mod = splitter()

# After split we have three submodules, _run_on_acc_0 and _run_on_gpu_1.
print(split_mod.graph)
"""
graph():
    %x : [#users=1] = placeholder[target=x]
    %_run_on_acc_0 : [#users=1] = call_module[target=_run_on_acc_0](args = (%x,), kwargs = {})
    %_run_on_gpu_1 : [#users=1] = call_module[target=_run_on_gpu_1](args = (%_run_on_acc_0,), kwargs = {})
    %_run_on_acc_2 : [#users=1] = call_module[target=_run_on_acc_2](args = (%_run_on_gpu_1,), kwargs = {})
    return _run_on_acc_2
"""

# Take a look at what inside each submodule. _run_on_acc_0 contains linear and relu while
# _run_on_gpu_1 contains linalg_norm which currently is not supported by fx2trt. _run_on_acc_3
# is the another submodule supported.
print(split_mod._run_on_acc_0.graph)
print(split_mod._run_on_gpu_1.graph)
print(split_mod._run_on_acc_2.graph)
"""
graph():
    %x : [#users=1] = placeholder[target=x]
    %linear_weight : [#users=1] = get_attr[target=linear.weight]
    %linear_bias : [#users=1] = get_attr[target=linear.bias]
    %linear_1 : [#users=1] = call_function[target=torch_tensorrt.fx.tracer.acc_tracer.acc_ops.linear](args = (), ...
    %relu_1 : [#users=1] = call_function[target=torch_tensorrt.fx.tracer.acc_tracer.acc_ops.relu](args = (), ...
    return relu_1
graph():
    %relu_1 : [#users=1] = placeholder[target=relu_1]
    %linalg_norm_1 : [#users=1] = call_function[target=torch_tensorrt.fx.tracer.acc_tracer.acc_ops.linalg_norm](args = (), ...
    return linalg_norm_1
graph():
    %linalg_norm_1 : [#users=1] = placeholder[target=linalg_norm_1]
    %relu_3 : [#users=1] = call_function[target=torch_tensorrt.fx.tracer.acc_tracer.acc_ops.relu](args = (), kwargs = {input: %linalg_norm_1, inplace: False})
    return relu_3
"""


def get_submod_inputs(mod, submod, inputs):
    acc_inputs = None

    def get_input(self, inputs):
        nonlocal acc_inputs
        acc_inputs = inputs

    handle = submod.register_forward_pre_hook(get_input)
    mod(*inputs)
    handle.remove()
    return acc_inputs


# Since the model is splitted into three segments. We need to lower each TRT eligible segment.
# If we know the model can be fully lowered, we can skip the splitter part.
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
        r = interp.run(lower_precision=LowerPrecision.FP32)
        trt_mod = TRTModule(*r)
        setattr(split_mod, name, trt_mod)

lowered_model_output = split_mod(*inputs)

# Save and load model
torch.save(split_mod, "trt.pt")
reload_trt_mod = torch.load("trt.pt")
reload_model_output = reload_trt_mod(*inputs)

# Make sure the results match
regular_model_output = model(*inputs)
torch.testing.assert_close(
    reload_model_output, regular_model_output, atol=3e-3, rtol=1e-2
)
