import torch
import traceback
from typing import Sequence
import torch._dynamo as td

from torch_tensorrt.fx.fx2trt import (
    InputTensorSpec,
    TRTInterpreter,
)
import tensorrt as trt
from torch_tensorrt.fx.trt_module import TRTModule
from torch_tensorrt.fx.utils import LowerPrecision

from torch import Tensor, SymInt
from torch._dynamo.backends.common import fake_tensor_unsupported
from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_compiler
from torch._decomp import register_decomposition
from torch._decomp import core_aten_decompositions


DECOMPOSITIONS = {**core_aten_decompositions()}
aten = torch._ops.ops.aten


def replace_inplace_op(aten_op, outplace_op):
    """Replace inplace operation with functional equivalent

    Adapted from:
    https://github.com/pytorch/pytorch/blob/3344d79e3f732dadd5c85b99a7aa1a022f187929/torch/_decomp/decompositions.py#L3355-L3361
    """

    @register_decomposition(aten_op, registry=DECOMPOSITIONS)
    def inplace_op(*args, **kwargs):
        out = outplace_op(*args, **kwargs)
        return args[0].copy_(out)

    return inplace_op


replace_inplace_op(aten.add_, aten.add)
replace_inplace_op(aten.addbmm_, aten.addbmm)
replace_inplace_op(aten.addmm_, aten.addmm)
replace_inplace_op(aten.addmv_, aten.addmv)
replace_inplace_op(aten.baddbmm_, aten.baddbmm)
replace_inplace_op(aten.cumprod_, aten.cumprod)
replace_inplace_op(aten.fill_, aten.fill)
replace_inplace_op(aten.gelu_, aten.gelu)
replace_inplace_op(aten.hardsigmoid_, aten.hardsigmoid)
replace_inplace_op(aten.index_put_, aten.index_put)
replace_inplace_op(aten.index_reduce_, aten.index_reduce)
replace_inplace_op(aten.logit_, aten.logit)
replace_inplace_op(aten.relu_, aten.relu)
replace_inplace_op(aten.renorm_, aten.renorm)
replace_inplace_op(aten.round_, aten.round)
replace_inplace_op(aten.scatter_, aten.scatter)
replace_inplace_op(aten.scatter_add_, aten.scatter_add)
replace_inplace_op(aten.scatter_reduce_, aten.scatter_reduce)


@register_decomposition(torch.ops.aten.clone.default, registry=DECOMPOSITIONS)
def clone_removal(self: Tensor) -> Tensor:
    return self


@register_decomposition(torch.ops.aten._unsafe_view.default, registry=DECOMPOSITIONS)
def unsafe_view_replacement(self: Tensor, size: Sequence[SymInt]) -> Tensor:
    return torch.reshape(self, size)


def partition(gm: torch.fx.GraphModule):
    pass


def tensorrt_backend(gm, sample_inputs):
    # Invoke AOTAutograd to compile model
    return aot_module_simplified(
        gm,
        sample_inputs,
        fw_compiler=make_boxed_compiler(fx2trt_compiler),
        decompositions=DECOMPOSITIONS,
    )


def fx2trt(model: torch.fx.GraphModule, inputs, **kwargs):
    partitioned_model = partition(model)

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

    for name, _ in partitioned_model.named_children():
        submod = getattr(partitioned_model, name)
        acc_inputs = get_submod_inputs(partitioned_model, submod, inputs)

        interp = TRTInterpreter(
            submod,
            InputTensorSpec.from_tensors(acc_inputs),
            explicit_batch_dimension=True,
            logger_level=trt.Logger.VERBOSE,
        )
        r = interp.run(
            max_workspace_size=20 << 30,
            lower_precision=precision,
            profiling_verbosity=trt.ProfilingVerbosity.VERBOSE,
        )

        trt_mod = TRTModule(*r)

        setattr(partitioned_model, name, trt_mod)

    return partitioned_model


@td.register_backend
@fake_tensor_unsupported
def fx2trt_compiler(gm: torch.fx.GraphModule, example_inputs):
    try:
        trt_compiled = fx2trt(gm, example_inputs)
        return trt_compiled
    except Exception:
        traceback.print_exc()
        print(
            "FX2TRT conversion failed on the subgraph. See trace above. Returning GraphModule forward instead"
        )
        return gm.forward
