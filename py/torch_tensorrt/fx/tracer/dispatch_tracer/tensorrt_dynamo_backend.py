import torch
import traceback
import torch._dynamo as td

from torch_tensorrt.fx.fx2trt import (
    InputTensorSpec,
    TRTInterpreter,
)
import tensorrt as trt
from torch_tensorrt.fx.tools.trt_splitter import (
    TRTSplitter,
    TRTSplitterSetting,
)
from torch_tensorrt.fx.tracer.dispatch_tracer import aten_tracer
from torch_tensorrt.fx.trt_module import TRTModule
from torch_tensorrt.fx.utils import LowerPrecision

from torch._dynamo.backends.common import fake_tensor_unsupported

from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_compiler

from torch._inductor.decomposition import decompositions

DECOMPOSITIONS = decompositions.copy()
MAX_SPLITS_THRESHOLD = 100


def tensorrt_backend(gm, sample_inputs):
    # Invoke AOTAutograd to compile model
    return aot_module_simplified(
        gm,
        sample_inputs,
        fw_compiler=make_boxed_compiler(fx2trt_compiler),
        decompositions=DECOMPOSITIONS,
    )


def fx2trt(gm: torch.fx.GraphModule, example_inputs, **kwargs):
    model = gm
    inputs = example_inputs

    # Perform lowering pass on model
    model = aten_tracer.opt_trace(model, inputs, perform_trace=False)

    # Split out unsupported ops --> Needs rewrite/revision for ATEN
    splitter_setting = TRTSplitterSetting()
    splitter_setting.use_implicit_batch_dim = False
    splitter = TRTSplitter(model, inputs, settings=splitter_setting)

    splitter.node_support_preview()
    split_mod = splitter()
    num_pieces = 0

    for name, _ in split_mod.named_children():
        print(f"Graph is split into {name}")
        num_pieces += 1

    # Select threshold above which segmentation is not beneficial and run graph in Torch
    if num_pieces > MAX_SPLITS_THRESHOLD:
        raise AssertionError(
            f"The graph module is split into {num_pieces} which is large than the \
            threshold={MAX_SPLITS_THRESHOLD}. Falling back to non-TRT module."
        )

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
            acc_inputs = get_submod_inputs(split_mod, submod, inputs)

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

            setattr(split_mod, name, trt_mod)

    return split_mod


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
