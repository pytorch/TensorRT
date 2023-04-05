import torch

from torch_tensorrt import EngineCapability, Device
from torch_tensorrt.dynamo.lowering import lower_module
from torch_tensorrt.dynamo.runtime import RuntimeOption, RUNTIMES


def compile(
    module: torch.fx.GraphModule,
    inputs=[],
    input_signature=None,
    device=Device._current_device(),
    disable_tf32=False,
    sparse_weights=False,
    enabled_precisions=set(),
    refit=False,
    debug=False,
    capability=EngineCapability.default,
    num_avg_timing_iters=1,
    workspace_size=0,
    dla_sram_size=1048576,
    dla_local_dram_size=1073741824,
    dla_global_dram_size=536870912,
    calibrator=None,
    truncate_long_and_double=False,
    require_full_compilation=False,
    min_block_size=3,
    torch_executed_ops=[],
    torch_executed_modules=[],
    runtime: RuntimeOption = RuntimeOption.TorchTensorRTRuntime,
) -> torch.fx.GraphModule:

    partitioned_model = lower_module(module)

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

        trt_mod = RUNTIMES[runtime](*r)

        setattr(partitioned_model, name, trt_mod)

    return partitioned_model
