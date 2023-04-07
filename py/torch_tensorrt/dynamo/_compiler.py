import torch
import logging

from torch_tensorrt import EngineCapability, Device

from torch_tensorrt.dynamo.lowering._partition import partition
from torch_tensorrt.dynamo import create_backend

from torch_tensorrt.fx.fx2trt import (
    InputTensorSpec,
    TRTInterpreter,
)
import tensorrt as trt

from torch_tensorrt.fx.trt_module import TRTModule
from torch_tensorrt.fx.utils import LowerPrecision

logger = logging.getLogger(__name__)


def compile(
    gm: torch.Module,
    example_inputs,
    *,
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
):
    custom_backend = create_backend(
        device=device,
        disable_tf32=disable_tf32,
        sparse_weights=sparse_weights,
        enabled_precisions=enabled_precisions,
        refit=refit,
        debug=debug,
        capability=capability,
        num_avg_timing_iters=num_avg_timing_iters,
        workspace_size=workspace_size,
        dla_sram_size=dla_sram_size,
        dla_local_dram_size=dla_local_dram_size,
        dla_global_dram_size=dla_global_dram_size,
        calibrator=calibrator,
        truncate_long_and_double=truncate_long_and_double,
        require_full_compilation=require_full_compilation,
        min_block_size=min_block_size,
        torch_executed_ops=torch_executed_ops,
        torch_executed_modules=torch_executed_modules,
    )

    model = torch.compile(gm, backend=custom_backend)
    # Ensure compilation
    model(example_inputs)

    return model


def compile_logic(gm: torch.fx.GraphModule, example_inputs):
    partitioned = partition(gm)

    precision = LowerPrecision.FP32

    def get_submod_inputs(mod, submod, inputs):
        """Helper function to get inputs to submodule"""
        acc_inputs = None

        def get_input(self, inputs):
            nonlocal acc_inputs
            acc_inputs = inputs

        handle = submod.register_forward_pre_hook(get_input)
        mod(*inputs)
        handle.remove()
        return acc_inputs

    for name, _ in partitioned.named_children():
        submod = getattr(partitioned, name)

        # Get submodule inputs
        acc_inputs = get_submod_inputs(partitioned, submod, example_inputs)

        # Create TRT Module from submodule
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

        # Replace FX Module with TRT Module
        setattr(partitioned, name, trt_mod)

    return partitioned
