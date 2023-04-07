import torch
import logging
import traceback
from functools import partial
import torch._dynamo as td
from torch_tensorrt import EngineCapability, Device
from torch_tensorrt.dynamo import compile

from torch_tensorrt.dynamo.lowering._decompositions import get_decompositions
from torch_tensorrt.dynamo.lowering._partition import partition, get_submod_inputs
from torch_tensorrt.dynamo.conversion import convert_module

from torch._dynamo.backends.common import fake_tensor_unsupported

from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_compiler

from torch_tensorrt.fx.fx2trt import (
    InputTensorSpec,
    TRTInterpreter,
)
import tensorrt as trt

from torch_tensorrt.fx.trt_module import TRTModule
from torch_tensorrt.fx.utils import LowerPrecision

logger = logging.getLogger(__name__)


def create_backend(
    input_signature=None,
    device=Device._current_device(),
    disable_tf32=False,
    sparse_weights=False,
    enabled_precisions=set(),
    refit=False,
    debug=False,
    capability=EngineCapability.default,
    num_avg_timing_iters=1,
    workspace_size=20 << 30,
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
    logger.warn(
        "The Dynamo backend is an experimental feature, for which the "
        + "following arguments are unsupported: "
        + "{input_signature, disable_tf32, sparse_weights, refit, capability, "
        + "num_avg_timing_iters, dla_sram_size, dla_local_dram_size, "
        + "dla_global_dram_size, calibrator, truncate_long_and_double, "
        + "require_full_compilation, min_block_size, torch_executed_ops, "
        + "torch_executed_modules}"
    )

    return partial(
        tensorrt_backend,
        debug=debug,
        enabled_precisions=enabled_precisions,
        device=device,
        workspace_size=workspace_size,
    )


@td.register_backend(name="tensorrt")
@fake_tensor_unsupported
def tensorrt_backend(
    gm: torch.Module,
    sample_inputs,
    *,
    debug=False,
    enabled_precisions=set(),
    device=Device._current_device(),
    workspace_size=20 << 30,
):

    custom_backend = partial(
        fx_dynamo_backend,
        debug=debug,
        enabled_precisions=enabled_precisions,
        device=device,
        workspace_size=workspace_size,
    )

    # Invoke AOTAutograd to translate operators to aten
    return aot_module_simplified(
        gm,
        sample_inputs,
        fw_compiler=make_boxed_compiler(custom_backend),
        decompositions=get_decompositions(),
    )


@td.register_backend(name="fx_tensorrt")
@fake_tensor_unsupported
def fx_dynamo_backend(
    gm: torch.fx.GraphModule,
    example_inputs,
    *,
    debug=False,
    enabled_precisions=set(),
    device=Device._current_device(),
    workspace_size=20 << 30,
):
    """Helper function to manage translation of FX module to TRT engines"""
    try:
        trt_compiled = compile_module(gm, example_inputs)
        return trt_compiled
    except:
        traceback.print_exc()
        print(
            "FX2TRT conversion failed on the subgraph. See trace above. "
            + "Returning GraphModule forward instead."
        )
        return gm.forward


def compile_module(
    gm: torch.fx.GraphModule,
    example_inputs,
    debug: bool = False,
    workspace_size: int = 20 << 30,
    precision: LowerPrecision = LowerPrecision.FP32,
) -> torch.fx.GraphModule:
    """Convert an FX module to a TRT module
    Args:
        module: FX GraphModule to convert
        inputs: Inputs to the module
        debug: Whether to print out verbose debugging information
        workspace_size: Maximum workspace TRT is allowed to use for the module
        precision: Model Layer precision
    Returns:
        TRTModule or TRTModuleNext
    """
    # Partition module into components that can be TRT-accelerated
    partitioned_module = partition(gm)

    # Iterate over all components that can be accelerated
    # Generate the corresponding TRT Module for those
    for name, _ in partitioned_module.named_children():
        submodule = getattr(partitioned_module, name)

        # Get submodule inputs
        submodule_inputs = get_submod_inputs(
            partitioned_module, submodule, example_inputs
        )

        # Create TRT Module from submodule
        trt_mod = convert_module(
            submodule,
            submodule_inputs,
            debug=debug,
            workspace_size=workspace_size,
            precision=precision,
        )

        # Replace FX Module with TRT Module
        setattr(partitioned_module, name, trt_mod)

    return partitioned_module
