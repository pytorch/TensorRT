from typing import Sequence
import torch
import traceback
from functools import partial
import torch._dynamo as td

from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering._decompositions import get_decompositions
from torch_tensorrt.dynamo.lowering._partition import partition, get_submod_inputs
from torch_tensorrt.dynamo.conversion import convert_module

from torch._dynamo.backends.common import fake_tensor_unsupported

from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_compiler


@td.register_backend(name="tensorrt")
@fake_tensor_unsupported
def tensorrt_backend(
    gm: torch.nn.Module,
    sample_inputs: Sequence[torch.Tensor],
    settings: CompilationSettings = CompilationSettings(),
):
    custom_backend = partial(
        fx_dynamo_backend,
        settings=settings,
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
    example_inputs: Sequence[torch.Tensor],
    settings: CompilationSettings = CompilationSettings(),
):
    """Helper function to manage translation of FX module to TRT engines

    Args:
        module: FX GraphModule to convert
        inputs: Inputs to the module
        settings: Compilation settings
    Returns:
        Compiled FX GraphModule
    """
    try:
        trt_compiled = compile_module(
            gm,
            example_inputs,
            settings=settings,
        )
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
    example_inputs: Sequence[torch.Tensor],
    settings: CompilationSettings = CompilationSettings(),
) -> torch.fx.GraphModule:
    """Compile an FX module

    Includes: Partitioning + Conversion Phases

    Args:
        module: FX GraphModule to convert
        inputs: Inputs to the module
        settings: Compilation settings
    Returns:
        Compiled FX GraphModule
    """
    # Partition module into components that can be TRT-accelerated
    partitioned_module = partition(
        gm, verbose=settings.debug, max_num_trt_engines=settings.max_num_trt_engines
    )

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
            debug=settings.debug,
            workspace_size=settings.workspace_size,
            precision=settings.precision,
        )

        # Replace FX Module with TRT Module
        setattr(partitioned_module, name, trt_mod)

    return partitioned_module
