import logging
from typing import Sequence
import torch
from functools import partial
import torch._dynamo as td

from torch_tensorrt.dynamo.backend._settings import CompilationSettings
from torch_tensorrt.dynamo.backend.lowering._decompositions import (
    get_decompositions,
)
from torch_tensorrt.dynamo.backend.lowering._partition import (
    partition,
    get_submod_inputs,
)

from torch_tensorrt.dynamo.backend.utils import repair_long_or_double_input
from torch_tensorrt.dynamo.backend.conversion import convert_module

from torch._dynamo.backends.common import fake_tensor_unsupported

from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_compiler


logger = logging.getLogger(__name__)


@td.register_backend(name="torch_tensorrt")
@fake_tensor_unsupported
def torch_tensorrt_backend(
    gm: torch.fx.GraphModule,
    sample_inputs: Sequence[torch.Tensor],
    settings: CompilationSettings = CompilationSettings(),
):
    DEFAULT_BACKEND = aot_torch_tensorrt_aten_backend

    return DEFAULT_BACKEND(gm, sample_inputs, settings=settings)


@td.register_backend(name="aot_torch_tensorrt_aten")
@fake_tensor_unsupported
def aot_torch_tensorrt_aten_backend(
    gm: torch.fx.GraphModule,
    sample_inputs: Sequence[torch.Tensor],
    settings: CompilationSettings = CompilationSettings(),
):
    custom_backend = partial(
        _pretraced_backend,
        settings=settings,
    )

    # Invoke AOTAutograd to translate operators to aten
    return aot_module_simplified(
        gm,
        sample_inputs,
        fw_compiler=make_boxed_compiler(custom_backend),
        decompositions=get_decompositions(),
    )


@fake_tensor_unsupported
def _pretraced_backend(
    gm: torch.fx.GraphModule,
    sample_inputs: Sequence[torch.Tensor],
    settings: CompilationSettings = CompilationSettings(),
):
    """Helper function to manage translation of traced FX module to TRT engines

    Args:
        module: FX GraphModule to convert
        inputs: Inputs to the module
        settings: Compilation settings
    Returns:
        Compiled FX GraphModule
    """
    try:
        trt_compiled = _compile_module(
            gm,
            sample_inputs,
            settings=settings,
        )
        return trt_compiled
    except:
        logger.error(
            "FX2TRT conversion failed on the subgraph. See trace above. "
            + "Returning GraphModule forward instead.",
            exc_info=True,
        )

        if not settings.pass_through_build_failures:
            return gm.forward
        else:
            raise AssertionError(
                "Halting compilation on build failure since "
                + "pass_through_build_failures was specified as True. "
                + "To return the default Torch implementation and avoid "
                + "halting compilation on engine build failures, "
                + "specify pass_through_build_failures=False."
            )


def _compile_module(
    gm: torch.fx.GraphModule,
    sample_inputs: Sequence[torch.Tensor],
    settings: CompilationSettings = CompilationSettings(),
) -> torch.fx.GraphModule:
    """Compile a traced FX module

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
        gm,
        verbose=settings.debug,
        min_block_size=settings.min_block_size,
        torch_executed_ops=settings.torch_executed_ops,
    )

    # Iterate over all components that can be accelerated
    # Generate the corresponding TRT Module for those
    for name, _ in partitioned_module.named_children():
        submodule = getattr(partitioned_module, name)

        # Get submodule inputs
        submodule_inputs = get_submod_inputs(
            partitioned_module, submodule, sample_inputs
        )

        # Ensure all submodule inputs do not require a gradient
        for param in submodule_inputs:
            param.requires_grad = False

        # Handle long/double inputs if requested by the user
        if settings.truncate_long_and_double:
            num_submodule_inputs = len(submodule_inputs)
            repaired_outputs_once = False

            # For each input to the TRT subgraph, check if its type is long/double
            for position in range(num_submodule_inputs):
                param = submodule_inputs[position]

                # If the data type of the input is long/double, insert necessary
                # casts to replace the operation
                if param.dtype in (torch.int64, torch.float64):
                    # Ensure outputs are only repaired once per submodule to avoid
                    # unnecessary ops showing up in the graph
                    if not repaired_outputs_once:
                        submodule_outputs = submodule(*submodule_inputs)

                    repair_long_or_double_input(
                        partitioned_module,
                        position,
                        name,
                        None if repaired_outputs_once else submodule_outputs,
                        param.dtype,
                    )

                    repaired_outputs_once = True

                    # Repair submodule inputs in accordance with inserted casts
                    dtype_32bit = (
                        torch.int32 if (param.dtype == torch.int64) else torch.float32
                    )
                    submodule_inputs = (
                        submodule_inputs[:position]
                        + (param.to(dtype_32bit),)
                        + submodule_inputs[position + 1 :]
                    )

        # Create TRT Module from submodule
        trt_mod = convert_module(
            submodule,
            submodule_inputs,
            settings=settings,
        )

        # Replace FX Module with TRT Module
        setattr(partitioned_module, name, trt_mod)

    return partitioned_module
