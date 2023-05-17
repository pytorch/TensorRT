from copy import deepcopy
from functools import partial
from typing import List, Sequence
import torch
from torch_tensorrt.dynamo.backend.lowering._decompositions import (
    get_decompositions,
)
from torch_tensorrt.dynamo.backend.lowering._partition import (
    partition,
)

from torch._dynamo.backends.common import fake_tensor_unsupported

from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_compiler


@fake_tensor_unsupported
def fx_dynamo_testing_backend(
    gm: torch.fx.GraphModule,
    sample_inputs: Sequence[torch.Tensor],
    *,
    store_intermediate_graphs: List,
):
    """Helper Dynamo backend exclusively for testing"""
    custom_backend = partial(
        compile_module_testing,
        store_intermediate_graphs=store_intermediate_graphs,
    )

    # Invoke AOTAutograd to translate operators to aten
    return aot_module_simplified(
        gm,
        sample_inputs,
        fw_compiler=make_boxed_compiler(custom_backend),
        decompositions=get_decompositions(),
    )


def compile_module_testing(
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[torch.Tensor],
    *,
    store_intermediate_graphs: List,
) -> torch.fx.GraphModule:
    """Helper compiler exclusively for testing"""
    partitioned_module = partition(gm)

    # Store intermediate graph from partitioned module
    store_intermediate_graphs.append(deepcopy(partitioned_module))

    return partitioned_module


def same_output_format(trt_output, torch_output, enforce_tensor_type=True):
    # For each encountered collection type, ensure the torch and trt outputs agree
    # on type and size, checking recursively through all member elements.
    if isinstance(trt_output, tuple):
        return (
            isinstance(torch_output, tuple)
            and (len(trt_output) == len(torch_output))
            and all(
                same_output_format(trt_entry, torch_entry, enforce_tensor_type)
                for trt_entry, torch_entry in zip(trt_output, torch_output)
            )
        )
    elif isinstance(trt_output, list):
        return (
            isinstance(torch_output, list)
            and (len(trt_output) == len(torch_output))
            and all(
                same_output_format(trt_entry, torch_entry, enforce_tensor_type)
                for trt_entry, torch_entry in zip(trt_output, torch_output)
            )
        )
    elif isinstance(trt_output, dict):
        return (
            isinstance(torch_output, dict)
            and (len(trt_output) == len(torch_output))
            and (trt_output.keys() == torch_output.keys())
            and all(
                same_output_format(
                    trt_output[key], torch_output[key], enforce_tensor_type
                )
                for key in trt_output.keys()
            )
        )
    elif isinstance(trt_output, set) or isinstance(trt_output, frozenset):
        raise AssertionError(
            "Unsupported output type 'set' encountered in output format check."
        )
    elif enforce_tensor_type:
        return type(trt_output) is type(torch_output)
    else:
        return True
