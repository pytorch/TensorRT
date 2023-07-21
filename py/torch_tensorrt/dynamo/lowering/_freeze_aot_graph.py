import unittest
from typing import List, Sequence, Tuple

import torch
import torch.fx.traceback as fx_traceback
from torch._dynamo.utils import detect_fake_mode
from torch._functorch.compile_utils import fx_graph_cse
from torch._inductor.compile_fx import fake_tensor_prop
from torch._inductor.freezing import constant_fold, replace_params_with_constants
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.tools_common import legalize_graph


def freeze_autograd_gm(
    aot_autograd_gm: torch.fx.GraphModule,
    example_inputs: Sequence[torch._subclasses.FakeTensor],
) -> Tuple[torch.fx.GraphModule, List[int]]:
    """
    Adapted from:
    https://github.com/pytorch/pytorch/blob/750b9b359f06cb8b8c2d5b6118bba636e2112cbb/torch/_inductor/freezing.py#L186-L243

    Inlines parameters that are not mutated into constants and optimizes the graph through constant propagation
    and other techniques. If enabled, the function also discards the original parameters of the module for memory efficiency.

    Assumes that this function is run in dynamo tracing post aot_autograd.

    Args:
        aot_autograd_gm (torch.fx.GraphModule): The aot_autograd constructed GraphModule to be frozen.
        example_inputs (List[torch.Tensor]): A list of example input tensors to be used in the freezing process.

    Returns:
        Tuple[torch.fx.GraphModule, List[int]]: A tuple containing the frozen GraphModule and a list of indices
        of the inputs that were preserved (not turned into constants).
    """
    # Extract necessary metadata and parameters
    fw_metadata = torch._guards.TracingContext.get().fw_metadata
    params_flat = torch._guards.TracingContext.get().params_flat
    assert fw_metadata is not None and params_flat is not None

    # Replace placeholders with get_attr nodes
    preserved_arg_indices = replace_params_with_constants(
        aot_autograd_gm, params_flat, fw_metadata
    )

    constant_fold(aot_autograd_gm)

    fake_mode = detect_fake_mode(example_inputs)

    # constant params will be real tensors, not fake
    # TODO: fake_mode should should enable py dispatcher if its symbolic ?
    with unittest.mock.patch.object(
        fake_mode, "allow_non_fake_inputs", True
    ), fake_mode:
        args = [e for i, e in enumerate(example_inputs) if i in preserved_arg_indices]
        with fx_traceback.preserve_node_meta():
            aot_autograd_gm = make_fx(aot_autograd_gm, _allow_non_fake_inputs=True)(
                *args
            )

    # TODO - further restrict cse ? right now needed to dedup aliasing ops
    cse_graph = fx_graph_cse(aot_autograd_gm.graph)
    aot_autograd_gm.graph = cse_graph
    aot_autograd_gm.recompile()

    # Make sure meta['val'] is properly setup(weight conversion
    # or decompose_unfused_batchnorms lost meta['val']).
    aot_example_inputs = [example_inputs[ind] for ind in preserved_arg_indices]
    fake_tensor_prop(aot_autograd_gm, aot_example_inputs, True)

    # TODO - apply legalization in pattern matcher
    legalize_graph(aot_autograd_gm)
    constant_fold(aot_autograd_gm)

    return aot_autograd_gm, preserved_arg_indices
