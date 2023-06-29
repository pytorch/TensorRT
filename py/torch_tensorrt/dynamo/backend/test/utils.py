from copy import deepcopy
from functools import partial
from typing import Any, List, Sequence, Set
import torch
from torch_tensorrt.dynamo.backend.lowering._decompositions import (
    get_decompositions,
)
from torch_tensorrt.dynamo.backend.lowering._partition import (
    partition,
)
from torch_tensorrt.dynamo.backend.lowering._pre_aot_lowering import (
    pre_aot_substitutions,
)

from torch._dynamo.backends.common import fake_tensor_unsupported

from torch._functorch.aot_autograd import aot_module_simplified, make_boxed_compiler


@fake_tensor_unsupported
def fx_dynamo_testing_backend(
    gm: torch.fx.GraphModule,
    sample_inputs: Sequence[torch.Tensor],
    *,
    store_intermediate_graphs: List,
    min_block_size: int = 3,
    torch_executed_ops: Sequence[str] = set(),
):
    """Helper Dynamo backend exclusively for testing"""
    custom_backend = partial(
        compile_module_testing,
        store_intermediate_graphs=store_intermediate_graphs,
        min_block_size=min_block_size,
        torch_executed_ops=torch_executed_ops,
    )

    gm = pre_aot_substitutions(gm)

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
    min_block_size: int = 3,
    torch_executed_ops: Sequence[str] = str(),
) -> torch.fx.GraphModule:
    """Helper compiler exclusively for testing"""
    partitioned_module = partition(
        gm, min_block_size=min_block_size, torch_executed_ops=torch_executed_ops
    )

    # Store intermediate graph from partitioned module
    store_intermediate_graphs.append(deepcopy(partitioned_module))

    return partitioned_module


def same_output_format(trt_output, torch_output, enforce_tensor_type=True):
    """Determines whether two objects containing Tensors have the same format

    ((Tensor, Tensor), Tensor) and (Tensor (Tensor, Tensor)) do not
    have the same format, for example.

    Args:
        trt_output: TensorRT output
        torch_output: Torch output
        enforce_tensor_type: Whether to enforce Tensor type equivalence
    Returns:
        bool: True if the outputs have the same format
    """
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


def lower_graph_testing(
    fx_graph: torch.fx.GraphModule,
    inputs: Any,
    *,
    expected_ops: Set = set(),
    unexpected_ops: Set = set(),
    min_block_size: int = 3,
    torch_executed_ops: Sequence[str] = set(),
    testing_partitioning: bool = False,
):
    """Helper function to assist with graph lowering for testing of Dynamo compile

    Args:
        fx_graph: Graph to lower
        inputs: Input values to the FX graph
        expected_ops: Operations to be expected in the lowered graph
        unexpected_ops: Operations not to be expected in the lowered graph
        min_block_size: Minimum number of operators per TRT-Engine Block
        torch_executed_ops: Sequence of operations to run in Torch, regardless of converter coverage
        testing_partitioning: Whether partitioning is being tested (to analyze only TRT-supported ops)
    Returns:
        If testing_partitioning:
            List[torch.fx.GraphModule], Set, Set: List of partitioned graph outputs, unexpected ops seen, expected ops unseen
        Else:
            Set, Set: unexpected ops seen and expected ops unseen (If the run was successful, both sets should be empty)
    """
    # Trace module and set up custom backend to track intermediate graphs
    partitioned_graphs = []
    custom_backend = partial(
        fx_dynamo_testing_backend,
        store_intermediate_graphs=partitioned_graphs,
        min_block_size=min_block_size,
        torch_executed_ops=torch_executed_ops,
    )

    # Invoke compilation
    compiled_graph = torch.compile(fx_graph, backend=custom_backend)
    compiled_graph(*inputs)

    unexpected_ops_seen = set()
    expected_ops_seen = set()

    def classify_node(node: torch.fx.Node):
        if node.target in unexpected_ops:
            unexpected_ops_seen.add(node.target)
        elif node.target in expected_ops:
            expected_ops_seen.add(node.target)

    # Iterate over intermediate graphs, attempt to match nodes
    # If an unexpected or expected op is encountered, register it
    for fx_module in partitioned_graphs:
        # For each function call in the set of graph nodes, classify the node
        for top_level_node in fx_module.graph.nodes:
            if top_level_node.op == "call_function" and not testing_partitioning:
                classify_node(top_level_node)
            elif top_level_node.op == "call_module":
                for node in fx_module.get_submodule(top_level_node.target).graph.nodes:
                    classify_node(node)

    # Return unexpected ops seen and expected ops unseen
    # If the run was successful, both sets should be empty
    expected_ops_unseen = expected_ops.difference(expected_ops_seen)

    if testing_partitioning:
        return unexpected_ops_seen, expected_ops_unseen, partitioned_graphs

    else:
        return unexpected_ops_seen, expected_ops_unseen
