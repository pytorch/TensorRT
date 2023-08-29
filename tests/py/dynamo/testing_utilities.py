import unittest
from copy import deepcopy
from functools import partial
from typing import Any, List, Sequence, Set

import torch
from torch._dynamo.utils import detect_fake_mode
from torch._functorch.aot_autograd import aot_export_joint_simple
from torch_tensorrt.dynamo import partitioning
from torch_tensorrt.dynamo.lowering import (
    apply_lowering_passes,
    get_decompositions,
    replace_builtin_inplace_ops,
)
from torch_tensorrt.dynamo.lowering._pre_aot_lowering import pre_aot_substitutions

DECIMALS_OF_AGREEMENT = 4


def fx_dynamo_testing_backend(
    gm: torch.fx.GraphModule,
    sample_inputs: Sequence[torch.Tensor],
    *,
    store_intermediate_graphs: List,
    min_block_size: int = 3,
    torch_executed_ops: Sequence[str] = set(),
    use_fast_partitioner: bool = True,
):
    """Helper Dynamo backend exclusively for testing"""
    custom_backend = partial(
        compile_module_testing,
        store_intermediate_graphs=store_intermediate_graphs,
        min_block_size=min_block_size,
        torch_executed_ops=torch_executed_ops,
        use_fast_partitioner=use_fast_partitioner,
    )

    gm = pre_aot_substitutions(gm)

    fake_mode = detect_fake_mode(sample_inputs)

    # Place backend tracing within FakeTensor context allowing nonfake Tensors
    with unittest.mock.patch.object(
        fake_mode, "allow_non_fake_inputs", True
    ), fake_mode:
        replace_builtin_inplace_ops(gm)

        # Invoke AOTAutograd to translate operators to aten
        gm = aot_export_joint_simple(
            gm,
            sample_inputs,
            decompositions=get_decompositions(),
        )

        gm = apply_lowering_passes(gm)

        trt_compiled = custom_backend(
            gm,
            sample_inputs,
        )
        return trt_compiled


def compile_module_testing(
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[torch.Tensor],
    *,
    store_intermediate_graphs: List,
    min_block_size: int = 3,
    torch_executed_ops: Sequence[str] = str(),
    use_fast_partitioner: bool = True,
) -> torch.fx.GraphModule:
    """Helper compiler exclusively for testing"""
    if use_fast_partitioner:
        partitioned_module = partitioning.fast_partition(
            gm,
            min_block_size=min_block_size,
            torch_executed_ops=torch_executed_ops,
        )
    else:
        partitioned_module = partitioning.global_partition(
            gm,
            min_block_size=min_block_size,
            torch_executed_ops=torch_executed_ops,
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
    use_fast_partitioner: bool = True,
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
        use_fast_partitioner: Whether to use the fast or global partitioner
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
        use_fast_partitioner=use_fast_partitioner,
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
            elif top_level_node.op == "call_module" and (
                not testing_partitioning
                or not use_fast_partitioner
                or ("_run_on_acc_" in top_level_node.target)
            ):
                for node in fx_module.get_submodule(top_level_node.target).graph.nodes:
                    classify_node(node)

    # Return unexpected ops seen and expected ops unseen
    # If the run was successful, both sets should be empty
    expected_ops_unseen = expected_ops.difference(expected_ops_seen)

    if testing_partitioning:
        return unexpected_ops_seen, expected_ops_unseen, partitioned_graphs

    else:
        return unexpected_ops_seen, expected_ops_unseen
