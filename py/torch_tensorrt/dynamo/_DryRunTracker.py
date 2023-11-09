import logging
import math
from dataclasses import dataclass, field
from typing import List, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class PerSubgraphData:
    """Class to track data on a per-subgraph level

    Args:
        subgraph_name (str): Name of the subgraph in the GraphModule
        subgraph_op_count (int): Number of operations in the subgraph
        subgraph_input_shapes (List[Tuple[int, ...]]): Shapes of input Tensors of the subgraph
        subgraph_input_dtypes (List[torch.device]): Input data types of the subgraph
        subgraph_output_shapes (List[Tuple[int, ...]]): Shapes of output Tensors of the subgraph
        subgraph_output_dtypes (List[torch.device]): Output data types of the subgraph
    """

    subgraph_name: str = ""
    subgraph_op_count: int = 0
    subgraph_input_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    subgraph_input_dtypes: List[torch.device] = field(default_factory=list)
    subgraph_output_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    subgraph_output_dtypes: List[torch.device] = field(default_factory=list)


@dataclass
class DryRunTracker:
    """Class to track data on a graph-wide level

    Args:
        total_ops_in_graph (int): Total number of operators in graph
        supported_ops_in_graph (int): Number of supported operators in graph
        graph_input_shapes (List[Tuple[int, ...]]): Shapes of input Tensors of the graph
        graph_input_dtypes (List[torch.device]): Input data types of the graph
        graph_output_shapes (List[Tuple[int, ...]]): Shapes of output Tensors of the graph
        graph_output_dtypes (List[torch.device]): Output data types of the graph
        per_subgraph_data (List[PerSubgraphData]): Per-subgraph data, see above class
        tensorrt_graph_count (int): Number of TensorRT engines to be generated
        truncated_long_and_double (bool): Whether truncate_long_and_double was enabled
    """

    total_ops_in_graph: int = 0
    supported_ops_in_graph: int = 0
    graph_input_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    graph_input_dtypes: List[torch.device] = field(default_factory=list)
    graph_output_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    graph_output_dtypes: List[torch.device] = field(default_factory=list)
    per_subgraph_data: List[PerSubgraphData] = field(default_factory=list)
    tensorrt_graph_count: int = 0
    truncated_long_and_double: bool = False


def dryrun_stats_display(dryrun_tracker: DryRunTracker, dryrun_enabled: bool) -> None:
    """Displays statistics about the dryrun either to debug logs or info logs"""
    # If user specified "dryrun=True", print to info logs, else debug
    if dryrun_enabled:
        dryrun_logger = logger.info
    else:
        dryrun_logger = logger.debug

    formatted_stats = "\n"

    # Print overall stats about the graph, operator counts, etc.
    formatted_stats += "+" * 50 + " Dry-Run Results for Graph " + "+" * 50 + "\n"
    formatted_stats += (
        f"The graph consists of {dryrun_tracker.total_ops_in_graph} Total Operators, "
        f"of which {dryrun_tracker.supported_ops_in_graph} operators are supported, "
        f"{round(dryrun_tracker.supported_ops_in_graph*100/dryrun_tracker.total_ops_in_graph, 2)}% coverage\n"
    )
    formatted_stats += f"Long and double inputs were {'' if dryrun_tracker.truncated_long_and_double else 'not'} truncated (truncate_long_and_double={dryrun_tracker.truncated_long_and_double})\n"
    formatted_stats += (
        f"{dryrun_tracker.tensorrt_graph_count} TRT Engine(s) were generated\n"
    )

    assert len(dryrun_tracker.per_subgraph_data) == dryrun_tracker.tensorrt_graph_count

    # Print schematic of the graph structure, as in:
    #
    #   Inputs: [Tensor: (1, 3, 224, 224)@float32]
    #    ...
    #    TRT Engine #1: _run_on_acc_0
    #     Engine Inputs: [Tensor: (1, 3, 224, 224)@float32]
    #     Number of Operators in Engine: 1
    #     Engine Outputs: [Tensor: (1, 64, 112, 112)@float32]
    #    ...
    #   Outputs: [Tensor: (1, 1000)@float32]
    #
    formatted_stats += " " * 2 + "Graph Structure:\n\n"
    formatted_stats += (
        " " * 3
        + f"Inputs: [{input_formatter(dryrun_tracker.graph_input_shapes, dryrun_tracker.graph_input_dtypes)}]\n"
    )

    for i, trt_subgraph_data in enumerate(dryrun_tracker.per_subgraph_data):
        assert len(trt_subgraph_data.subgraph_input_dtypes) == len(
            trt_subgraph_data.subgraph_input_shapes
        )
        assert len(trt_subgraph_data.subgraph_output_dtypes) == len(
            trt_subgraph_data.subgraph_output_shapes
        )
        formatted_stats += " " * 4 + "...\n"
        formatted_stats += (
            " " * 4 + f"TRT Engine #{i+1}: {trt_subgraph_data.subgraph_name}\n"
        )
        formatted_stats += (
            " " * 5
            + f"Engine Inputs: [{input_formatter(trt_subgraph_data.subgraph_input_shapes, trt_subgraph_data.subgraph_input_dtypes)}]\n"
        )
        formatted_stats += (
            " " * 5
            + f"Number of Operators in Engine: {trt_subgraph_data.subgraph_op_count}\n"
        )
        formatted_stats += (
            " " * 5
            + f"Engine Outputs: [{input_formatter(trt_subgraph_data.subgraph_output_shapes, trt_subgraph_data.subgraph_output_dtypes)}]\n"
        )

    formatted_stats += " " * 4 + "...\n"
    formatted_stats += (
        " " * 3
        + f"Outputs: [{input_formatter(dryrun_tracker.graph_output_shapes, dryrun_tracker.graph_output_dtypes)}]\n"
    )

    # Print aggregate statistics about the graph structure, including recommended "min_block_size" options
    if dryrun_tracker.tensorrt_graph_count > 0:
        min_ops_in_an_engine = min(
            trt_subgraph.subgraph_op_count
            for trt_subgraph in dryrun_tracker.per_subgraph_data
        )
        avg_ops_per_engine = (
            sum(
                trt_subgraph.subgraph_op_count
                for trt_subgraph in dryrun_tracker.per_subgraph_data
            )
            / dryrun_tracker.tensorrt_graph_count
        )
        avg_ops_per_engine = round(avg_ops_per_engine, 2)
        most_ops_in_an_engine = max(
            trt_subgraph.subgraph_op_count
            for trt_subgraph in dryrun_tracker.per_subgraph_data
        )

        formatted_stats += "\n" + " " * 2 + "-" * 25 + " Aggregate Stats " + "-" * 25
        formatted_stats += (
            "\n\n"
            + " " * 3
            + "Average Number of Operators per TRT Engine: "
            + f"{avg_ops_per_engine}"
        )

        formatted_stats += (
            "\n"
            + " " * 3
            + "Most Operators in a TRT Engine: "
            + f"{most_ops_in_an_engine}"
        )

        formatted_stats += "\n\n" + " " * 2 + "*" * 10 + " Recommendations " + "*" * 10
        formatted_stats += (
            "\n\n"
            + " " * 3
            + "- For minimal graph segmentation, select min_block_size="
            + f"{most_ops_in_an_engine} which would generate "
            + f"{len([1 for trt_subgraph in dryrun_tracker.per_subgraph_data if trt_subgraph.subgraph_op_count >= most_ops_in_an_engine])} TRT engines"
        )
        if math.ceil(avg_ops_per_engine) != most_ops_in_an_engine:
            formatted_stats += (
                "\n"
                + " " * 3
                + "- For moderate graph segmentation, select min_block_size="
                + f"{math.ceil(avg_ops_per_engine)} which would generate "
                + f"{len([1 for trt_subgraph in dryrun_tracker.per_subgraph_data if trt_subgraph.subgraph_op_count >= math.ceil(avg_ops_per_engine)])} TRT engines"
            )

        formatted_stats += (
            "\n"
            + " " * 3
            + "- The current level of graph segmentation is equivalent to selecting min_block_size="
            + f"{min_ops_in_an_engine} which generates "
            + f"{len([1 for trt_subgraph in dryrun_tracker.per_subgraph_data if trt_subgraph.subgraph_op_count >= min_ops_in_an_engine])} TRT engines"
        )
    else:
        formatted_stats += (
            "\n"
            + " " * 2
            + "Aggregate stats not available since no TRT Engines were generated."
        )

    dryrun_logger(formatted_stats)


def input_formatter(shapes: List[Tuple[int, ...]], dtypes: List[torch.dtype]) -> str:
    """Format shapes and dtypes of input Tensors into a readable string"""
    formatted_str = ", "

    for shape, dtype in zip(shapes, dtypes):
        formatted_str += f"Tensor: {shape}@{str(dtype)[6:]}, "

    return formatted_str[2:-2]
