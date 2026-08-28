from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Set

import torch
from torch.fx.node import _get_qualified_name
from torch_tensorrt._enums import dtype
from torch_tensorrt._Input import Input
from torch_tensorrt.dynamo.utils import get_output_metadata, get_torch_inputs

logger = logging.getLogger(__name__)


def _extract_downstream_get_nodes(
    module_node: torch.fx.Node, output_indices: Set[int]
) -> Sequence[torch.fx.Node]:
    """Extracts downstream users of a node which get the item at a particular index

    Certain module-type nodes have multiple outputs (tuple of outputs). This function
    returns downstream nodes which call the _operator.getitem function, which extracts
    the element at a particular index in the tuple

    Args:
        module_node: FX module-type node to analyze
        output_index: Indices in the module node output to search for
    Returns:
        List of nodes which get the item at the specified index in the module node output
    """
    get_nodes = []

    # Iterate over all downstream users of the node object
    for user in module_node.users:
        # If the user is a "get" node accessing the specified index, store it
        if _get_qualified_name(user.target) == "_operator.getitem" and (
            user.args[1] in output_indices
        ):
            get_nodes.append(user)

    return get_nodes


def _metadata_dtype(metadata: Dict[str, Any]) -> Optional[torch.dtype]:
    """Return the dtype of tensor metadata, ignoring scalar outputs."""
    value = metadata.get("val")
    if isinstance(value, torch.Tensor):
        return value.dtype

    tensor_meta = metadata.get("tensor_meta")
    return getattr(tensor_meta, "dtype", None)


def _metadata_to_dtype(
    metadata: Dict[str, Any], target_dtype: torch.dtype
) -> Dict[str, Any]:
    """Copy tensor metadata while changing its dtype."""
    updated = metadata.copy()
    value = updated.get("val")
    if isinstance(value, torch.Tensor):
        updated["val"] = value.to(target_dtype)

    tensor_meta = updated.get("tensor_meta")
    if tensor_meta is not None and hasattr(tensor_meta, "_replace"):
        updated["tensor_meta"] = tensor_meta._replace(dtype=target_dtype)

    return updated


def _repair_64bit_input(
    gm: torch.fx.GraphModule,
    position: int,
    submodule_name: str,
    submodule_output_metadata: Optional[Sequence[Dict[str, Any]]],
    is_collection_output: bool,
    dtype: torch.dtype,
) -> None:
    """Fix a single double input and any double outputs at a TRT boundary.

    The output dtypes come from the partition's FX metadata. Compilation must not
    execute the partition merely to discover information already recorded there.
    """
    assert dtype == torch.float64, f"dtype argument must be torch.float64, got {dtype}"

    logger.info(
        f"Downcasting a 64-bit input at position {position} of submodule {submodule_name}"
    )

    dtype_64bit = dtype
    dtype_32bit = torch.float32

    module_node = None
    for node in gm.graph.nodes:
        if node.op == "call_module" and str(node.target) == submodule_name:
            module_node = node
            break

    if module_node is None:
        raise AssertionError(
            f"Sought module node {submodule_name}, could not find in graph:\n{gm.graph}"
        )

    node_64bit = module_node.all_input_nodes[position]
    with gm.graph.inserting_before(module_node):
        node_32bit = gm.graph.call_function(
            torch.ops.aten._to_copy.default,
            args=(node_64bit,),
            kwargs={"dtype": dtype_32bit},
        )
        node_32bit.meta = _metadata_to_dtype(node_64bit.meta, dtype_32bit)

    module_node.replace_input_with(node_64bit, node_32bit)

    output_positions_64bit: Set[int] = set()
    original_output_metadata = list(submodule_output_metadata or ())
    truncated_output_metadata = []
    for output_position, metadata in enumerate(original_output_metadata):
        if _metadata_dtype(metadata) == dtype_64bit:
            output_positions_64bit.add(output_position)
            truncated_output_metadata.append(_metadata_to_dtype(metadata, dtype_32bit))
        else:
            truncated_output_metadata.append(metadata.copy())

    # The call_module node describes the actual engine boundary. Preserve its
    # container convention while correcting tensor dtypes to what TRT emits.
    if truncated_output_metadata:
        for key in ("val", "tensor_meta"):
            values = [
                metadata[key]
                for metadata in truncated_output_metadata
                if key in metadata
            ]
            if not values:
                continue
            current = module_node.meta.get(key)
            if isinstance(current, tuple):
                module_node.meta[key] = tuple(values)
            elif isinstance(current, list) or len(values) > 1:
                module_node.meta[key] = values
            else:
                module_node.meta[key] = values[0]

    if output_positions_64bit:
        if not is_collection_output:
            with gm.graph.inserting_after(module_node):
                cast_node_64bit = gm.graph.call_function(
                    torch.ops.aten._to_copy.default,
                    args=(module_node,),
                    kwargs={"dtype": dtype_64bit},
                )
                cast_node_64bit.meta = original_output_metadata[0].copy()

            module_node.replace_all_uses_with(
                cast_node_64bit, delete_user_cb=lambda user: user != cast_node_64bit
            )
        else:
            get_nodes = _extract_downstream_get_nodes(
                module_node, output_positions_64bit
            )
            for get_node in get_nodes:
                output_position = get_node.args[1]
                get_node.meta = truncated_output_metadata[output_position].copy()
                with gm.graph.inserting_after(get_node):
                    cast_node_64bit = gm.graph.call_function(
                        torch.ops.aten._to_copy.default,
                        args=(get_node,),
                        kwargs={"dtype": dtype_64bit},
                    )
                    cast_node_64bit.meta = original_output_metadata[
                        output_position
                    ].copy()

                get_node.replace_all_uses_with(
                    cast_node_64bit,
                    delete_user_cb=lambda user: user != cast_node_64bit,
                )

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()


def repair_double_inputs(
    parent_graph: torch.fx.GraphModule,
    submodule: torch.fx.GraphModule,
    submodule_inputs: Sequence[Input],
    device: torch.device,
    submodule_name: Optional[str] = None,
) -> Sequence[Input]:
    """Fixes all Long/Double type inputs to a TRT-accelerated subgraph

    In-Place modifies the provided graph

    Inserts a cast to the 32-bit equivalent type for TRT, then if necessary,
    inserts an upcast back to the 64-bit type for subsequent Torch operations

    Args:
        parent_graph: FX GraphModule enclosing the TRT subgraph
        submodule: Child submodule to repair inputs on
        submodule_inputs: Input tensor(s) of TRT-accelerated subgraph (used for dtypes/structure)
        submodule_name: Optionally specify the name of the submodule target in the parent graph
    Returns:
        New submodule inputs, updated accordingly with long/double truncation
    """
    submodule_torch_inputs = get_torch_inputs(submodule_inputs, device)
    num_submodule_inputs = len(submodule_inputs)
    repaired_outputs_once = False
    output_node = next(node for node in submodule.graph.nodes if node.op == "output")
    is_collection_output = isinstance(output_node.args[0], (tuple, list))
    submodule_output_metadata = get_output_metadata(submodule)

    # For each input to the TRT subgraph, check if its type is double.
    for position in range(num_submodule_inputs):
        param = submodule_torch_inputs[position]

        if isinstance(param, torch.Tensor) and param.dtype == torch.float64:
            _repair_64bit_input(
                parent_graph,
                position,
                submodule_name if submodule_name is not None else submodule._get_name(),
                None if repaired_outputs_once else submodule_output_metadata,
                is_collection_output,
                param.dtype,
            )

            repaired_outputs_once = True

            # Repair submodule inputs in accordance with inserted casts
            dtype_32bit = torch.float32
            submodule_torch_inputs = (
                list(submodule_torch_inputs[:position])
                + [
                    param.to(dtype_32bit),
                ]
                + list(submodule_torch_inputs[position + 1 :])
            )

            # Set the 32bit inputs and their types to the submodule Inputs
            for idx in range(len(submodule_inputs)):
                submodule_inputs[idx].torch_tensor = submodule_torch_inputs[idx]
                submodule_inputs[idx].dtype = dtype._from(
                    submodule_torch_inputs[idx].dtype
                )

    return submodule_inputs
