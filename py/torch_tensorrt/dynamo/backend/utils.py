import torch
from torch.fx.node import _get_qualified_name
from typing import Any, Optional, Union, Sequence, Dict
from torch_tensorrt import _Input, Device


def prepare_inputs(
    inputs: Union[_Input.Input, torch.Tensor, Sequence, Dict],
    device: torch.device = torch.device("cuda"),
) -> Any:
    if isinstance(inputs, _Input.Input):
        if isinstance(inputs.shape, dict):
            return inputs.example_tensor(optimization_profile_field="opt_shape").to(
                device
            )
        else:
            return inputs.example_tensor().to(device)

    elif isinstance(inputs, torch.Tensor):
        return inputs

    elif isinstance(inputs, list):
        prepared_input = list()

        for input_obj in inputs:
            prepared_input.append(prepare_inputs(input_obj))

        return prepared_input

    elif isinstance(inputs, tuple):
        prepared_input = list()

        for input_obj in inputs:
            prepared_input.append(prepare_inputs(input_obj))

        return tuple(prepared_input)

    elif isinstance(inputs, dict):
        prepared_input = dict()

        for key, input_obj in inputs.items():
            prepared_input[key] = prepare_inputs(input_obj)

        return prepared_input

    else:
        raise ValueError(
            f"Invalid input type {type(inputs)} encountered in the dynamo_compile input parsing. "
            + "Allowed input types: {torch_tensorrt.Input, torch.Tensor, list, tuple, dict}"
        )


def prepare_device(device: Union[Device, torch.device]) -> torch.device:
    if isinstance(device, Device):
        if device.gpu_id != -1:
            device = torch.device(device.gpu_id)
        else:
            raise ValueError("Invalid GPU ID provided for the CUDA device provided")

    elif isinstance(device, torch.device):
        device = device

    else:
        raise ValueError(
            "Invalid device provided. Supported options: torch.device | torch_tensorrt.Device"
        )

    return device


def _extract_downstream_get_nodes(
    module_node: torch.fx.Node, output_indices: Sequence[int]
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


def repair_long_or_double_input(
    gm: torch.fx.GraphModule,
    position: int,
    submodule_name: str,
    submodule_outputs: Optional[Union[torch.Tensor, Sequence[torch.Tensor]]],
    dtype: torch.dtype,
):
    """Fixes Long/Double type inputs to TRT-accelerated subgraphs

    In-Place modifies the provided graph

    Inserts a cast to the 32-bit equivalent type for TRT, then if necessary,
    inserts an upcast back to the 64-bit type for subsequent Torch operations

    Args:
        gm: FX GraphModule enclosing the TRT subgraph
        position: Index in the submodule inputs at which the long or double input is found
        submodule_name: Name of TRT-accelerated subgraph module in FX graph
        submodule_outputs: Output tensor(s) of TRT-accelerated subgraph (used for dtypes/structure)
        dtype: Data type of tensor at position in submodule (double/long)
    """
    assert dtype in (
        torch.int64,
        torch.float64,
    ), f"dtype argument must be torch.int64 or torch.float64, got {dtype}"

    # Determine target data type in 32 and 64 bit forms
    dtype_64bit = dtype
    dtype_32bit = torch.int32 if (dtype == torch.int64) else torch.float32

    # Find the node representing the submodule in the graph
    module_node = None

    # Iterate over all nodes in the graph, seeking target module name match
    for n in gm.graph.nodes:
        if n.op == "call_module" and str(n.target) == submodule_name:
            module_node = n
            break

    if module_node is None:
        raise AssertionError(
            f"Sought module node {submodule_name}, could not find in graph:\n{gm.graph}"
        )

    # Extract the 64-bit node of the input
    node_64bit = module_node.all_input_nodes[position]

    # Prior to the module, insert a cast to the 32-bit equivalent node
    with gm.graph.inserting_before(module_node):
        node_32bit = gm.graph.call_function(
            torch.ops.aten._to_copy.default,
            args=(node_64bit,),
            kwargs={"dtype": dtype_32bit},
        )

    # Replace 64-bit input to TRT module with new 32-bit cast node
    module_node.replace_input_with(node_64bit, node_32bit)

    output_positions_64bit = set()
    outputs_list = (
        [submodule_outputs]
        if isinstance(submodule_outputs, torch.Tensor)
        else submodule_outputs
    )

    # Determine if any outputs of the model are 64-bit type and store their indices
    if submodule_outputs is not None:
        for output_position, output in enumerate(outputs_list):
            if output.dtype == dtype_64bit:
                output_positions_64bit.add(output_position)

    # Only enter this code block if there exists a 64-bit output
    # This implies a cast is needed, since TRT cannot output 64-bit tensors
    if output_positions_64bit:
        # Determine whther the outputs of the module are tuple-type or not
        is_collection_output = False
        if isinstance(submodule_outputs, tuple):
            is_collection_output = True

        if not is_collection_output:
            # If the output is a single tensor, insert a cast back to int64
            with gm.graph.inserting_after(module_node):
                cast_node_64bit = gm.graph.call_function(
                    torch.ops.aten._to_copy.default,
                    args=(module_node,),
                    kwargs={"dtype": dtype_64bit},
                )

            # Replace all uses of the TRT module (except the cast node) with the 64-bit equivalent
            module_node.replace_all_uses_with(
                cast_node_64bit, delete_user_cb=lambda user: (user != cast_node_64bit)
            )

        else:
            # If the output is a tuple of tensors, extract downstream users for each 64-bit output
            get_nodes = _extract_downstream_get_nodes(
                module_node, output_positions_64bit
            )

            # For each downstream user, append a cast node back to the 64-bit precision
            for get_node in get_nodes:
                with gm.graph.inserting_after(get_node):
                    cast_node_64bit = gm.graph.call_function(
                        torch.ops.aten._to_copy.default,
                        args=(get_node,),
                        kwargs={"dtype": torch.int64},
                    )

                get_node.replace_all_uses_with(
                    cast_node_64bit,
                    delete_user_cb=lambda user: (user != cast_node_64bit),
                )

    # Clean up graph and ensure invariants are preserved
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
