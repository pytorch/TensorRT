from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import tensorrt
import torch
import torch_tensorrt
from torch._export.utils import _detect_fake_mode_from_gm
from torch._ops import OpOverloadPacket
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx import Graph, GraphModule, Node
from torch.fx.node import Target
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils._pytree import _LEAF_SPEC


def get_kv_nodes(gm):
    """
    Extract key and value nodes from scaled dot-product attention operations in the graph.

    This function searches through the graph for scaled_dot_product_attention operations
    and extracts the key and value tensor nodes from each operation's arguments.

    Args:
        gm: A torch.fx.GraphModule containing the computational graph

    Returns:
        List[Tuple[Node, Node]]: A list of tuples, where each tuple contains
            (key_node, value_node) from a scaled dot-product attention operation
    """
    kv_nodes = []
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and node.target == torch._C._nn.scaled_dot_product_attention
        ):
            q_node, k_node, v_node = node.args[:3]
            kv_nodes.append((k_node, v_node))
    return kv_nodes


def get_random_tensor_from_node(node: Node) -> torch.Tensor:
    """
    Creates a random tensor based on the shape information in a node's metadata.
    For symbolic dimensions, extracts the maximum value from the shape environment.

    Args:
        node: A torch.fx.Node object with metadata containing tensor information

    Returns:
        A random tensor with shape matching the node's metadata, or None if no valid
        tensor information is found
    """
    if "val" not in node.meta:
        raise ValueError(
            f"No tensor information found in node metadata for node: {node}"
        )

    fake_tensor = node.meta["val"]
    shape = []

    # Iterate through each dimension and handle symbolic dimensions
    for dim in fake_tensor.shape:
        if isinstance(dim, torch.SymInt):
            # Extract the maximum value from the shape environment
            max_val = dim.node.hint
            shape.append(max_val)
        else:
            shape.append(dim)

    # Create a random tensor with the determined shape
    dtype = fake_tensor.dtype
    device = fake_tensor.device
    random_tensor = torch.rand(shape, dtype=dtype, device=device)

    return random_tensor


def create_random_output_tensors(nodes: List[Node]) -> List[torch.Tensor]:
    """
    Creates random tensors based on the shape information in node metadata.
    For symbolic dimensions, extracts the maximum value from the shape environment.

    Args:
        nodes: List of torch.fx.Node objects with metadata

    Returns:
        List of random tensors with shapes matching the nodes' metadata
    """
    random_tensors = []

    for node in nodes:
        if isinstance(node, Node):
            node_tensor = get_random_tensor_from_node(node)
        elif isinstance(node, tuple):
            node_tensor_list = []
            for n in node:
                random_tensor = get_random_tensor_from_node(n)
                node_tensor_list.append(random_tensor)
            node_tensor = tuple(node_tensor_list)

        random_tensors.append(node_tensor)

    return random_tensors


def _add_graph_input(
    gm: GraphModule, name: str, val: Optional[torch.Tensor] = None, dynamic_shape=None
) -> Node:
    """Add a graph input to the given GraphModule and return the newly created node.

    NOTE: function does NOT do any graph canonicalization. This is left to the user!

    Args:
        gm (GraphModule): The GraphModule to add the input to.
        name (str): The name of the input.
        val (torch.Tensor): An example tensor to use for the input.
        dynamic_shape: The dynamic shape of the input tensor [NOT SUPPORTED YET]
    """
    # check that no dynamic shape is provided...
    if dynamic_shape:
        raise NotImplementedError("Dynamic shape not supported for adding graph inputs")

    # extract graph and input spec
    graph: Graph = gm.graph

    in_spec = graph._codegen.pytree_info.in_spec
    in_spec_for_args = in_spec.children_specs[0]
    orig_args = graph._codegen.pytree_info.orig_args
    assert in_spec_for_args.type is tuple

    # insert input node after currently last input node
    node_last_input = graph.find_nodes(op="placeholder", sort=True)[-1]
    with graph.inserting_after(node_last_input):
        in_node = graph.placeholder(name)
        in_spec_for_args.children_specs.append(_LEAF_SPEC)
        orig_args.append(f"arg_{name}")

    # update pytree info recursively with __post_init__ starting at leaves
    def call_post_init(spec):
        for child_spec in spec.children_specs:
            call_post_init(child_spec)
        spec.__post_init__()

    call_post_init(in_spec)

    # set fake tensor information if all required information is available
    fake_mode: Optional[FakeTensorMode] = _detect_fake_mode_from_gm(gm)
    if fake_mode and val is not None and isinstance(val, torch.Tensor):
        if isinstance(val, FakeTensor):
            fake_tensor = val
        else:
            fake_tensor: FakeTensor = fake_mode.from_tensor(val, static_shapes=True)
        in_node.meta["val"] = fake_tensor
        in_node.meta["tensor_meta"] = _extract_tensor_metadata(fake_tensor)

    # return new node...
    return in_node


def is_op(node: Node, ops: Union[OpOverloadPacket, Iterable[OpOverloadPacket]]) -> bool:
    """Check if the node is a call to one of the ops."""
    if node.op != "call_function":
        return False
    # check if it's a single op that's provided
    if isinstance(ops, OpOverloadPacket):
        ops = [ops]

    # check if it's the op itself instead of an overload
    if any(node.target == op for op in ops):
        return True

    return False


def get_all_input_output_nodes(graph: Graph) -> Tuple[List[Node], List[Node]]:
    input_nodes: List[Node] = graph.find_nodes(op="placeholder")
    output_nodes: List[Node] = graph.find_nodes(op="output")
    return (input_nodes, output_nodes)
