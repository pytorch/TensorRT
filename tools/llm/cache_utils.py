# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Iterable, List, Optional, Tuple, Union

import torch
from torch._export.utils import _detect_fake_mode_from_gm
from torch._ops import OpOverloadPacket
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx import Graph, GraphModule, Node
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils._pytree import _LEAF_SPEC, TreeSpec


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


_PYTREE_INFO_FIELDS = ("orig_args", "in_spec", "out_spec")


class PytreeInfoUnavailableError(RuntimeError):
    """The graph has no pytree codegen info (it was not produced with a pytree calling convention)."""


def _get_pytree_info(graph: Graph):
    """Return the graph's pytree codegen info, the one place this module touches the private
    ``Graph._codegen`` / ``_PyTreeInfo``.

    torch.fx has no public API for changing a GraphModule's calling convention, so the info is read
    and replaced through the codegen.

    Raises:
        PytreeInfoUnavailableError: the graph has no pytree codegen info.
        RuntimeError: the info exists but torch changed its layout, so it cannot be used safely.
    """
    info = getattr(getattr(graph, "_codegen", None), "pytree_info", None)
    if info is None:
        raise PytreeInfoUnavailableError(
            "Graph has no pytree codegen info (graph._codegen.pytree_info)"
        )
    missing = [f for f in (*_PYTREE_INFO_FIELDS, "_replace") if not hasattr(info, f)]
    if missing:
        raise RuntimeError(
            f"Unsupported torch.fx pytree info layout for torch {torch.__version__}: "
            f"{type(info).__name__} is missing {missing}"
        )
    return info


def _set_pytree_info(graph: Graph, **changes: Any) -> None:
    """Replace fields of the graph's pytree info (see ``_get_pytree_info``)."""
    info = _get_pytree_info(graph)
    graph._codegen.pytree_info = info._replace(**changes)


def add_graph_input(
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

    # TreeSpec is frozen: validate everything first, then rebuild the specs with the extra leaf
    # so a failure cannot leave the graph and its pytree info out of sync.
    info = _get_pytree_info(graph)
    in_spec: TreeSpec = info.in_spec
    if in_spec.type is not tuple or in_spec.num_children != 2:
        raise ValueError(
            f"Unexpected in_spec structure (expected (args, kwargs)): {in_spec}"
        )
    args_spec = in_spec.child(0)
    if args_spec.type is not tuple:
        raise ValueError(
            f"Expected positional args spec to be a tuple, got {args_spec.type}"
        )
    if any(n.name == name for n in graph.nodes):
        raise ValueError(f"A node named '{name}' already exists in the graph")

    new_args_spec = TreeSpec(
        tuple, args_spec.context, args_spec.children() + [_LEAF_SPEC]
    )
    new_in_spec = TreeSpec(
        in_spec.type,
        in_spec.context,
        [new_args_spec, *in_spec.children()[1:]],
    )
    new_orig_args = [*info.orig_args, f"arg_{name}"]

    # insert input node after currently last input node
    node_last_input = graph.find_nodes(op="placeholder", sort=True)[-1]
    with graph.inserting_after(node_last_input):
        in_node = graph.placeholder(name)

    _set_pytree_info(graph, in_spec=new_in_spec, orig_args=new_orig_args)
    if hasattr(gm, "_in_spec"):
        gm._in_spec = new_in_spec

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


def set_output_spec(gm: GraphModule, out_spec: Any) -> None:
    """Make ``gm`` return outputs structured as ``out_spec`` (e.g. after appending KV-cache outputs).

    ``GraphModule.recompile()`` re-derives ``gm._out_spec`` from the graph's pytree codegen, so
    assigning the attribute alone is silently undone by any later recompile (Torch-TensorRT
    recompiles after the lowering passes). The codegen's ``pytree_info`` is the source of truth.
    """
    gm._out_spec = out_spec
    try:
        _set_pytree_info(gm.graph, out_spec=out_spec)
    except PytreeInfoUnavailableError:
        # no pytree codegen: ``gm._out_spec`` alone is authoritative
        pass


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
