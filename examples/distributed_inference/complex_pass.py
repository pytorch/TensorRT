from typing import List, Optional, Set

import torch
from torch.fx import GraphModule, Node


class ComplexSubGraphInfo:
    def __init__(
        self,
        anchor_nodes: List[Node],
        subgraph_nodes: List[Node],
        input_nodes: List[Node],
    ):
        self.anchor_nodes = anchor_nodes
        self.subgraph_nodes = subgraph_nodes
        self.input_nodes = input_nodes

    def __repr__(self):
        return (
            f"ComplexOpSubGraphInfo(anchor_nodes={[n.name for n in self.anchor_nodes]}, "
            f"subgraph={[n.name for n in self.subgraph_nodes]}, "
            f"inputs={[n.name for n in self.input_nodes]})"
        )


class ComplexOpDetector:
    def __init__(self):
        pass

    def is_complex_dtype(self, node: Node) -> bool:
        # Check if node's metadata or dtype is complex
        dtype = None
        if "val" in node.meta:
            val = node.meta["val"]
            if hasattr(val, "dtype"):
                dtype = val.dtype

        print("dtype of node:", dtype)
        return dtype in {torch.complex64, torch.complex128}

    def node_include_in_subgraph(self, node: Node) -> bool:
        # Include only call_function ops on complex tensors
        print("node.op:", node.op, "node name:", node.name)
        print("is_complex_dtype:", self.is_complex_dtype(node))
        return node.op == "call_function" and self.is_complex_dtype(node)

    def subgraph_from_anchor(self, anchor_node: Node) -> ComplexSubGraphInfo:
        subgraph_nodes: Set[Node] = set()
        input_nodes: Set[Node] = set()
        stack = [anchor_node]
        while stack:
            n = stack.pop()
            if n in subgraph_nodes:
                continue
            subgraph_nodes.add(n)
            for inp in n.all_input_nodes:
                if self.node_include_in_subgraph(inp):
                    print("node inp is added to stack:", inp.name)
                    stack.append(inp)
                else:
                    print("node inp is not added to stack BUT INP:", inp.name)
                    input_nodes.add(inp)
        return ComplexSubGraphInfo(
            [anchor_node], list(subgraph_nodes), list(input_nodes)
        )

    def find_complex_op_subgraphs(
        self, gm: GraphModule, anchor_target: str
    ) -> List[ComplexSubGraphInfo]:
        complex_op_subgraphs: List[ComplexSubGraphInfo] = []
        for node in gm.graph.nodes:
            if node.target == anchor_target:
                print("node.target:===", node.target, "node.name:", node.name)
                new_sub = self.subgraph_from_anchor(node)
                # new_nodes = set(sub.subgraph_nodes) - seen
                # if any intersecting nodes between seen and sub.subgraph_nodes they should be merged
                merged = False
                for existing_sub in complex_op_subgraphs:
                    if set(existing_sub.subgraph_nodes) & set(new_sub.subgraph_nodes):
                        print("merging subgraphs:=======", existing_sub, new_sub)
                        # merge the two subgraphs
                        existing_sub.subgraph_nodes = list(
                            set(existing_sub.subgraph_nodes)
                            | set(new_sub.subgraph_nodes)
                        )
                        existing_sub.input_nodes = list(
                            set(existing_sub.input_nodes) | set(new_sub.input_nodes)
                        )
                        existing_sub.anchor_nodes = list(
                            set(existing_sub.anchor_nodes) | set(new_sub.anchor_nodes)
                        )
                        merged = True
                        break
                if not merged:
                    complex_op_subgraphs.append(new_sub)
        return complex_op_subgraphs


def find_complex_indices(torch_inputs: List[torch.Tensor]) -> List[int]:
    complex_idxs = []
    for i, t in enumerate(torch_inputs):
        if isinstance(t, torch.Tensor) and t.dtype in {
            torch.complex64,
            torch.complex128,
        }:
            complex_idxs.append(i)
    return complex_idxs


# complex_graph_rewrite.py


def replace_complex_input_nodes(
    graph_module: GraphModule,
    subgraphs: List[ComplexSubGraphInfo],
    truncate_double: bool = False,
):
    # iterate placeholders and adjust shapes/dtypes
    for sub in subgraphs:
        for inp in sub.input_nodes:
            if inp.op == "placeholder":
                # modify metadata: expand last dim by 2, change dtype
                shape = list(inp.meta["tensor_meta"].shape)
                shape[-1] *= 2
                new_dtype = torch.float32 if truncate_double else None
                # apply to placeholder
                inp.meta["tensor_meta"].shape = tuple(shape)
                if new_dtype:
                    inp.meta["tensor_meta"].dtype = new_dtype


def complex_graph_rewrite(
    graph_module: GraphModule, subgraphs: List[ComplexSubGraphInfo]
):
    # rewrite reshape and slice nodes and replace mul with custom complex mul
    for sub in subgraphs:
        for n in sub.subgraph_nodes:
            if n.target == torch.reshape:
                # append 2 in shape args
                args = list(n.args)
                shape = list(args[1])
                shape.append(2)
                args[1] = tuple(shape)
                n.args = tuple(args)
            elif n.target == torch.ops.aten.mul:
                # mark for custom complex mul in TRT
                n.target = torch.ops.my_ops.complex_mul
                # adjust args: merge real/imag pairs
                # ...
    return graph_module
