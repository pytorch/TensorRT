from functools import lru_cache
from typing import Dict, List, Set

import torch
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher
from torch.ops import aten


class ConvBNReLU(torch.nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        stride: List[int],
        padding: List[int],
        dilation: List[int],
        transposed: bool,
        output_padding: List[int],
        groups: int,
        bn_weight: torch.Tensor,
        bn_bias: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        momentum: float,
        eps: float,
    ) -> torch.Tensor:
        x = aten.convolution.default(
            x,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )
        x = aten._native_batch_norm_legit_no_training.default(
            x, bn_weight, bn_bias, running_mean, running_var, momentum, eps
        )[0]
        x = aten.relu.default(x)
        return x


class ConvReLU(torch.nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        stride: List[int],
        padding: List[int],
        dilation: List[int],
        transposed: bool,
        output_padding: List[int],
        groups: int,
    ) -> torch.Tensor:
        x = aten.convolution.default(
            x,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )
        x = aten.relu.default(x)
        return x


class ConvGelu(torch.nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        stride: List[int],
        padding: List[int],
        dilation: List[int],
        transposed: bool,
        output_padding: List[int],
        groups: int,
    ) -> torch.Tensor:
        x = aten.convolution.default(
            x,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        )
        x = aten.gelu.default(x)
        return x


class ConvSilu(torch.nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        x = aten.convolution.default(
            x, weight, bias, [1, 1], [1, 1], [1, 1], False, [0, 0], 1
        )
        x = aten.silu.default(x)
        return x


class MulAdd(torch.nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        x = aten.mul.Tensor(x, weight)
        x = aten.add.Tensor(x, bias)
        return x


class MulMul(torch.nn.Module):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        x = aten.mul.Tensor(x, y)
        x = aten.mul.Tensor(x, z)
        return x


All_FUSION_PATTERNS = [
    ConvBNReLU,
    ConvReLU,
    ConvGelu,
    ConvSilu,
    MulAdd,
    MulMul,
]


@lru_cache(maxsize=None)
def get_node_in_fusion_pattern(
    graph: torch.fx.Graph,
) -> Dict[torch.fx.Node, Set[torch.fx.Node]]:
    """
    This function gets the nodes map of the fusion pattern from the graph.
    Key: node that appears in the fusion pattern
    Value: the list of nodes that should be fused together
    """
    fusion_nodes = {}
    for pattern in All_FUSION_PATTERNS:
        pattern_graph = torch.fx.symbolic_trace(pattern())
        subgraph_matcher = SubgraphMatcher(pattern_graph.graph)
        match_result = subgraph_matcher.match(graph)
        for match in match_result:
            fusion_group = {
                node
                for node in match.nodes_map.values()
                if node
                and type(node) == torch.fx.Node
                and node.op == "call_function"
                and node not in match.placeholder_nodes
            }
            for node in fusion_group:
                fusion_nodes[node] = fusion_group

    return fusion_nodes
