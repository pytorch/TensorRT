import torch
from torch_tensorrt.fx.tracer.acc_tracer import acc_ops


def check_permute(node: torch.fx.Node) -> bool:
    ranks = len(node.meta["tensor_meta"].shape)
    permutation = [i % ranks for i in node.kwargs["permutation"]]
    allowed_permutation = list(range(ranks))
    allowed_permutation[-1] = ranks - 2
    allowed_permutation[-2] = ranks - 1
    return permutation == allowed_permutation


def trt_transposed_matmul(
    lhs: torch.Tensor, rhs: torch.Tensor, lhs_transposed: bool, rhs_transposed: bool
) -> torch.Tensor:
    if lhs_transposed:
        lhs = lhs.transpose(-1, -2)
    if rhs_transposed:
        rhs = rhs.transpose(-1, -2)
    return torch.matmul(lhs, rhs)


def fuse_permute_matmul(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Fuse pattern like permute + matmul if permute is transposing the last two dimension.
    """
    for node in gm.graph.nodes:
        if node.target == acc_ops.matmul:
            lhs, rhs = node.kwargs["input"], node.kwargs["other"]
            lhs_transposed = rhs_tranposed = False
            skip = False

            if lhs.target == acc_ops.permute and check_permute(lhs):
                lhs_transposed = True
                lhs = lhs.kwargs["input"]

            if rhs.target == acc_ops.permute and check_permute(rhs):
                rhs_tranposed = True
                rhs = rhs.kwargs["input"]

            if (not skip) and (lhs_transposed or rhs_tranposed):
                with gm.graph.inserting_before(node):
                    fused_node = gm.graph.call_function(
                        trt_transposed_matmul,
                        args=(lhs, rhs, lhs_transposed, rhs_tranposed),
                    )
                node.replace_all_uses_with(fused_node)

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm


def trt_transposed_linear(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    return torch.matmul(input.transpose(-1, -2), weight.t()) + bias


def fuse_permute_linear(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Fuse pattern like permute + linear if permute is transposing the last two dimension.
    """
    for node in gm.graph.nodes:
        if node.target == acc_ops.linear:
            inp = node.kwargs["input"]
            if inp.target == acc_ops.permute and check_permute(inp):
                inp = inp.kwargs["input"]
                weight = node.kwargs["weight"]
                bias = node.kwargs["bias"]
                with gm.graph.inserting_before(node):
                    fused_node = gm.graph.call_function(
                        trt_transposed_linear, args=(inp, weight, bias)
                    )
                    node.replace_all_uses_with(fused_node)

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm
