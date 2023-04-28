import copy
import logging
import operator
import warnings
from typing import Any, Optional

import torch
import torch.fx
import torch.fx as fx
from torch.fx.experimental.const_fold import split_const_subgraphs

#FIXME is this required
from torch_tensorrt.fx.tracer.acc_tracer import acc_ops


def replace_op_with_indices(module: torch.fx.GraphModule) -> torch.fx.GraphModule:
    for n in module.graph.nodes:
        if n.op == "call_function" and n.target in (
            torch.ops.aten.max_pool2d_with_indices.default,
            torch.ops.aten.max_pool3d_with_indices.default,
            torch.ops.aten.native_batch_norm.default,
        ):
            if len(n.users) != 1:
                raise RuntimeError(
                    f"{n.target} has users={len(n.users)}. We can only handle it with 1 user"
                )
            if n.target == torch.ops.aten.max_pool2d_with_indices.default:
                new_op = torch.ops.aten.max_pool2d
                new_args = n.args
            elif n.target == torch.ops.aten.max_pool3d_with_indices.default:
                new_op = torch.ops.aten.max_pool3d
                new_args = n.args
            elif n.target == torch.ops.aten.native_batch_norm.default:
                new_op = torch.ops.aten.batch_norm
                new_args = list(n.args)
                new_args.append(False)
                new_args = tuple(new_args)

            getitem_node = next(iter(n.users))
            with module.graph.inserting_after(getitem_node):
                new_node = module.graph.create_node(
                    "call_function",
                    new_op,
                    args=new_args,
                    kwargs=n.kwargs,
                )
                getitem_node.replace_all_uses_with(new_node)
                module.graph.erase_node(getitem_node)
    module.graph.eliminate_dead_code()
    module.recompile()
    return module


def run_const_fold(traced_mod: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # Now we do constant folding on traced module. We want to skip pattern like
    # weights -> quant -> dequant -> op during constant folding when the model is
    # a quantized int8 model.
    def skip_folding_quant_dequant(node: torch.fx.Node):
        #FIXME: is this condition required since these are dynamo converters
        if node.target != acc_ops.quantize_per_tensor:
            return False
        # If quantize_per_node -> dequantize, then skip folding.
        for user in node.users:
            if user.target == acc_ops.dequantize:
                return True
        return False

    const_split_mod = split_const_subgraphs(traced_mod, skip_folding_quant_dequant)
    const_split_mod.run_folding()
    return const_split_mod