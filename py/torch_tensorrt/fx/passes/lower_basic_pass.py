import copy
import operator
import warnings
from typing import Any

import torch
import torch.fx
from torch.fx.experimental.const_fold import split_const_subgraphs

from ..observer import observable

from ..tracer.acc_tracer import acc_ops
from ..tracer.acc_tracer.acc_utils import get_attr
from .pass_utils import log_before_after, validate_inference

# Create an alias for module input type to avoid littering pyre-ignore for Any
# throughout the file.
Input = Any


def run_const_fold(traced_mod: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # Now we do constant folding on traced module. We want to skip pattern like
    # weights -> quant -> dequant -> op during constant folding when the model is
    # a quantized int8 model.
    def skip_folding_quant_dequant(node: torch.fx.Node):
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


@log_before_after
@validate_inference(atol=1e-3, rtol=1e-2)
def fuse_sparse_matmul_add(gm: torch.fx.GraphModule, input: Input):
    """
    Replace acc_ops.matmul + acc_ops.add with acc_ops.linear
    TRT8.2 can take advantage of structured sparsity (2:4), but the graph needs contain a single FC layer.
    Later versions of TRT should work with matmul.

    Example before:
    def forward(self, x):
        a = self.a
        b = self.b
        addmm_mm = torch_tensorrt.fx.tracer.acc_tracer.acc_ops.matmul(input = a, other = b);  a = b = None
        addmm_add = torch_tensorrt.fx.tracer.acc_tracer.acc_ops.add(input = addmm_mm, other = x);  addmm_mm = x = None
        return addmm_add

    After:
    def forward(self, x):
        a = self.a
        b = self.b
        linear_1 = torch_tensorrt.fx.tracer.acc_tracer.acc_ops.linear(input = a, weight = b, bias = x);  a = b = x = None
        return linear_1
    """
    counter = 0
    for node in gm.graph.nodes:
        if node.target != acc_ops.add:
            continue
        add_node = node
        bias = add_node.kwargs["other"]

        if bias.op != "get_attr":
            continue
        # test that bias tensor is one-dimensional, should correspond to shape (out_features)
        if get_attr(bias).dim() > 1:
            continue

        node = add_node.kwargs["input"]
        if node.target != acc_ops.matmul:
            continue
        matmul_node = node
        a = matmul_node.kwargs["input"]

        node = matmul_node.kwargs["other"]
        if node.op != "get_attr":
            continue

        get_attr_node = node
        weight = get_attr(get_attr_node)
        # TODO: verify that weight comply with TRT structured sparsity requirements:
        # For each output channel and for each spatial pixel in the kernel weights,
        # every 4 input channels must have at least 2 zeros.

        # test that weight tensor is two-dimensional, should correspond to shape (out_features, in_features)
        if weight.dim() != 2:
            continue

        weight_t = weight.transpose(0, 1)
        weight_t_name = "weight_t_tensor_" + str(counter)
        gm.register_buffer(weight_t_name, weight_t)
        counter += 1

        with gm.graph.inserting_before(add_node):
            weight_t_attr = gm.graph.get_attr(weight_t_name)
            fused_node = gm.graph.call_function(
                acc_ops.linear,
                kwargs={"input": a, "weight": weight_t_attr, "bias": bias},
            )
        add_node.replace_all_uses_with(fused_node)

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm


def trt_transposed_matmul(
    lhs: torch.Tensor, rhs: torch.Tensor, lhs_transposed: bool, rhs_transposed: bool
):
    if lhs_transposed:
        lhs = lhs.transpose(-1, -2)
    if rhs_transposed:
        rhs = rhs.transpose(-1, -2)
    return torch.matmul(lhs, rhs)


def trt_transposed_linear(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
):
    return torch.matmul(input.transpose(-1, -2), weight.t()) + bias


def check_permute(node: torch.fx.Node):
    ranks = len(node.meta["tensor_meta"].shape)
    permutation = list(i % ranks for i in node.kwargs["permutation"])  # type: ignore[union-attr]
    allowed_permutation = list(i for i in range(ranks))
    allowed_permutation[-1] = ranks - 2
    allowed_permutation[-2] = ranks - 1
    return permutation == allowed_permutation


@observable()
@log_before_after
@validate_inference(atol=1e-3, rtol=1e-2)
def fuse_permute_linear(gm: torch.fx.GraphModule, input: Input):
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


@observable()
@log_before_after
@validate_inference(atol=1e-3, rtol=1e-2)
def fuse_permute_matmul(gm: torch.fx.GraphModule, input: Input):
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


try:
    # @manual=//deeplearning/trt/python:py_tensorrt
    import tensorrt as trt
    from torch_tensorrt.fx.converter_registry import tensorrt_converter
    from torch_tensorrt.fx.converters.converter_utils import (
        add_binary_elementwise_layer,
        broadcast,
        get_trt_tensor,
        set_layer_name,
    )
except Exception as e:
    warnings.warn(f"Unable to import TensorRT related libraries.: {e}")
else:

    @tensorrt_converter(trt_transposed_matmul)
    def trt_transposed_matmul_converter(network, target, args, kwargs, name):
        lhs, rhs, lhs_transposed, rhs_transposed = args

        if isinstance(lhs, torch.nn.Parameter):
            lhs = get_trt_tensor(network, lhs, f"{name}_lhs")
        if isinstance(rhs, torch.nn.Parameter):
            rhs = get_trt_tensor(network, rhs, f"{name}_rhs")
        layer = network.add_matrix_multiply(
            lhs,
            trt.MatrixOperation.TRANSPOSE
            if lhs_transposed
            else trt.MatrixOperation.NONE,
            rhs,
            trt.MatrixOperation.TRANSPOSE
            if rhs_transposed
            else trt.MatrixOperation.NONE,
        )
        set_layer_name(layer, target, name)
        return layer.get_output(0)

    @tensorrt_converter(trt_transposed_linear)
    def trt_transposed_linear_converter(network, target, args, kwargs, name):
        input, weight, bias = args

        weight = get_trt_tensor(network, weight.t(), f"{name}_weight")
        bias = get_trt_tensor(network, bias.reshape(1, -1), f"{name}_bias")

        input, weight = broadcast(
            network,
            input,
            weight,
            f"{input.name}_broadcast",
            f"{weight.name}_broadcast",
        )
        layer = network.add_matrix_multiply(
            input,
            trt.MatrixOperation.TRANSPOSE,
            weight,
            trt.MatrixOperation.NONE,
        )
        set_layer_name(layer, target, f"{name}_mm")
        return add_binary_elementwise_layer(
            network,
            layer.get_output(0),
            bias,
            trt.ElementWiseOperation.SUM,
            target,
            f"{name}_add",
        )


def slice_list(sli: slice, dim: int, size: int):
    slice_all = slice(None, None, None)
    if size == 1:
        return [sli]
    elif size == 2:
        if dim == 0:
            return [sli, slice_all]
        elif dim == 1:
            return [slice_all, sli]
    elif size == 3:
        if dim == 0:
            return [sli, slice_all, slice_all]
        elif dim == 1:
            return [slice_all, sli, slice_all]
        elif dim == 2:
            return [slice_all, slice_all, sli]
    elif size == 4:
        if dim == 0:
            return [sli, slice_all, slice_all, slice_all]
        elif dim == 1:
            return [slice_all, sli, slice_all, slice_all]
        elif dim == 2:
            return [slice_all, slice_all, sli, slice_all]
        elif dim == 3:
            return [slice_all, slice_all, slice_all, sli]


def split_across(
    gm: torch.fx.GraphModule, sli: slice, input_node: torch.fx.Node, dim: int, size: int
):
    start_node = end_node = mid_node = None
    if sli.start is None and sli.stop is None:
        return (start_node, input_node, end_node)
    if sli.start is not None:
        st_sli = slice(0, sli.start, None)
        slice_list_gen = slice_list(st_sli, dim, size)
        start_node = gm.graph.call_function(
            operator.getitem, args=(input_node, slice_list_gen)
        )
    if sli.stop is not None:
        end_sli = slice(sli.stop, None, None)
        slice_list_gen = slice_list(end_sli, dim, size)
        end_node = gm.graph.call_function(
            operator.getitem, args=(input_node, slice_list_gen)
        )
    if dim != size - 1:
        mid_sli = slice(sli.start, sli.stop, None)
        slice_list_gen = slice_list(mid_sli, dim, size)
        mid_node = gm.graph.call_function(
            operator.getitem, args=(input_node, slice_list_gen)
        )
    return (start_node, mid_node, end_node)


def list_gen(
    start_node: torch.fx.Node,
    end_node: torch.fx.Node,
    input_node: torch.fx.Node,
    gm: torch.fx.GraphModule,
    dim: int,
):
    if start_node:
        if end_node:
            concat_list = [start_node, input_node, end_node]
        else:
            concat_list = [start_node, input_node]
    else:
        if end_node:
            concat_list = [input_node, end_node]
        else:
            concat_list = [input_node]
    if len(concat_list) > 1:
        concat_node = gm.graph.call_function(torch.cat, args=(concat_list, dim))
    else:
        concat_node = concat_list[0]
    return concat_node


def transform_setitem(gm: torch.fx.GraphModule, input: Input):
    """
    Setitem is not tracable in fx and acc tracer but is available in dynamo trace. This pass works for dynamo trace only.
    The implementation decompose the setitem into a few getitem op and assembly together again through concat.
    The major reason is that TRT does not support in-place copy and memory reference.
    """
    map_replace = {}
    for node in gm.graph.nodes:
        for old_node in map_replace:
            node.replace_input_with(old_node, map_replace[old_node])

        if node.target == operator.setitem:
            input_node = node.args[0]
            sli = node.args[1]
            inp = node.args[2]

            inp_flag = False
            if type(inp) == torch.fx.node.Node and inp.target == operator.getitem:
                new_args = list(copy.deepcopy(inp.args[1]))
                for ind, val in enumerate(new_args):
                    if type(val) == int:
                        inp_flag = True
                        if val == -1:
                            new_args[ind] = slice(-1, None, None)
                        else:
                            new_args[ind] = slice(val, val + 1, None)

                if inp_flag:
                    with gm.graph.inserting_before(inp):
                        new_node = gm.graph.call_function(
                            operator.getitem, args=(inp.args[0], new_args)
                        )
                        inp.replace_all_uses_with(new_node)
                    inp = new_node

            if type(sli) is not tuple:
                sli = [sli]

            tmp_sli = []
            for x in sli:
                if type(x) == int:
                    if x == -1:
                        tmp_sli.append(slice(-1, None, None))
                    else:
                        tmp_sli.append(slice(x, x + 1, None))
                else:
                    tmp_sli.append(x)
            sli = tmp_sli

            dimension = len(sli)
            with gm.graph.inserting_before(node):
                if dimension == 1:
                    start_node_0, _, end_node_0 = split_across(
                        gm, sli[0], input_node, dim=0, size=1
                    )
                    concat_node_0 = list_gen(start_node_0, end_node_0, inp, gm, 0)
                elif dimension == 2:
                    start_node_0, mid_node_0, end_node_0 = split_across(
                        gm, sli[0], input_node, dim=0, size=2
                    )
                    start_node_1, _, end_node_1 = split_across(
                        gm, sli[1], mid_node_0, dim=1, size=2
                    )
                    concat_node_1 = list_gen(start_node_1, end_node_1, inp, gm, 1)
                    concat_node_0 = list_gen(
                        start_node_0, end_node_0, concat_node_1, gm, 0
                    )
                elif dimension == 3:
                    start_node_0, mid_node_0, end_node_0 = split_across(
                        gm, sli[0], input_node, dim=0, size=3
                    )
                    start_node_1, mid_node_1, end_node_1 = split_across(
                        gm, sli[1], mid_node_0, dim=1, size=3
                    )
                    start_node_2, _, end_node_2 = split_across(
                        gm, sli[2], mid_node_1, dim=2, size=3
                    )
                    concat_node_2 = list_gen(start_node_2, end_node_2, inp, gm, 2)
                    concat_node_1 = list_gen(
                        start_node_1, end_node_1, concat_node_2, gm, 1
                    )
                    concat_node_0 = list_gen(
                        start_node_0, end_node_0, concat_node_1, gm, 0
                    )
                elif dimension == 4:
                    start_node_0, mid_node_0, end_node_0 = split_across(
                        gm, sli[0], input_node, dim=0, size=4
                    )
                    start_node_1, mid_node_1, end_node_1 = split_across(
                        gm, sli[1], mid_node_0, dim=1, size=4
                    )
                    start_node_2, mid_node_2, end_node_2 = split_across(
                        gm, sli[2], mid_node_1, dim=2, size=4
                    )
                    start_node_3, _, end_node_3 = split_across(
                        gm, sli[3], mid_node_2, dim=3, size=4
                    )
                    concat_node_3 = list_gen(start_node_3, end_node_3, inp, gm, 3)
                    concat_node_2 = list_gen(
                        start_node_2, end_node_2, concat_node_3, gm, 2
                    )
                    concat_node_1 = list_gen(
                        start_node_1, end_node_1, concat_node_2, gm, 1
                    )
                    concat_node_0 = list_gen(
                        start_node_0, end_node_0, concat_node_1, gm, 0
                    )
                else:
                    warnings.warn(f"setitem does not support dimension={dimension}")
                    continue
                node.replace_all_uses_with(concat_node_0)
                map_replace[input_node] = concat_node_0
            gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()
    return gm
