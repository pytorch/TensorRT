# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import operator
from typing import Collection, Dict, List, Optional

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.inline_autocast_regions import (
    _may_read_storage,
)
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
    iter_cond_subgraphs,
)
from torch_tensorrt.dynamo.lowering.passes.reset_folded_constructors import (
    _aliases_input,
)

logger = logging.getLogger(__name__)

aten = torch.ops.aten

_LOW_PRECISION = (torch.float16, torch.bfloat16)
_FLOATING = (*_LOW_PRECISION, torch.float32)
_POINTWISE = {
    aten.abs.default,
    aten.add.Scalar,
    aten.add.Tensor,
    aten.clamp.default,
    aten.cos.default,
    aten.div.Scalar,
    aten.div.Tensor,
    aten.erf.default,
    aten.exp.default,
    aten.gelu.default,
    aten.log.default,
    aten.maximum.default,
    aten.minimum.default,
    aten.mul.Scalar,
    aten.mul.Tensor,
    aten.neg.default,
    aten.pow.Tensor_Scalar,
    aten.pow.Tensor_Tensor,
    aten.relu.default,
    aten.sigmoid.default,
    aten.sin.default,
    aten.sqrt.default,
    aten.sub.Scalar,
    aten.sub.Tensor,
    aten.tanh.default,
    aten.where.self,
}
_REDUCTIONS = {
    aten.amax.default,
    aten.amin.default,
    aten.mean.default,
    aten.mean.dim,
    aten.sum.default,
    aten.sum.dim_IntList,
}
# Rounding the float32 result of these ops back gives their 16-bit result.
_EXACT = {
    aten.abs.default,
    aten.amax.default,
    aten.amin.default,
    aten.clamp.default,
    aten.maximum.default,
    aten.minimum.default,
    aten.neg.default,
    aten.relu.default,
    aten.where.self,
}
_VIEWS = {
    aten._reshape_copy.default,
    aten.cat.default,
    aten.clone.default,
    aten.expand.default,
    aten.permute.default,
    aten.reshape.default,
    aten.select.int,
    aten.slice.Tensor,
    aten.squeeze.dim,
    aten.squeeze.dims,
    aten.unsqueeze.default,
    aten.view.default,
}


def _dtype(node: torch.fx.Node) -> Optional[torch.dtype]:
    val = node.meta.get("val")
    return val.dtype if isinstance(val, torch.Tensor) else None


def _storage_is_read(node: torch.fx.Node) -> bool:
    """Whether an op, such as as_strided, reads the memory around the elements of
    node, directly or through views of it, including the pieces of a split."""
    pending, seen = list(node.users), set()
    while pending:
        user = pending.pop()
        if user in seen:
            continue
        seen.add(user)
        if _may_read_storage(user):
            return True
        if user.target is operator.getitem or _aliases_input(user):
            pending.extend(user.users)
    return False


def _may_join(node: torch.fx.Node) -> bool:
    """Whether node is an op that a region can hold, whatever its inputs."""
    if node.op != "call_function" or _dtype(node) not in _FLOATING:
        return False
    # An earlier pass may add a node without meta, and then its dtype is unknown.
    if any("val" not in arg.meta for arg in node.all_input_nodes):
        return False
    if node.target == aten._to_copy.default:
        source = node.args[0].meta["val"]
        # type_as and to(tensor) name the device and layout the tensor already has.
        return (
            _dtype(node.args[0]) in _FLOATING
            and set(node.kwargs) <= {"dtype", "device", "layout", "memory_format"}
            and node.kwargs.get("device", source.device) == source.device
            and node.kwargs.get("layout", source.layout) == source.layout
        )
    # An op with its own output dtype keeps that dtype, so it stays out.
    if "dtype" in node.kwargs:
        return False
    return (
        node.target in _VIEWS or node.target in _POINTWISE or node.target in _REDUCTIONS
    )


def _feeds_region_math(node: torch.fx.Node) -> bool:
    """Whether node has uses, and every use, followed through views and casts, is a
    pointwise op or a reduction that can join a region."""
    pending, seen = list(node.users), set()
    if not pending:
        return False
    while pending:
        user = pending.pop()
        if user in seen:
            continue
        seen.add(user)
        if not _may_join(user):
            return False
        if user.target in _VIEWS or user.target == aten._to_copy.default:
            if not user.users:
                return False
            pending.extend(user.users)
    return True


def _in_region(node: torch.fx.Node, members: Collection[torch.fx.Node]) -> bool:
    if not _may_join(node):
        return False
    # A cast joins only when its result goes on into region math, so a value cast for
    # a matrix multiply is rounded to its 16-bit dtype first, as without the setting.
    if node.target == aten._to_copy.default:
        source = node.args[0]
        # It rounds a value from outside to 16 bits on purpose, as fake quantization does.
        if (
            source not in members
            and _dtype(node) in _LOW_PRECISION
            and _dtype(node) != _dtype(source)
        ):
            return False
        return _feeds_region_math(node)
    if node.target in _VIEWS:
        # The cast back gives new memory, so a view whose memory is read stays out.
        if _storage_is_read(node):
            return False
        # Like a cast, so a value is cast before it is expanded, on the smaller tensor.
        if node.target == aten.expand.default:
            return _feeds_region_math(node)
        tensors = [arg for arg in node.all_input_nodes if _dtype(arg) is not None]
        if all(arg in members for arg in tensors):
            return True
        # An outside input is cast in full, so it joins only when the result goes back
        # into float32 math anyway. A cache that a concatenation extends for a matrix
        # multiply stays out.
        return any(arg in members for arg in tensors) and _feeds_region_math(node)
    return True


def _regions(gm: torch.fx.GraphModule) -> List[List[torch.fx.Node]]:
    """The connected groups of region ops, each in graph order."""
    root: Dict[torch.fx.Node, torch.fx.Node] = {}
    for node in gm.graph.nodes:
        if _in_region(node, root):
            root[node] = node

    def find(node: torch.fx.Node) -> torch.fx.Node:
        while root[node] is not node:
            root[node] = root[root[node]]
            node = root[node]
        return node

    for node in root:
        for arg in node.all_input_nodes:
            if arg in root:
                root[find(arg)] = find(node)
    regions: Dict[torch.fx.Node, List[torch.fx.Node]] = {}
    for node in root:
        regions.setdefault(find(node), []).append(node)
    return list(regions.values())


def _cast(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    dtype: torch.dtype,
    owner: torch.fx.Node,
) -> torch.fx.Node:
    with gm.graph.inserting_after(node):
        cast = gm.graph.call_function(aten._to_copy.default, (node,), {"dtype": dtype})
    # Take the meta of the region node the cast serves, for its nn_module_stack, which
    # engine layer names read.
    cast.meta.update({k: v for k, v in owner.meta.items() if k != "tensor_meta"})
    cast.meta["val"] = node.meta["val"].to(dtype)
    return cast


def pointwise_in_float32(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Compute each float16 or bfloat16 region of pointwise ops in float32.

    TensorRT builds a strongly typed network, so each 16-bit op rounds its own
    result, and the error grows with the length of a chain of such ops. A region is
    a connected group of pointwise, reduction, view and cast ops that a 16-bit value
    enters from outside and that computes at least one 16-bit value with an op whose
    result can change in float32, unlike a view, a cast that keeps its dtype, or an
    op such as relu or maximum. A cast or an expand joins only when its result goes
    on only into region math, so a value is cast before it is expanded. A cast of a
    value from outside to another 16-bit dtype never joins, so it keeps its
    rounding. Another view or a concatenation joins when its inputs are in the
    region, or when one is and its result goes on only into region math. No view
    joins when an op such as as_strided reads the memory around it. Any other op,
    such as a matrix multiply, an attention or a softmax, ends a region. Each 16-bit
    value that enters a region is cast to float32 once, casts inside it become
    float32, and each value that leaves it is cast back to its own dtype. This is
    how PyTorch's Inductor computes inside one fused kernel, but a region can be
    larger: residual adds join the ops around a matrix multiply into one region, so
    a residual stream stays in float32 from layer to layer.

    It runs on each graph that becomes a TensorRT engine, including the branches of
    a torch.cond in it, and never on code that runs in PyTorch.
    """
    if not settings.pointwise_in_float32:
        return gm
    count = 0
    inputs: Dict[torch.fx.Node, torch.fx.Node] = {}
    for region in _regions(gm):
        members = set(region)
        entering = {
            arg
            for n in region
            for arg in n.all_input_nodes
            if arg not in members and _dtype(arg) in _LOW_PRECISION
        }
        # A region that no 16-bit value enters computes from float32 values, so its
        # casts to 16 bits are rounding the model asks for, as in fake quantization.
        # Another region already computes in float32 or would compute the same
        # values, so new casts would only add work.
        if not entering or not any(
            _dtype(n) in _LOW_PRECISION
            and n.target not in _VIEWS
            and n.target not in _EXACT
            and (n.target != aten._to_copy.default or _dtype(n.args[0]) != _dtype(n))
            for n in region
        ):
            continue
        dtypes = {n: _dtype(n) for n in region}
        for node in region:
            for arg in node.all_input_nodes:
                if arg in entering:
                    if arg not in inputs:
                        inputs[arg] = _cast(gm, arg, torch.float32, node)
                    node.replace_input_with(arg, inputs[arg])
                    # A float32 cast of the shared cast would cast it again.
                    if node.target == aten._to_copy.default:
                        node.replace_all_uses_with(inputs[arg])
            if node.target == aten._to_copy.default:
                node.kwargs = {**node.kwargs, "dtype": torch.float32}
            node.meta["val"] = node.meta["val"].to(torch.float32)
            node.meta.pop("tensor_meta", None)
        for node in region:
            outside = [u for u in node.users if u not in members]
            if dtypes[node] in _LOW_PRECISION and outside:
                output = _cast(gm, node, dtypes[node], node)
                for user in outside:
                    user.replace_input_with(node, output)
        count += len(region)
    if count:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(f"Computing {count} node(s) of 16-bit regions in float32")
    # The branches of a torch.cond are converted into the same engine.
    for branch in dict.fromkeys(iter_cond_subgraphs(gm)):
        pointwise_in_float32(branch, settings)
    return gm
