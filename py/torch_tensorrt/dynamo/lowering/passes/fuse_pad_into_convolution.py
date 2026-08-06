# mypy: disallow-untyped-decorators=False

"""Fold ``constant_pad_nd -> convolution`` into a single asymmetric-padded conv.

Torch-TensorRT's convolution converter only sets symmetric ``padding_nd``. When a
model applies padding itself (common for causal 3D convolutions that need
``(2 * pad_t, 0)`` in time), that pad stays a materialized copy. TensorRT's ONNX
parser folds the same pattern into ``pre_padding``/``post_padding``; this pass
recovers the equivalent for the Dynamo path.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)

# Torch-TRT routes conv1d through unsqueeze/squeeze; keep fusion on 2D/3D only.
_SUPPORTED_SPATIAL_RANKS = (2, 3)


@torch.library.custom_op("tensorrt::conv_asym_pad", mutates_args=())
def _conv_asym_pad_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: List[int],
    pre_padding: List[int],
    post_padding: List[int],
    dilation: List[int],
    groups: int,
) -> torch.Tensor:
    """Convolution with independent leading/trailing padding per spatial dim."""
    flat: List[int] = []
    for pre, post in zip(reversed(pre_padding), reversed(post_padding)):
        flat.extend((pre, post))
    padded = F.pad(x, flat)
    conv = {2: F.conv2d, 3: F.conv3d}[len(pre_padding)]
    return conv(padded, weight, bias, stride, 0, dilation, groups)


@_conv_asym_pad_impl.register_fake
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: List[int],
    pre_padding: List[int],
    post_padding: List[int],
    dilation: List[int],
    groups: int,
) -> torch.Tensor:
    del bias, groups
    spatial = []
    for index, (pre, post) in enumerate(zip(pre_padding, post_padding)):
        extent = x.shape[2 + index] + pre + post
        kernel = (weight.shape[2 + index] - 1) * dilation[index] + 1
        spatial.append((extent - kernel) // stride[index] + 1)
    return x.new_empty((x.shape[0], weight.shape[0], *spatial))


# Public alias used as the FX node target and converter key.
tensorrt_conv_asym_pad_op = torch.ops.tensorrt.conv_asym_pad.default


def _split_pad(
    pad: List[int], num_spatial: int
) -> Optional[Tuple[List[int], List[int]]]:
    """Convert ``constant_pad_nd`` amounts into per-spatial-dim pre/post lists.

    ``constant_pad_nd`` pads trailing dimensions first, so ``pad`` is ordered
    ``[last_pre, last_post, second_last_pre, ...]``. Returns ``None`` when the
    pad reaches past the spatial dims into channels or batch.
    """
    if len(pad) % 2 or len(pad) > 2 * num_spatial:
        return None
    pre = [0] * num_spatial
    post = [0] * num_spatial
    for index in range(len(pad) // 2):
        dim = num_spatial - 1 - index
        pre[dim] = pad[2 * index]
        post[dim] = pad[2 * index + 1]
    return pre, post


def fuse_pad_into_convolution(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Rewrite ``constant_pad_nd -> convolution`` into ``tensorrt::conv_asym_pad``."""
    del settings
    fused = 0

    for node in list(gm.graph.nodes):
        if (
            node.op != "call_function"
            or node.target != torch.ops.aten.convolution.default
        ):
            continue

        (
            source,
            weight,
            bias,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
        ) = node.args
        if transposed or any(output_padding):
            continue
        if len(padding) not in _SUPPORTED_SPATIAL_RANKS:
            continue

        pad_node = source
        if (
            not isinstance(pad_node, torch.fx.Node)
            or pad_node.target != torch.ops.aten.constant_pad_nd.default
            or len(pad_node.users) != 1
        ):
            continue

        pad = list(pad_node.args[1])
        fill = pad_node.args[2] if len(pad_node.args) > 2 else 0
        # Non-zero fill is not expressible as convolution padding; negatives are crops.
        if fill not in (0, 0.0) or any(amount < 0 for amount in pad):
            continue

        split = _split_pad(pad, len(padding))
        if split is None:
            continue
        pre, post = split
        pre = [amount + padding[index] for index, amount in enumerate(pre)]
        post = [amount + padding[index] for index, amount in enumerate(post)]

        with gm.graph.inserting_before(node):
            replacement = gm.graph.call_function(
                tensorrt_conv_asym_pad_op,
                args=(
                    pad_node.args[0],
                    weight,
                    bias,
                    list(stride),
                    pre,
                    post,
                    list(dilation),
                    groups,
                ),
            )
        replacement.meta.update(node.meta)
        node.replace_all_uses_with(replacement)
        gm.graph.erase_node(node)
        fused += 1

    if fused:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(f"Folded padding into {fused} convolution(s)")
    return gm
