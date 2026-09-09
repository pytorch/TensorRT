from typing import Optional, Union

import numpy as np
import tensorrt as trt
import torch
from torch.fx.node import Target
from torch_tensorrt import _enums
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import (
    cast_trt_tensor,
    get_trt_tensor,
)


def histc(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: trt.ITensor,
    bins: int,
    min: float,
    max: float,
) -> trt.ITensor:
    # aten returns the input dtype, so integer ids in means integer counts out.
    out_dtype = _enums.dtype._from(input.dtype).to(torch.dtype)

    flat = impl.shuffle.reshape(ctx, target, source_ir, f"{name}_flat", input, (-1, 1))
    x = cast_trt_tensor(ctx, flat, torch.float32, f"{name}_f32", target, source_ir)

    # bin = floor((x - min) / (max - min) * bins), with x == max folded back into
    # the last bin instead of overflowing to index `bins`.
    shifted = impl.elementwise.sub(ctx, target, source_ir, f"{name}_shift", x, min)
    scaled = impl.elementwise.mul(
        ctx, target, source_ir, f"{name}_scale", shifted, bins / (max - min)
    )
    idx = impl.unary.floor(ctx, target, source_ir, f"{name}_floor", scaled)
    idx = impl.elementwise.min(
        ctx, target, source_ir, f"{name}_lastbin", idx, float(bins - 1)
    )

    # Values outside [min, max] are dropped, not clamped.
    in_range = impl.elementwise.logical_and(
        ctx,
        target,
        source_ir,
        f"{name}_inrange",
        impl.elementwise.ge(ctx, target, source_ir, f"{name}_ge_min", x, min),
        impl.elementwise.le(ctx, target, source_ir, f"{name}_le_max", x, max),
    )

    edges = get_trt_tensor(
        ctx, np.arange(bins, dtype=np.float32).reshape(1, bins), f"{name}_edges"
    )
    hit = impl.elementwise.eq(ctx, target, source_ir, f"{name}_hit", idx, edges)
    hit = impl.elementwise.logical_and(
        ctx, target, source_ir, f"{name}_hit_valid", hit, in_range
    )

    counts = impl.reduce.sum(
        ctx,
        target,
        source_ir,
        f"{name}_count",
        cast_trt_tensor(ctx, hit, torch.float32, f"{name}_hit_f32", target, source_ir),
        dim=0,
        keepdim=False,
    )
    return cast_trt_tensor(ctx, counts, out_dtype, f"{name}_out", target, source_ir)


def grouped_mm(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    mat1: trt.ITensor,
    mat2: trt.ITensor,
    offs: trt.ITensor,
    out_dtype: Optional[Union[torch.dtype, trt.DataType]] = None,
) -> trt.ITensor:
    num_rows = int(mat1.shape[0])
    num_experts = int(mat2.shape[0])
    compute_dtype = _enums.dtype._from(mat2.dtype).to(torch.dtype)

    # Rows arrive sorted by group and `offs` holds each group's exclusive end,
    # so the group owning row i is how many offsets are <= i.
    rows = get_trt_tensor(
        ctx, np.arange(num_rows, dtype=np.int32).reshape(num_rows, 1), f"{name}_rows"
    )
    offs_i32 = cast_trt_tensor(
        ctx, offs, torch.int32, f"{name}_offs_i32", target, source_ir
    )
    offs_row = impl.shuffle.reshape(
        ctx, target, source_ir, f"{name}_offs_row", offs_i32, (1, -1)
    )
    passed = impl.elementwise.le(ctx, target, source_ir, f"{name}_passed", offs_row, rows)
    group = impl.reduce.sum(
        ctx,
        target,
        source_ir,
        f"{name}_group",
        cast_trt_tensor(
            ctx, passed, torch.int32, f"{name}_passed_i32", target, source_ir
        ),
        dim=1,
        keepdim=False,
    )

    # Dense lowering: every row through every expert, then mask.
    lhs = impl.shuffle.reshape(
        ctx, target, source_ir, f"{name}_lhs", mat1, (1, num_rows, -1)
    )
    per_expert = impl.matmul.matrix_multiply(
        ctx, target, source_ir, f"{name}_mm", lhs, mat2
    )

    expert_ids = get_trt_tensor(
        ctx,
        np.arange(num_experts, dtype=np.int32).reshape(num_experts, 1),
        f"{name}_expert_ids",
    )
    group_row = impl.shuffle.reshape(
        ctx, target, source_ir, f"{name}_group_row", group, (1, num_rows)
    )
    selected = impl.elementwise.eq(
        ctx, target, source_ir, f"{name}_selected", expert_ids, group_row
    )
    selected = impl.shuffle.reshape(
        ctx,
        target,
        source_ir,
        f"{name}_selected_3d",
        cast_trt_tensor(
            ctx, selected, compute_dtype, f"{name}_selected_cast", target, source_ir
        ),
        (num_experts, num_rows, 1),
    )

    masked = impl.elementwise.mul(
        ctx, target, source_ir, f"{name}_masked", per_expert, selected
    )
    out = impl.reduce.sum(
        ctx, target, source_ir, f"{name}_reduce", masked, dim=0, keepdim=False
    )

    if out_dtype is not None:
        out = cast_trt_tensor(
            ctx, out, out_dtype, f"{name}_out_dtype", target, source_ir
        )
    return out
