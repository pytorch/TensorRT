from typing import Optional

from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion import impl
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.converter_utils import get_positive_dim


def cross(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input: TRTTensor,
    other: TRTTensor,
    dim: int,
) -> TRTTensor:
    # aten.linalg_cross requires input.shape[dim] == other.shape[dim] == 3.
    # Split each operand into its 3 components along dim with ISelect
    # (gather), combine with the standard cross-product formula, then
    # concatenate the 3 result components back along dim.
    def component(t: TRTTensor, index: int, prefix: str) -> TRTTensor:
        t_dim = get_positive_dim(dim, len(t.shape))
        return impl.select.select(
            ctx, target, source_ir, f"{name}_{prefix}{index}", t, t_dim, index
        )

    a0, a1, a2 = (component(input, i, "a") for i in range(3))
    b0, b1, b2 = (component(other, i, "b") for i in range(3))

    def mul(lhs: TRTTensor, rhs: TRTTensor, label: str) -> TRTTensor:
        return impl.elementwise.mul(ctx, target, source_ir, f"{name}_{label}", lhs, rhs)

    def sub(lhs: TRTTensor, rhs: TRTTensor, label: str) -> TRTTensor:
        return impl.elementwise.sub(ctx, target, source_ir, f"{name}_{label}", lhs, rhs)

    r0 = sub(mul(a1, b2, "a1b2"), mul(a2, b1, "a2b1"), "r0")
    r1 = sub(mul(a2, b0, "a2b0"), mul(a0, b2, "a0b2"), "r1")
    r2 = sub(mul(a0, b1, "a0b1"), mul(a1, b0, "a1b0"), "r2")

    out_dim = get_positive_dim(dim, max(len(input.shape), len(other.shape)))
    results = [
        impl.unsqueeze.unsqueeze(
            ctx, target, source_ir, f"{name}_unsqueeze_r{i}", r, out_dim
        )
        for i, r in enumerate((r0, r1, r2))
    ]

    return impl.cat.cat(ctx, target, source_ir, f"{name}_cat", results, out_dim)
