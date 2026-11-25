# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Check a fused_op operator and name it so that a saved program can find it again."""

from typing import Any, Optional

import torch

from ._errors import RegionCaptureError


def check_operator(op: Any) -> torch._ops.OpOverload:
    """Return the overload a fused_op region calls, or explain why it cannot be used."""
    if isinstance(op, torch._library.custom_ops.CustomOpDef):
        op = op._opoverload
    if isinstance(op, torch._ops.OpOverloadPacket):
        try:
            op = op.default
        except AttributeError as error:
            raise TypeError(
                "fused_op requires an explicit custom-op overload"
            ) from error
    if not isinstance(op, torch._ops.OpOverload) or op.namespace in {
        "aten",
        "prim",
        "prims",
        "torch_tensorrt_region",
    }:
        raise TypeError("fused_op requires a registered torch custom operator")
    schema = op._schema
    if schema.is_mutable:
        raise RegionCaptureError("fused_op does not support a mutating operator schema")
    if any(
        argument.alias_info is not None
        for argument in (*schema.arguments, *schema.returns)
    ):
        raise RegionCaptureError(
            "fused_op does not support an aliasing operator schema"
        )
    output_types = [argument.type for argument in schema.returns]
    if len(output_types) == 1 and isinstance(output_types[0], torch.TupleType):
        output_types = output_types[0].elements()
    if (
        not output_types
        or any(argument.kwarg_only for argument in schema.arguments)
        or any(
            argument_type != torch._C.TensorType.get()
            for argument_type in [argument.type for argument in schema.arguments]
            + output_types
        )
    ):
        raise RegionCaptureError(
            "fused_op requires positional Tensor inputs and Tensor outputs"
        )
    return op


def operator_name(op: torch._ops.OpOverload) -> str:
    """``namespace::name.overload``, which is all a traced program records."""
    return f"{op._schema.name}.{op._schema.overload_name or 'default'}"


def find_operator(name: str) -> Optional[torch._ops.OpOverload]:
    """Look up an operator by the name a trace recorded, in this process."""
    namespace, _, rest = name.partition("::")
    op_name, _, overload = rest.rpartition(".")
    try:
        op = getattr(getattr(getattr(torch.ops, namespace), op_name), overload)
    except (AttributeError, RuntimeError):
        return None
    return op if isinstance(op, torch._ops.OpOverload) else None
