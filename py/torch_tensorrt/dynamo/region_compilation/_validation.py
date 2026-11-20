# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Tensor boundary and whole-program restrictions for explicit regions."""

from typing import Sequence

import torch
from torch.export.graph_signature import OutputKind
from torch_tensorrt.region._errors import RegionCaptureError


def validate_output_aliases(
    inputs: Sequence[torch.Tensor], outputs: Sequence[torch.Tensor], region_id: str
) -> None:
    for index, output in enumerate(outputs):
        for values, boundary in (
            (inputs, "region inputs"),
            (outputs[:index], "each other"),
        ):
            if any(torch._C._is_alias_of(output, value) for value in values):
                raise RegionCaptureError(
                    f"Region {region_id!r} outputs must not alias {boundary}; "
                    "clone the aliased outputs inside the region"
                )


def validate_region_mutations(exported_program: torch.export.ExportedProgram) -> None:
    """Reject input/buffer writes anywhere in a model containing live regions."""
    from ._capture import region_hop_records

    if not region_hop_records(exported_program.graph_module):
        return
    mutations: list[str] = []
    for output in exported_program.graph_signature.output_specs:
        if output.kind == OutputKind.BUFFER_MUTATION:
            mutations.append(f"buffer {output.target!r}")
        elif output.kind == OutputKind.USER_INPUT_MUTATION:
            mutations.append(f"input {output.target!r}")
    if mutations:
        raise RegionCaptureError(
            "Explicit regions require a functional model: mutation of "
            + ", ".join(mutations)
            + " is unsupported anywhere in a model containing live regions, "
            "including outside the annotated scope."
        )


def validate_model_mode(model: torch.nn.Module) -> None:
    """Reject a model with regions that has a module in training mode."""
    training = [
        name or "<the model>" for name, child in model.named_modules() if child.training
    ]
    if training:
        shown = ", ".join(training[:5]) + (", ..." if len(training) > 5 else "")
        raise RegionCaptureError(
            f"Regions currently require model.eval(); still in training mode: {shown}"
        )


def validate_eval_mode(gm: torch.fx.GraphModule) -> None:
    """Reject training-mode sampling outside the regions of a traced program.

    A traced program does not say whether its model was in training mode; the
    dropout it recorded does. Batch-norm updates show up as buffer mutations,
    which ``validate_region_mutations`` rejects. A caller that checked the
    model itself says so on the compilation session, and this check is skipped:
    an eval-mode model may keep dropout active on purpose.
    """
    from torch_tensorrt.region._session import current_session

    from ._capture import _SAMPLING_SWITCHES, _draws_random_numbers, region_hop_records

    session = current_session()
    if getattr(session, "model_in_eval_mode", False) or not region_hop_records(gm):
        return
    for node in gm.graph.nodes:
        if not isinstance(node.target, torch._ops.OpOverload):
            continue
        # An in-place variant shares its switches with the plain operator.
        name = node.target._schema.name.partition("::")[2].rstrip("_")
        packet = getattr(torch.ops.aten, name, None)
        if packet in _SAMPLING_SWITCHES and _draws_random_numbers(node):
            raise RegionCaptureError(
                "Regions currently require a model in eval mode: "
                f"{node.target} at {node.name!r} samples at random, as a "
                "model traced in training mode does. Call model.eval() before "
                "tracing it"
            )
