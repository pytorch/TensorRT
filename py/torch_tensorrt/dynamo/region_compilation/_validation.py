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
                    f"Region {region_id} outputs must not alias {boundary}; "
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
