# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

"""Whole-program restrictions for graphs containing explicit regions."""

import torch
from torch.export.graph_signature import OutputKind

from torch_tensorrt.region_annotations._errors import RegionCaptureError

from ._capture import region_hop_records


def validate_region_mutations(exported_program: torch.export.ExportedProgram) -> None:
    """Reject input/buffer writes anywhere in a model containing live regions."""
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
