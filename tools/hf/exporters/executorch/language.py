from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch_tensorrt.executorch import export as export_executorch

from ..ops import call_language_prefill
from .artifact import EdgeExecuTorchArtifact
from .partitioner import EdgeLLMPartitioner


class EdgeLanguagePrefillModule(nn.Module):
    """Export-only module containing one language prefill component."""

    def __init__(self, artifact: EdgeExecuTorchArtifact) -> None:
        super().__init__()
        self.metadata_json = artifact.edge_metadata_json
        self.register_buffer(
            "trt_blob",
            torch.frombuffer(
                bytearray(artifact.trt_blob),
                dtype=torch.uint8,
            ),
            persistent=True,
        )

    def forward(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return call_language_prefill(
            self.trt_blob,
            self.metadata_json,
            *tensors,
        )


def lower_language_prefill_to_executorch(
    artifact: EdgeExecuTorchArtifact,
    example_inputs: tuple[torch.Tensor, ...],
) -> Any:
    if not example_inputs:
        raise ValueError("Language prefill requires at least one example input")

    exported = torch.export.export(
        EdgeLanguagePrefillModule(artifact).eval(),
        example_inputs,
        strict=False,
    )
    return export_executorch(
        exported,
        partitioners=[EdgeLLMPartitioner()],
    )


def save_language_prefill_pte(
    artifact: EdgeExecuTorchArtifact,
    example_inputs: tuple[torch.Tensor, ...],
    output_path: str | Path,
) -> None:
    edge_program = lower_language_prefill_to_executorch(
        artifact,
        example_inputs,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        edge_program.to_executorch().write_to_file(output)
