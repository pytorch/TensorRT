from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch_tensorrt.executorch import export as export_executorch

from ..ops import call_vision_tower
from .artifact import EdgeExecuTorchArtifact
from .partitioner import EdgeLLMPartitioner


class EdgeVisionModule(nn.Module):
    """Export-only module containing one embedded Edge vision component."""

    def __init__(self, artifact: EdgeExecuTorchArtifact) -> None:
        super().__init__()
        self.metadata_json = artifact.edge_metadata_json
        self.register_buffer(
            "trt_blob",
            torch.frombuffer(bytearray(artifact.trt_blob), dtype=torch.uint8),
            persistent=True,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return call_vision_tower(self.trt_blob, self.metadata_json, pixel_values)[0]


def lower_vision_to_executorch(
    artifact: EdgeExecuTorchArtifact,
    example_input_hwc: torch.Tensor,
) -> Any:
    if example_input_hwc.ndim != 4:
        raise ValueError(
            "Edge VitRunner example input must have shape [B,H,W,C], got "
            f"{tuple(example_input_hwc.shape)}"
        )
    if example_input_hwc.dtype != torch.float16:
        raise ValueError(
            f"Edge VitRunner example input must be float16, got {example_input_hwc.dtype}"
        )

    exported = torch.export.export(
        EdgeVisionModule(artifact).eval(),
        (example_input_hwc,),
        strict=False,
    )
    return export_executorch(
        exported,
        partitioners=[EdgeLLMPartitioner()],
    )


def save_vision_pte(
    artifact: EdgeExecuTorchArtifact,
    example_input_hwc: torch.Tensor,
    output_path: str | Path,
) -> None:
    edge_program = lower_vision_to_executorch(artifact, example_input_hwc)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        edge_program.to_executorch().write_to_file(output)
