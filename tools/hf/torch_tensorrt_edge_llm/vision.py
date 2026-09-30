from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .artifact import EdgeExecuTorchArtifact
from .ops import call_vision_tower


class EdgeVisionModule(nn.Module):
    """Runnable module containing one embedded Edge vision component."""

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


def export_vision(
    artifact: EdgeExecuTorchArtifact,
    example_input_hwc: torch.Tensor,
) -> torch.export.ExportedProgram:
    """Export a runnable vision operator without importing ExecuTorch."""
    if example_input_hwc.ndim != 4:
        raise ValueError(
            "Edge VitRunner example input must have shape [B,H,W,C], got "
            f"{tuple(example_input_hwc.shape)}"
        )
    if example_input_hwc.dtype != torch.float16:
        raise ValueError(
            f"Edge VitRunner example input must be float16, got {example_input_hwc.dtype}"
        )

    return torch.export.export(
        EdgeVisionModule(artifact).eval(),
        (example_input_hwc,),
        strict=False,
    )


def lower_vision_to_executorch(
    artifact: EdgeExecuTorchArtifact,
    example_input_hwc: torch.Tensor,
) -> Any:
    from torch_tensorrt.executorch import export as export_executorch

    from .executorch import EdgeLLMPartitioner

    return export_executorch(
        export_vision(artifact, example_input_hwc),
        partitioners=[EdgeLLMPartitioner()],
    )


def save_vision_pte(
    artifact: EdgeExecuTorchArtifact,
    example_input_hwc: torch.Tensor,
    output_path: str | Path,
) -> None:
    import torch_tensorrt

    from .executorch import EdgeLLMPartitioner

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch_tensorrt.save(
        export_vision(artifact, example_input_hwc),
        str(path),
        output_format="executorch",
        partitioners=[EdgeLLMPartitioner()],
    )
