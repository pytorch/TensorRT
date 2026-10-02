from __future__ import annotations

import torch
import torch.nn as nn


def nchw_to_hwc(pixel_values: torch.Tensor) -> torch.Tensor:
    if pixel_values.ndim != 4:
        raise ValueError(
            f"PI0.5 vision input must be rank 4, got {tuple(pixel_values.shape)}"
        )
    return pixel_values.permute(0, 2, 3, 1).contiguous()


class Pi05HwcVision(nn.Module):
    """Adapt VitRunner HWC input to the patched PI0.5 NCHW vision module."""

    def __init__(self, paligemma: nn.Module) -> None:
        super().__init__()
        self.paligemma = paligemma

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim != 4:
            raise ValueError(
                f"PI0.5 HWC vision input must be rank 4, got {tuple(pixel_values.shape)}"
            )
        nchw = pixel_values.permute(0, 3, 1, 2).contiguous()
        features = self.paligemma(nchw)
        if features.ndim == 3:
            features = features.reshape(-1, features.shape[-1])
        if features.ndim != 2:
            raise ValueError(
                "PI0.5 vision output must be rank 2 or 3 before flattening, "
                f"got {tuple(features.shape)}"
            )
        return features
