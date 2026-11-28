from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

GROOT_EMBODIMENT_MAPPING = {
    "new_embodiment": 31,
    "oxe_droid": 17,
    "agibot_genie1": 26,
    "gr1": 24,
}


def make_embodiment_id(
    policy: Any,
    state: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    embodiment_tag = getattr(policy.config, "embodiment_tag", "new_embodiment")
    return torch.full(
        (state.shape[0],),
        GROOT_EMBODIMENT_MAPPING.get(embodiment_tag, 0),
        dtype=dtype,
        device=device,
    )


def _groot(model: nn.Module) -> nn.Module:
    if hasattr(model, "_groot_model"):
        return model._groot_model
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "eagle_model"):
        return model
    raise RuntimeError(
        "GR00T spec expected GrootPolicy or a module with backbone.eagle_model"
    )
