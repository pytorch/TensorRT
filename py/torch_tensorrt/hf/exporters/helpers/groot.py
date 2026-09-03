from __future__ import annotations

from typing import Any

import torch

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
