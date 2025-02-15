from typing import Optional, Tuple

import flashinfer
import torch


@torch.library.custom_op("tensorrt::flashinfer_forward", mutates_args=())
def flashinfer_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
) -> torch.Tensor:

    return flashinfer.single_decode_with_kv_cache(query, key, value)


@flashinfer_forward.register_fake
def _(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
) -> torch.Tensor:

    return torch.empty_like(query)
