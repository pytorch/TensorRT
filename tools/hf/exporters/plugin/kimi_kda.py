"""Kimi KDA custom op lowered to the Edge-LLM ``kimi_kda`` plugin."""

from __future__ import annotations

from typing import Tuple

import torch


def register_kimi_kda_plugin_op() -> None:
    """Register ``trt::kimi_kda_plugin`` for Dynamo export."""
    if hasattr(torch.ops, "trt") and hasattr(torch.ops.trt, "kimi_kda_plugin"):
        return

    @torch.library.custom_op("trt::kimi_kda_plugin", mutates_args=())
    def kimi_kda_plugin(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state: torch.Tensor,
        context_lengths: torch.Tensor,
        lower_bound: float,
        use_lower_bound: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del q, k, gate, beta, a_log, dt_bias
        del context_lengths, lower_bound, use_lower_bound
        return torch.empty_like(v), torch.empty_like(state)

    @kimi_kda_plugin.register_fake
    def _(
        q,
        k,
        v,
        gate,
        beta,
        a_log,
        dt_bias,
        state,
        context_lengths,
        lower_bound,
        use_lower_bound,
    ):
        del q, k, gate, beta, a_log, dt_bias
        del context_lengths, lower_bound, use_lower_bound
        return torch.empty_like(v), torch.empty_like(state)
