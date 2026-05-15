"""Comprehensive MoE subgraph tests for TRT converter bug discovery.

Covers all Mixture-of-Experts routing and dispatch variants found in popular
open-source models: Mixtral, Llama4, Qwen2-MoE, Qwen3-MoE, DeepSeek-V2/V3,
and NVIDIA Nemotron-H.  Each test class instantiates a self-contained MoE
block and validates TRT output against PyTorch reference.

Routing variants covered
------------------------
  Softmax routing (Mixtral, Qwen2, Qwen3):
    softmax(gate_logits) → topk → optional renormalization
  Sigmoid routing (Llama4, DeepSeek-V3/R1, Nemotron):
    sigmoid(gate_logits) → topk

Group-limited greedy selection (DeepSeek, Nemotron):
  Two sub-variants are represented:
    max-per-group (DeepSeek-V2):  group score = max expert score in group
  top2-sum-per-group (DeepSeek-V3 / Nemotron):
    group score = sum of top-2 expert scores; e_score_correction_bias added

Shared expert variants
----------------------
  None (Mixtral, Qwen3): all computation goes through the routed experts only
  Always-on, unweighted (Llama4, DeepSeek, Nemotron):
    shared output always added to routed output
  Sigmoid-gated scalar (Qwen2):
    shared output weighted by sigmoid(Linear(hidden, 1)) per token

Expert MLP styles
-----------------
  SwiGLU / gated MLP (all except Nemotron):
    output = down_proj(act(gate_proj(x)) * up_proj(x))
  Plain 2-layer MLP (Nemotron):
    output = down_proj(act(up_proj(x)))

Dispatch mechanism
------------------
  Scatter-based dense dispatch (used in all test classes here):
    Build routing_weight_matrix [T, N] via scatter_, run every expert on all
    tokens, accumulate weighted outputs.  This is the only dispatch pattern
    compatible with torch.export + static shapes.

    The original models use three dispatch patterns that are NOT directly
    exportable and are therefore approximated:
      index_add dispatch (Mixtral, Qwen2, Qwen3, Nemotron):
        torch.where(expert_mask) returns dynamic-size indices; the subsequent
        hidden_states[top_x] index is data-dependent → rejected by torch.export.
      Sort-based dispatch (DeepSeek moe_infer):
        tokens_per_expert.cpu().numpy() is a device sync + Python loop over
        dynamic counts → rejected by torch.export.
      Dense-broadcast dispatch (Llama4):
        hidden.repeat(N, 1) + sigmoid mask → zero-weight experts contribute ~0;
        tested as-is since it IS export-friendly.

  The scatter-based approximation in all non-Llama4 classes computes the
  identical numerical result as the original index_add dispatch; it is a
  mathematical equivalence, not a compromise.

Known limitations
-----------------
  FP32 MoE with large token counts: accumulated rounding in the routing
    scatter + expert matmul chain causes larger divergence than FP16; tests
    use atol=1e-3 for FP32 cases.

Test classes
------------
  TestMixtralStyleMoE      softmax routing, SwiGLU, no shared expert
  TestQwen2StyleMoE        softmax routing, SwiGLU, sigmoid-gated shared expert
  TestQwen3StyleMoE        softmax routing + optional norm_topk_prob, SwiGLU, no shared
  TestLlama4StyleMoE       sigmoid routing, dense broadcast, batched bmm experts, shared expert
  TestDeepSeekV2StyleMoE   sigmoid/softmax + group_limited_greedy (max-per-group), shared expert
  TestDeepSeekV3StyleMoE   sigmoid + group_limited_greedy (top2-sum + bias), shared expert
  TestNemotronStyleMoE     sigmoid + group_limited_greedy (top2-sum + bias), shared expert, plain MLP
"""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from parameterized import parameterized
from torch.testing._internal.common_utils import run_tests

from ..conversion.harness import DispatchTestCase

_BF16_SKIP = unittest.skipIf(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(torch.cuda.current_device()).major < 8,
    "BF16 requires Ampere (SM80) or higher",
)


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


class SwiGLUExpert(nn.Module):
    """Gated MLP used by Mixtral, Qwen, DeepSeek (gate_proj * act + up_proj → down_proj)."""

    def __init__(self, hidden_size: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class PlainMLPExpert(nn.Module):
    """Non-gated MLP used by Nemotron-H (up_proj → act → down_proj, no gate)."""

    def __init__(self, hidden_size: int, ffn_dim: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.up_proj(x)))


def _scatter_dispatch(
    hidden: torch.Tensor,
    experts: nn.ModuleList,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Export-friendly dense dispatch: one_hot routing weights, run all experts.

    Mathematically equivalent to the index_add dispatch used in Mixtral/Qwen;
    avoids data-dependent torch.where indexing that torch.export rejects, and
    avoids torch.zeros(T, N) which fails when T is an FX Proxy.

    Args:
        hidden:           [T, hidden_size]
        routing_weights:  [T, top_k]
        selected_experts: [T, top_k]  (int indices)
        num_experts:      N

    Returns:
        [T, hidden_size]
    """
    # one_hot: [T, top_k, N]; multiply by weights then sum over top_k → [T, N]
    one_hot_mask = F.one_hot(selected_experts.long(), num_classes=num_experts).to(
        routing_weights.dtype
    )  # [T, top_k, N]
    weight_matrix = (one_hot_mask * routing_weights.unsqueeze(-1)).sum(dim=1)  # [T, N]

    final = torch.zeros_like(hidden)
    for i, expert in enumerate(experts):
        expert_out = expert(hidden)  # [T, hidden_size]
        final = final + expert_out * weight_matrix[:, i : i + 1].to(hidden.dtype)
    return final


# ---------------------------------------------------------------------------
# TestMixtralStyleMoE
# ---------------------------------------------------------------------------


class MixtralStyleMoE(nn.Module):
    """Softmax-routed MoE without shared expert (Mixtral, Qwen3 baseline).

    Routing: softmax(gate) → topk → always renormalize to sum=1.
    Dispatch: scatter-based dense (export-friendly equivalent of index_add).
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLUExpert(hidden_size, ffn_dim) for _ in range(num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, H = hidden_states.shape
        hidden = hidden_states.view(-1, H)
        router_logits = self.gate(hidden)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True
            )
        routing_weights = routing_weights.to(hidden.dtype)
        out = _scatter_dispatch(
            hidden, self.experts, routing_weights, selected_experts, self.num_experts
        )
        return out.view(B, S, H)


class TestMixtralStyleMoE(DispatchTestCase):
    # fmt: off
    @parameterized.expand(
        [
            # (name, batch, seq, hidden, ffn, num_experts, top_k, norm, dtype, atol)
            # --- Basic FP16 ---
            ("b1_s32_e4_k2_fp16",   1,  32, 64, 128, 4, 2, True,  torch.float16, 1e-2),
            ("b2_s64_e4_k2_fp16",   2,  64, 64, 128, 4, 2, True,  torch.float16, 1e-2),
            ("b1_s128_e8_k2_fp16",  1, 128, 64, 128, 8, 2, True,  torch.float16, 1e-2),
            # top_k=1 (single-expert routing)
            ("b1_s32_e4_k1_fp16",   1,  32, 64, 128, 4, 1, True,  torch.float16, 1e-2),
            # norm_topk_prob=False (Qwen3 config with norm disabled)
            ("b1_s32_e4_k2_nonorm_fp16", 1, 32, 64, 128, 4, 2, False, torch.float16, 1e-2),
            # Larger hidden_size
            ("b1_s32_e4_k2_h128_fp16", 1, 32, 128, 256, 4, 2, True, torch.float16, 1e-2),
            # --- FP32 ---
            ("b1_s32_e4_k2_fp32",   1,  32, 64, 128, 4, 2, True,  torch.float32, 1e-3),
            ("b1_s64_e8_k2_fp32",   1,  64, 64, 128, 8, 2, True,  torch.float32, 1e-3),
            # Mixtral-realistic (small proxy): 8 experts, top-2
            ("mixtral_proxy_fp16",  1,  64, 64, 128, 8, 2, True,  torch.float16, 1e-2),
            # Qwen3-realistic (small proxy): 8 experts, top-2, no norm
            ("qwen3_proxy_fp16",    1,  64, 64, 128, 8, 2, False, torch.float16, 1e-2),
        ]
    )
    # fmt: on
    def test_mixtral_style(
        self, name, batch, seq, hidden, ffn, n_exp, top_k, norm, dtype, atol
    ):
        mod = (
            MixtralStyleMoE(hidden, ffn, n_exp, top_k, norm_topk_prob=norm)
            .eval()
            .cuda()
            .to(dtype)
        )
        x = torch.randn(batch, seq, hidden, dtype=dtype)
        if dtype == torch.float32:
            # small diff between tf32 and float32 may cause the topk function to choose different experts
            disable_tf32 = True
        else:
            disable_tf32 = False
        self.run_test(
            mod,
            [x],
            rtol=1e-2,
            atol=atol,
            precision=dtype,
            enable_passes=True,
            use_dynamo_tracer=True,
            require_full_compilation=True,
            disable_tf32=disable_tf32,
        )


# ---------------------------------------------------------------------------
# TestQwen2StyleMoE
# ---------------------------------------------------------------------------


class Qwen2StyleMoE(nn.Module):
    """Softmax-routed MoE with sigmoid-gated scalar shared expert (Qwen2-MoE).

    Routing: softmax(gate) → topk → optional renorm.
    Shared expert: shared_output * sigmoid(Linear(hidden, 1)) added to routed output.
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_dim: int,
        shared_ffn_dim: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLUExpert(hidden_size, ffn_dim) for _ in range(num_experts)]
        )
        self.shared_expert = SwiGLUExpert(hidden_size, shared_ffn_dim)
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, H = hidden_states.shape
        hidden = hidden_states.view(-1, H)
        router_logits = self.gate(hidden)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True
            )
        routing_weights = routing_weights.to(hidden.dtype)
        routed_out = _scatter_dispatch(
            hidden, self.experts, routing_weights, selected_experts, self.num_experts
        )
        shared_out = self.shared_expert(hidden)
        shared_gate = torch.sigmoid(self.shared_expert_gate(hidden))  # [T, 1]
        out = routed_out + shared_gate * shared_out
        return out.view(B, S, H)


class TestQwen2StyleMoE(DispatchTestCase):
    # fmt: off
    @parameterized.expand(
        [
            # (name, batch, seq, hidden, ffn, shared_ffn, num_experts, top_k, norm, dtype, atol)
            ("b1_s32_e4_k2_fp16",     1,  32, 64, 128, 256, 4, 2, False, torch.float16, 1e-2),
            ("b2_s64_e4_k2_fp16",     2,  64, 64, 128, 256, 4, 2, False, torch.float16, 1e-2),
            ("b1_s32_e8_k2_fp16",     1,  32, 64, 128, 256, 8, 2, False, torch.float16, 1e-2),
            # norm_topk_prob=True
            ("b1_s32_e4_k2_norm_fp16",   1, 32, 64, 128, 256, 4, 2, True, torch.float16, 1e-2),
            # Larger shared expert intermediate size
            ("b1_s32_e4_k2_bigshared_fp16", 1, 32, 64, 128, 512, 4, 2, False, torch.float16, 1e-2),
            # FP32
            ("b1_s32_e4_k2_fp32",     1,  32, 64, 128, 256, 4, 2, False, torch.float32, 1e-3),
            # Qwen2-realistic proxy
            ("qwen2_proxy_fp16",      1,  64, 64, 128, 256, 8, 2, False, torch.float16, 1e-2),
        ]
    )
    # fmt: on
    def test_qwen2_style(
        self, name, batch, seq, hidden, ffn, shared_ffn, n_exp, top_k, norm, dtype, atol
    ):
        mod = (
            Qwen2StyleMoE(hidden, ffn, shared_ffn, n_exp, top_k, norm_topk_prob=norm)
            .eval()
            .cuda()
            .to(dtype)
        )
        x = torch.randn(batch, seq, hidden, dtype=dtype)
        if dtype == torch.float32:
            # small diff between tf32 and float32 may cause the topk function to choose different experts
            disable_tf32 = True
        else:
            disable_tf32 = False
        self.run_test(
            mod,
            [x],
            rtol=1e-2,
            atol=atol,
            precision=dtype,
            enable_passes=True,
            use_dynamo_tracer=True,
            require_full_compilation=True,
            disable_tf32=disable_tf32,
        )


# ---------------------------------------------------------------------------
# TestQwen3StyleMoE
# ---------------------------------------------------------------------------


class Qwen3StyleMoE(nn.Module):
    """Softmax-routed MoE without shared expert, configurable norm (Qwen3-MoE).

    Identical structure to MixtralStyleMoE but captures Qwen3's specific
    combination of optional norm_topk_prob and moe_intermediate_size.
    """

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                SwiGLUExpert(hidden_size, moe_intermediate_size)
                for _ in range(num_experts)
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, H = hidden_states.shape
        hidden = hidden_states.view(-1, H)
        router_logits = self.gate(hidden)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True
            )
        routing_weights = routing_weights.to(hidden.dtype)
        out = _scatter_dispatch(
            hidden, self.experts, routing_weights, selected_experts, self.num_experts
        )
        return out.view(B, S, H)


class TestQwen3StyleMoE(DispatchTestCase):
    # fmt: off
    @parameterized.expand(
        [
            # (name, batch, seq, hidden, moe_ffn, num_experts, top_k, norm, dtype, atol)
            ("b1_s32_e4_k2_norm_fp16",    1,  32, 64, 128, 4, 2, True,  torch.float16, 1e-2),
            ("b1_s32_e4_k2_nonorm_fp16",  1,  32, 64, 128, 4, 2, False, torch.float16, 1e-2),
            ("b2_s64_e8_k2_norm_fp16",    2,  64, 64, 128, 8, 2, True,  torch.float16, 1e-2),
            ("b1_s128_e4_k1_fp16",        1, 128, 64, 128, 4, 1, True,  torch.float16, 1e-2),
            ("b1_s32_e4_k2_fp32",         1,  32, 64, 128, 4, 2, True,  torch.float32, 1e-3),
            # Qwen3-MoE-0.6B proxy: 64 experts, top-8 → scaled down to 8 experts, top-2
            ("qwen3_moe_proxy_fp16",      1,  64, 64, 128, 8, 2, True,  torch.float16, 1e-2),
        ]
    )
    # fmt: on
    def test_qwen3_style(
        self, name, batch, seq, hidden, moe_ffn, n_exp, top_k, norm, dtype, atol
    ):
        mod = (
            Qwen3StyleMoE(hidden, moe_ffn, n_exp, top_k, norm_topk_prob=norm)
            .eval()
            .cuda()
            .to(dtype)
        )
        x = torch.randn(batch, seq, hidden, dtype=dtype)
        self.run_test(
            mod,
            [x],
            rtol=1e-2,
            atol=atol,
            precision=dtype,
            enable_passes=True,
            use_dynamo_tracer=True,
            require_full_compilation=True,
        )


# ---------------------------------------------------------------------------
# TestLlama4StyleMoE
# ---------------------------------------------------------------------------


class Llama4StyleExperts(nn.Module):
    """All experts fused into batched matmuls — Llama4TextExperts pattern.

    Weights shape: [N, hidden, 2*ffn] (gate+up fused) and [N, ffn, hidden] (down).
    Input is tiled [N*T, hidden], reshaped to [N, T, hidden] for bmm.
    """

    def __init__(self, num_experts: int, hidden_size: int, ffn_dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, hidden_size, 2 * ffn_dim)
        )
        self.down_proj = nn.Parameter(torch.empty(num_experts, ffn_dim, hidden_size))
        nn.init.normal_(self.gate_up_proj, std=0.02)
        nn.init.normal_(self.down_proj, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [N*T, hidden_size]
        h = hidden_states.view(self.num_experts, -1, self.hidden_size)
        gate_up = torch.bmm(h, self.gate_up_proj)  # [N, T, 2*ffn]
        gate, up = gate_up.chunk(2, dim=-1)
        out = torch.bmm(up * F.silu(gate), self.down_proj)  # [N, T, hidden]
        return out.view(-1, self.hidden_size)


class Llama4StyleMoE(nn.Module):
    """Sigmoid-routed MoE with dense broadcast dispatch and always-on shared expert (Llama4).

    Routing: topk(logits) → scatter back to full expert space → sigmoid.
    Dispatch: tile all tokens N times; zero out non-selected via sigmoid(-inf)≈0.
    Shared expert: always-on Llama4TextMLP, output added unconditionally.
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_dim: int,
        shared_ffn_dim: int,
        num_experts: int,
        top_k: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = Llama4StyleExperts(num_experts, hidden_size, ffn_dim)
        self.shared_expert = SwiGLUExpert(hidden_size, shared_ffn_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, H = hidden_states.shape
        hidden = hidden_states.view(-1, H)  # [T, H]
        T = hidden.shape[0]

        router_logits = self.router(hidden)  # [T, N]
        top_values, top_indices = torch.topk(router_logits, self.top_k, dim=1)

        # Scatter selected logits back; fill unselected with -inf → sigmoid → ~0
        router_scores = (
            torch.full_like(router_logits, float("-inf"))
            .scatter_(1, top_indices, top_values)
            .transpose(0, 1)  # [N, T]
        )
        router_scores = torch.sigmoid(router_scores.float()).to(hidden.dtype)  # [N, T]

        # Dense broadcast: tile all tokens for all experts
        routed_in = hidden.repeat(self.num_experts, 1)  # [N*T, H]
        routed_in = routed_in * router_scores.reshape(-1, 1)  # zero non-selected
        routed_out = self.experts(routed_in)  # [N*T, H]

        # Sum contributions across experts
        expert_sum = routed_out.reshape(self.num_experts, T, H).sum(dim=0)  # [T, H]

        out = self.shared_expert(hidden) + expert_sum
        return out.view(B, S, H)


class TestLlama4StyleMoE(DispatchTestCase):
    # fmt: off
    @parameterized.expand(
        [
            # (name, batch, seq, hidden, ffn, shared_ffn, num_experts, top_k, dtype, atol)
            ("b1_s32_e4_k2_fp16",       1,  32, 64, 128, 256, 4, 2, torch.float16, 1e-2),
            ("b2_s64_e4_k2_fp16",       2,  64, 64, 128, 256, 4, 2, torch.float16, 1e-2),
            ("b1_s128_e8_k2_fp16",      1, 128, 64, 128, 256, 8, 2, torch.float16, 1e-2),
            # top_k=1
            ("b1_s32_e4_k1_fp16",       1,  32, 64, 128, 256, 4, 1, torch.float16, 1e-2),
            # FP32: dense-broadcast accumulation in FP32 has larger rounding; loosen atol
            ("b1_s32_e4_k2_fp32",       1,  32, 64, 128, 256, 4, 2, torch.float32, 1e-2),
            # Llama4-Scout proxy (16 experts, top-1)
            ("llama4_scout_proxy_fp16", 1,  64, 64, 128, 256, 8, 1, torch.float16, 1e-2),
        ]
    )
    # fmt: on
    def test_llama4_style(
        self, name, batch, seq, hidden, ffn, shared_ffn, n_exp, top_k, dtype, atol
    ):
        mod = (
            Llama4StyleMoE(hidden, ffn, shared_ffn, n_exp, top_k)
            .eval()
            .cuda()
            .to(dtype)
        )
        x = torch.randn(batch, seq, hidden, dtype=dtype)
        self.run_test(
            mod,
            [x],
            rtol=1e-2,
            atol=atol,
            precision=dtype,
            enable_passes=True,
            use_dynamo_tracer=True,
            require_full_compilation=True,
        )


# ---------------------------------------------------------------------------
# TestDeepSeekV2StyleMoE
# ---------------------------------------------------------------------------


def _group_limited_greedy_topk_max(
    scores: torch.Tensor, top_k: int, n_group: int, topk_group: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek-V2 group-limited greedy: group score = max expert score in group.

    Args:
        scores:     [T, N_experts] — softmax or sigmoid scores
        top_k:      number of experts to select per token
        n_group:    number of expert groups
        topk_group: number of groups to select

    Returns:
        topk_weight [T, top_k], topk_idx [T, top_k]
    """
    T, N = scores.shape
    experts_per_group = N // n_group

    # Score each group by its best expert
    group_scores = (
        scores.view(T, n_group, experts_per_group).max(dim=-1).values
    )  # [T, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [T, topk_group]

    # Build per-expert mask from selected groups
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = (
        group_mask.unsqueeze(-1).expand(T, n_group, experts_per_group).reshape(T, N)
    )

    masked_scores = scores.masked_fill(~score_mask.bool(), 0.0)
    topk_weight, topk_idx = torch.topk(masked_scores, k=top_k, dim=-1, sorted=False)
    return topk_weight, topk_idx


class DeepSeekV2StyleMoE(nn.Module):
    """Group-limited greedy MoE (max-per-group) with shared expert (DeepSeek-V2).

    Routing: softmax(gate) → group_limited_greedy (max per group) → optional renorm.
    Shared expert: always-on SwiGLU added to routed output.
    """

    def __init__(
        self,
        hidden_size: int,
        moe_ffn_dim: int,
        shared_ffn_dim: int,
        num_experts: int,
        top_k: int,
        n_group: int,
        topk_group: int,
        norm_topk_prob: bool = False,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.gate_weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        nn.init.normal_(self.gate_weight, std=0.02)
        self.experts = nn.ModuleList(
            [SwiGLUExpert(hidden_size, moe_ffn_dim) for _ in range(num_experts)]
        )
        self.shared_expert = SwiGLUExpert(hidden_size, shared_ffn_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, H = hidden_states.shape
        hidden = hidden_states.view(-1, H)

        logits = F.linear(hidden.float(), self.gate_weight.float())
        scores = F.softmax(logits, dim=-1)

        topk_weight, topk_idx = _group_limited_greedy_topk_max(
            scores, self.top_k, self.n_group, self.topk_group
        )
        if self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        else:
            topk_weight = topk_weight * self.routed_scaling_factor
        topk_weight = topk_weight.to(hidden.dtype)

        routed_out = _scatter_dispatch(
            hidden, self.experts, topk_weight, topk_idx, self.num_experts
        )
        out = routed_out + self.shared_expert(hidden)
        return out.view(B, S, H)


class TestDeepSeekV2StyleMoE(DispatchTestCase):
    # fmt: off
    @parameterized.expand(
        [
            # (name, batch, seq, hidden, moe_ffn, shared_ffn, n_exp, top_k, n_group, topk_group, norm, scale, dtype, atol)
            ("b1_s32_e4_k2_g2_tg1_fp16",   1,  32, 64, 128, 256, 4, 2, 2, 1, False, 1.0, torch.float16, 1e-2),
            ("b2_s64_e4_k2_g2_tg1_fp16",   2,  64, 64, 128, 256, 4, 2, 2, 1, False, 1.0, torch.float16, 1e-2),
            ("b1_s32_e8_k2_g4_tg2_fp16",   1,  32, 64, 128, 256, 8, 2, 4, 2, False, 1.0, torch.float16, 1e-2),
            # norm_topk_prob=True
            ("b1_s32_e4_k2_g2_tg1_norm_fp16", 1, 32, 64, 128, 256, 4, 2, 2, 1, True, 1.0, torch.float16, 1e-2),
            # routed_scaling_factor != 1
            ("b1_s32_e4_k2_scale16_fp16",  1,  32, 64, 128, 256, 4, 2, 2, 1, False, 1.6, torch.float16, 1e-2),
            # FP32
            ("b1_s32_e4_k2_g2_tg1_fp32",   1,  32, 64, 128, 256, 4, 2, 2, 1, False, 1.0, torch.float32, 1e-3),
            # DeepSeek-V2-Lite proxy: 64 experts → 8 here, top-6 → top-2, 8 groups → 2
            ("deepseekv2_proxy_fp16",       1,  64, 64, 128, 256, 8, 2, 2, 1, True,  1.0, torch.float16, 1e-2),
        ]
    )
    # fmt: on
    def test_deepseekv2_style(
        self,
        name,
        batch,
        seq,
        hidden,
        moe_ffn,
        shared_ffn,
        n_exp,
        top_k,
        n_group,
        topk_group,
        norm,
        scale,
        dtype,
        atol,
    ):
        mod = (
            DeepSeekV2StyleMoE(
                hidden,
                moe_ffn,
                shared_ffn,
                n_exp,
                top_k,
                n_group,
                topk_group,
                norm_topk_prob=norm,
                routed_scaling_factor=scale,
            )
            .eval()
            .cuda()
            .to(dtype)
        )
        x = torch.randn(batch, seq, hidden, dtype=dtype)
        self.run_test(
            mod,
            [x],
            rtol=1e-2,
            atol=atol,
            precision=dtype,
            enable_passes=True,
            use_dynamo_tracer=True,
            require_full_compilation=True,
        )


# ---------------------------------------------------------------------------
# TestDeepSeekV3StyleMoE
# ---------------------------------------------------------------------------


def _group_limited_greedy_topk_top2sum(
    scores: torch.Tensor,
    top_k: int,
    n_group: int,
    topk_group: int,
    correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek-V3 / Nemotron group-limited greedy: group score = sum of top-2.

    Args:
        scores:           [T, N_experts] — raw sigmoid scores (before bias)
        top_k:            experts to select per token
        n_group:          number of expert groups
        topk_group:       groups to select
        correction_bias:  [N_experts] — per-expert additive bias for selection only

    Returns:
        topk_weight [T, top_k], topk_idx [T, top_k]
        (weights use raw sigmoid scores, not biased scores)
    """
    T, N = scores.shape
    experts_per_group = N // n_group

    scores_for_choice = scores + correction_bias.unsqueeze(0)  # [T, N]
    group_scores = (
        scores_for_choice.view(T, n_group, experts_per_group)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )  # [T, n_group]

    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = (
        group_mask.unsqueeze(-1).expand(T, n_group, experts_per_group).reshape(T, N)
    )

    masked_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    topk_idx = torch.topk(masked_scores, k=top_k, dim=-1, sorted=False)[1]
    topk_weight = scores.gather(1, topk_idx)  # use raw scores (not biased)
    return topk_weight, topk_idx


class DeepSeekV3StyleMoE(nn.Module):
    """Group-limited greedy MoE (top2-sum-per-group, correction bias) with shared expert (DeepSeek-V3/R1).

    Routing: sigmoid(gate) → group_limited_greedy (top-2 sum per group + bias) → optional renorm × scale.
    Shared expert: always-on SwiGLU added to routed output.
    """

    def __init__(
        self,
        hidden_size: int,
        moe_ffn_dim: int,
        shared_ffn_dim: int,
        num_experts: int,
        top_k: int,
        n_group: int,
        topk_group: int,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.gate_weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        nn.init.normal_(self.gate_weight, std=0.02)
        self.e_score_correction_bias = nn.Parameter(torch.zeros(num_experts))
        self.experts = nn.ModuleList(
            [SwiGLUExpert(hidden_size, moe_ffn_dim) for _ in range(num_experts)]
        )
        self.shared_expert = SwiGLUExpert(hidden_size, shared_ffn_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, H = hidden_states.shape
        hidden = hidden_states.view(-1, H)

        logits = F.linear(hidden.float(), self.gate_weight.float())
        scores = torch.sigmoid(logits)  # [T, N]

        topk_weight, topk_idx = _group_limited_greedy_topk_top2sum(
            scores,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.e_score_correction_bias.float(),
        )
        if self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weight = (topk_weight * self.routed_scaling_factor).to(hidden.dtype)

        routed_out = _scatter_dispatch(
            hidden, self.experts, topk_weight, topk_idx, self.num_experts
        )
        out = routed_out + self.shared_expert(hidden)
        return out.view(B, S, H)


class TestDeepSeekV3StyleMoE(DispatchTestCase):
    # fmt: off
    @parameterized.expand(
        [
            # (name, batch, seq, hidden, moe_ffn, shared_ffn, n_exp, top_k, n_group, topk_group, norm, scale, dtype, atol)
            ("b1_s32_e4_k2_g2_tg1_fp16",   1,  32, 64, 128, 256, 4, 2, 2, 1, True,  1.0, torch.float16, 1e-2),
            ("b2_s64_e4_k2_g2_tg1_fp16",   2,  64, 64, 128, 256, 4, 2, 2, 1, True,  1.0, torch.float16, 1e-2),
            ("b1_s32_e8_k2_g4_tg2_fp16",   1,  32, 64, 128, 256, 8, 2, 4, 2, True,  1.0, torch.float16, 1e-2),
            # norm_topk_prob=False
            ("b1_s32_e4_k2_nonorm_fp16",   1,  32, 64, 128, 256, 4, 2, 2, 1, False, 1.0, torch.float16, 1e-2),
            # routed_scaling_factor != 1
            ("b1_s32_e4_k2_scale25_fp16",  1,  32, 64, 128, 256, 4, 2, 2, 1, True,  2.5, torch.float16, 1e-2),
            # FP32
            ("b1_s32_e4_k2_fp32",          1,  32, 64, 128, 256, 4, 2, 2, 1, True,  1.0, torch.float32, 1e-3),
            # DeepSeek-R1 proxy: sigmoid + top2-sum-per-group
            ("deepseekr1_proxy_fp16",       1,  64, 64, 128, 256, 8, 2, 2, 1, True,  1.0, torch.float16, 1e-2),
        ]
    )
    # fmt: on
    def test_deepseekv3_style(
        self,
        name,
        batch,
        seq,
        hidden,
        moe_ffn,
        shared_ffn,
        n_exp,
        top_k,
        n_group,
        topk_group,
        norm,
        scale,
        dtype,
        atol,
    ):
        mod = (
            DeepSeekV3StyleMoE(
                hidden,
                moe_ffn,
                shared_ffn,
                n_exp,
                top_k,
                n_group,
                topk_group,
                norm_topk_prob=norm,
                routed_scaling_factor=scale,
            )
            .eval()
            .cuda()
            .to(dtype)
        )
        x = torch.randn(batch, seq, hidden, dtype=dtype)
        self.run_test(
            mod,
            [x],
            rtol=1e-2,
            atol=atol,
            precision=dtype,
            enable_passes=True,
            use_dynamo_tracer=True,
            require_full_compilation=True,
        )


# ---------------------------------------------------------------------------
# TestNemotronStyleMoE
# ---------------------------------------------------------------------------


class NemotronStyleMoE(nn.Module):
    """Group-limited greedy MoE (top2-sum + bias) with plain MLP experts and shared expert (Nemotron-H).

    Key difference from DeepSeekV3: expert MLP is non-gated (up → act → down, no gate_proj).
    Routing and dispatch are otherwise identical to DeepSeekV3.
    """

    def __init__(
        self,
        hidden_size: int,
        moe_ffn_dim: int,
        shared_ffn_dim: int,
        num_experts: int,
        top_k: int,
        n_group: int,
        topk_group: int,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.gate_weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        nn.init.normal_(self.gate_weight, std=0.02)
        self.e_score_correction_bias = nn.Parameter(torch.zeros(num_experts))
        # Plain MLP (no gate_proj) — Nemotron-H's non-gated expert
        self.experts = nn.ModuleList(
            [PlainMLPExpert(hidden_size, moe_ffn_dim) for _ in range(num_experts)]
        )
        self.shared_expert = PlainMLPExpert(hidden_size, shared_ffn_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, H = hidden_states.shape
        hidden = hidden_states.view(-1, H)

        logits = F.linear(hidden.float(), self.gate_weight.float())
        scores = torch.sigmoid(logits)

        topk_weight, topk_idx = _group_limited_greedy_topk_top2sum(
            scores,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.e_score_correction_bias.float(),
        )
        if self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weight = (topk_weight * self.routed_scaling_factor).to(hidden.dtype)

        routed_out = _scatter_dispatch(
            hidden, self.experts, topk_weight, topk_idx, self.num_experts
        )
        out = routed_out + self.shared_expert(hidden)
        return out.view(B, S, H)


class TestNemotronStyleMoE(DispatchTestCase):
    # fmt: off
    @parameterized.expand(
        [
            # (name, batch, seq, hidden, moe_ffn, shared_ffn, n_exp, top_k, n_group, topk_group, norm, scale, dtype, atol)
            ("b1_s32_e4_k2_g2_tg1_fp16",    1,  32, 64, 128, 256, 4, 2, 2, 1, True,  1.0, torch.float16, 1e-2),
            ("b2_s64_e4_k2_g2_tg1_fp16",    2,  64, 64, 128, 256, 4, 2, 2, 1, True,  1.0, torch.float16, 1e-2),
            ("b1_s128_e8_k2_g4_tg2_fp16",   1, 128, 64, 128, 256, 8, 2, 4, 2, True,  1.0, torch.float16, 1e-2),
            # norm_topk_prob=False
            ("b1_s32_e4_k2_nonorm_fp16",    1,  32, 64, 128, 256, 4, 2, 2, 1, False, 1.0, torch.float16, 1e-2),
            # Non-zero correction bias (Nemotron initializes it to zero but it's learned)
            ("b1_s32_e4_k2_bias_fp16",      1,  32, 64, 128, 256, 4, 2, 2, 1, True,  1.0, torch.float16, 1e-2),
            # FP32
            ("b1_s32_e4_k2_fp32",           1,  32, 64, 128, 256, 4, 2, 2, 1, True,  1.0, torch.float32, 1e-3),
            # Nemotron-H proxy: plain MLP + group-limited greedy
            ("nemotron_proxy_fp16",          1,  64, 64, 128, 256, 8, 2, 2, 1, True,  1.0, torch.float16, 1e-2),
        ]
    )
    # fmt: on
    def test_nemotron_style(
        self,
        name,
        batch,
        seq,
        hidden,
        moe_ffn,
        shared_ffn,
        n_exp,
        top_k,
        n_group,
        topk_group,
        norm,
        scale,
        dtype,
        atol,
    ):
        mod = (
            NemotronStyleMoE(
                hidden,
                moe_ffn,
                shared_ffn,
                n_exp,
                top_k,
                n_group,
                topk_group,
                norm_topk_prob=norm,
                routed_scaling_factor=scale,
            )
            .eval()
            .cuda()
            .to(dtype)
        )
        # Non-zero correction bias to exercise the bias path
        if "bias" in name:
            with torch.no_grad():
                mod.e_score_correction_bias.data = (
                    torch.randn(n_exp, device="cuda") * 0.1
                )
        x = torch.randn(batch, seq, hidden, dtype=dtype)
        self.run_test(
            mod,
            [x],
            rtol=1e-2,
            atol=atol,
            precision=dtype,
            enable_passes=True,
            use_dynamo_tracer=True,
            require_full_compilation=True,
        )


if __name__ == "__main__":
    run_tests()
