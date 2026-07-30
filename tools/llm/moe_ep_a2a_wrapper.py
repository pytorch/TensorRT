"""
Manual all_to_all Expert-Parallel MoE wrapper (export-friendly).

WHY THIS EXISTS
---------------
HF's OOTB ``DistributedConfig(enable_expert_parallel=True)`` lowers to
``all_reduce`` (dense expert sharding), NEVER ``all_to_all`` -- verified on the
Qwen3-30B graph (528 all_reduce, 0 all_to_all, with and without tp_plan). To get
true token-routing EP (the scalable multi-node form) we must insert all_to_all
ourselves. DTensor EP is also compile-only; the proven all_to_all mechanism
(Ulysses CP) is on the EXPORT path. So this wrapper does manual expert sharding
+ md.all_to_all so the block is torch.export-traceable and emits real all_to_all
nodes that #4321's converter lowers to a native TRT DistCollective.

DESIGN (GShard-style capacity dispatch)
---------------------------------------
Tokens must be SHARDED across ranks for all_to_all to mean anything (replicated
tokens => all_reduce is natural, which is what OOTB already does). So inside the
block:

    [B,S,H] (replicated)
      -> flatten [T,H]; slice this rank's T/N tokens        (shard; input replicated => free)
      -> router: top-k experts + weights
      -> capacity dispatch: scatter tokens into [E, cap, H]  (fixed cap => static shapes)
      -> md.all_to_all (dispatch): rank receives its experts' tokens from all ranks
      -> run local El = E/N experts
      -> md.all_to_all (combine): results back to token owners
      -> weighted combine -> [T/N, H]
      -> md.all_gather -> [T, H] -> [B,S,H]                  (re-gather for replicated next layer)

Fixed ``cap`` makes every all_to_all chunk equal-sized, so md.all_to_all
(equal-chunk) works unchanged.

STATUS: v1, UNVALIDATED. Validate in this order:
  1. eager self-test: wrapper output vs original block (run_self_test below), N=1 then N=2
  2. torch.export the wrapped model
  3. torch_tensorrt.dynamo.compile + numerics vs eager

ASSUMPTIONS to confirm on a box with transformers:
  - Qwen3MoeSparseMoeBlock has: .gate (Linear H->E), .experts (ModuleList[E]),
    .top_k, .num_experts, .norm_topk_prob
  - Each expert is Qwen3MoeMLP(x) -> [*, H]; block.forward returns
    (hidden_states, router_logits). We return the same tuple.
  - E % world_size == 0 and T % world_size == 0.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_tensorrt.distributed.md_conversion as md
except ImportError:
    # Fallback shim: the wrapper only needs set_config / all_to_all / all_gather,
    # which are thin wrappers over the standard torch functional collectives
    # (torch.ops._c10d_functional.*, present in any recent PyTorch). This lets the
    # EAGER self-test run on a machine without the dev-branch torch-tensorrt.
    # NOTE: export + TRT compile (moe_ep_a2a_export.py) still requires the
    # dev-branch torch-tensorrt (native DistCollective converters + distributed
    # runtime) — the shim only unblocks stage-1 logic validation.
    import types

    md = types.SimpleNamespace(world_size=None, rank=None, group_name=None)

    def _set_config(nb_ranks, rank_n, group):
        md.world_size, md.rank, md.group_name = nb_ranks, rank_n, group

    def _all_to_all(x, nb_ranks=None, dim=1):
        n = nb_ranks if nb_ranks is not None else md.world_size
        x = x.transpose(0, dim).contiguous()
        split = [x.shape[0] // n] * n
        f = torch.ops._c10d_functional.all_to_all_single.default(
            x, split, split, md.group_name
        )
        f = torch.ops._c10d_functional.wait_tensor(f)
        return f.transpose(0, dim).contiguous()

    def _all_gather(x, nb_ranks=None, dim=1):
        n = nb_ranks if nb_ranks is not None else md.world_size
        x = x.transpose(0, dim).contiguous()
        f = torch.ops._c10d_functional.all_gather_into_tensor(x, n, md.group_name)
        f = torch.ops._c10d_functional.wait_tensor(f)
        return f.transpose(0, dim).contiguous()

    md.set_config, md.all_to_all, md.all_gather = _set_config, _all_to_all, _all_gather


class ExpertParallelMoE(nn.Module):
    """Drop-in replacement for a sparse-MoE block that routes tokens with
    all_to_all across ``world_size`` ranks. Keeps only this rank's experts."""

    def __init__(
        self,
        gate: nn.Module,
        local_experts: nn.ModuleList,
        num_experts: int,
        top_k: int,
        world_size: int,
        rank: int,
        norm_topk_prob: bool = True,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        assert num_experts % world_size == 0, "E must be divisible by world_size"
        self.gate = gate
        self.local_experts = local_experts  # E/N experts owned by this rank
        self.num_experts = num_experts  # E (global)
        self.experts_per_rank = num_experts // world_size  # El
        self.top_k = top_k
        self.world_size = world_size
        self.rank = rank
        self.norm_topk_prob = norm_topk_prob
        self.capacity_factor = capacity_factor

    # -- capacity is fixed at trace time so shapes stay static for export --
    def _capacity(self, tokens_per_rank: int) -> int:
        # expected tokens per expert = Tl * k / E ; scale by factor, round up.
        exp = (tokens_per_rank * self.top_k) / self.num_experts
        cap = int(self.capacity_factor * exp + 0.999)
        return max(cap, 1)

    def forward(self, hidden_states: torch.Tensor):
        B, S, H = hidden_states.shape
        N, El = self.world_size, self.experts_per_rank
        x = hidden_states.reshape(-1, H)  # [T, H]
        T = x.shape[0]
        assert T % N == 0, "T must be divisible by world_size"
        Tl = T // N
        cap = self._capacity(Tl)

        # 1. shard tokens: this rank owns rows [rank*Tl : (rank+1)*Tl]
        x_local = x[self.rank * Tl : (self.rank + 1) * Tl]  # [Tl, H]

        # 2. route (float32 softmax like the HF block)
        # Qwen3MoeTopKRouter returns (logits, scores, indices); plain Linear returns logits.
        gate_out = self.gate(x_local)
        logits = gate_out[0] if isinstance(gate_out, tuple) else gate_out  # [Tl, E]
        weights = F.softmax(logits, dim=-1, dtype=torch.float)
        topw, topi = torch.topk(weights, self.top_k, dim=-1)  # [Tl, k]
        if self.norm_topk_prob:
            topw = topw / topw.sum(dim=-1, keepdim=True)
        topw = topw.to(x.dtype)

        # 3. capacity dispatch (GShard): position of each (token, slot) within
        #    its expert's buffer via cumsum of the one-hot over the token axis.
        onehot = F.one_hot(topi, self.num_experts).to(torch.int32)  # [Tl, k, E]
        # flatten (token,slot) -> a stable order for cumulative counting
        flat = onehot.reshape(Tl * self.top_k, self.num_experts)  # [Tl*k, E]
        pos = torch.cumsum(flat, dim=0) - 1  # slot index per expert
        pos = pos.reshape(Tl, self.top_k, self.num_experts)
        keep = (pos < cap) & (onehot > 0)  # [Tl,k,E] bool
        # Clamp BOTH ends: where an expert has no assignment yet, pos = -1 (cumsum
        # of an all-zero prefix minus 1) and F.one_hot rejects negatives. `keep`
        # already zeroes those entries, so clamping them to 0 is safe.
        pos = pos.clamp(min=0, max=cap - 1)

        # dispatch_mask[t, e, c] = 1 if token t (any slot) maps to (expert e, cap-slot c)
        pos_oh = F.one_hot(pos, cap).to(x.dtype)  # [Tl,k,E,cap]
        keepf = keep.unsqueeze(-1).to(x.dtype)  # [Tl,k,E,1]
        dispatch = (pos_oh * keepf).sum(dim=1)  # [Tl,E,cap]
        # combine_weights carries the routing weight at each kept slot
        combine = (pos_oh * keepf * topw[..., None, None]).sum(dim=1)  # [Tl,E,cap]

        # scatter tokens into the [E, cap, H] buffer
        dispatched = torch.einsum("tec,th->ech", dispatch, x_local)  # [E, cap, H]

        # 4. all_to_all dispatch: group experts by owner rank, exchange
        d = dispatched.reshape(N, El * cap, H)  # dim0 = owner rank
        d = md.all_to_all(d, N, dim=0)  # dim0 now = source rank
        d = d.reshape(N, El, cap, H).permute(1, 0, 2, 3).reshape(El, N * cap, H)

        # 5. local experts on [N*cap, H]
        outs: List[torch.Tensor] = []
        for i in range(El):
            outs.append(self.local_experts[i](d[i]))  # [N*cap, H]
        e_out = torch.stack(outs, dim=0)  # [El, N*cap, H]

        # 6. all_to_all combine: send results back to token owners
        e_out = e_out.reshape(El, N, cap, H).permute(1, 0, 2, 3).reshape(N, El * cap, H)
        e_out = md.all_to_all(e_out, N, dim=0)
        e_out = e_out.reshape(self.num_experts, cap, H)  # [E, cap, H]

        # 7. weighted combine -> [Tl, H]
        out_local = torch.einsum("tec,ech->th", combine, e_out)  # [Tl, H]

        # 8. re-gather full token set for the replicated next layer
        out_full = md.all_gather(out_local.unsqueeze(0), N, dim=1)  # [1, T, H]
        out_full = out_full.reshape(B, S, H)

        # match HF block's (hidden_states, router_logits) return contract
        return out_full, logits


class _Qwen3ExpertSlice(nn.Module):
    """Single expert sliced from a fused Qwen3MoeExperts weight tensor.

    Qwen3MoeExperts packs all experts into 3-D tensors:
        gate_up_proj: [E, 2*inter, hidden]
        down_proj:    [E, hidden,  inter]
    This wrapper holds one expert's slice and exposes the standard
    (x: [T, H]) -> [T, H] signature expected by ExpertParallelMoE.
    """

    def __init__(self, gate_up_w: torch.Tensor, down_w: torch.Tensor, act_fn):
        super().__init__()
        self.gate_up_proj = nn.Parameter(gate_up_w.clone())
        self.down_proj = nn.Parameter(down_w.clone())
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = F.linear(x, self.gate_up_proj).chunk(2, dim=-1)
        return F.linear(self.act_fn(gate) * up, self.down_proj)


def shard_moe_block(
    block: nn.Module, world_size: int, rank: int, capacity_factor: float = 1.25
) -> ExpertParallelMoE:
    """Replace a Qwen3MoeSparseMoeBlock with an ExpertParallelMoE that keeps only
    this rank's slice of experts (the rest are dropped -> the memory win).

    Handles Qwen3MoeExperts (fused 3-D weight tensors) by slicing per-expert
    weights into individual _Qwen3ExpertSlice modules.
    """
    num_experts = block.experts.num_experts
    top_k = block.experts.config.num_experts_per_tok
    norm_topk_prob = getattr(block.experts.config, "norm_topk_prob", True)
    El = num_experts // world_size
    lo = rank * El
    local = nn.ModuleList(
        [
            _Qwen3ExpertSlice(
                block.experts.gate_up_proj[lo + i].detach(),
                block.experts.down_proj[lo + i].detach(),
                block.experts.act_fn,
            )
            for i in range(El)
        ]
    )
    return ExpertParallelMoE(
        gate=block.gate,
        local_experts=local,
        num_experts=num_experts,
        top_k=top_k,
        world_size=world_size,
        rank=rank,
        norm_topk_prob=norm_topk_prob,
        capacity_factor=capacity_factor,
    )


# ---------------------------------------------------------------------------
# Minimal dense MoE reference (mirrors Qwen3MoeSparseMoeBlock semantics) with no
# transformers dependency. Module-scope so both run_self_test and the export
# driver (moe_ep_a2a_export.py) can import it. Replace with a real Qwen3 block later.
# ---------------------------------------------------------------------------
class _MLP(nn.Module):
    def __init__(self, h, inter):
        super().__init__()
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class RefMoE(nn.Module):
    """Minimal dense reference mirroring Qwen3MoeSparseMoeBlock semantics."""

    def __init__(self, h, inter, E, k):
        super().__init__()
        self.num_experts, self.top_k, self.norm_topk_prob = E, k, True
        self.gate = nn.Linear(h, E, bias=False)
        self.experts = nn.ModuleList([_MLP(h, inter) for _ in range(E)])

    def forward(self, hs):
        B, S, H = hs.shape
        x = hs.reshape(-1, H)
        w = F.softmax(self.gate(x), dim=-1, dtype=torch.float)
        tw, ti = torch.topk(w, self.top_k, dim=-1)
        tw = (tw / tw.sum(-1, keepdim=True)).to(x.dtype)
        out = torch.zeros_like(x)
        for t in range(x.shape[0]):
            for j in range(self.top_k):
                out[t] += tw[t, j] * self.experts[ti[t, j]](x[t : t + 1])[0]
        return out.reshape(B, S, H)


# ---------------------------------------------------------------------------
# Eager self-test: verify the wrapper reproduces a reference dense MoE block.
# Run BEFORE any export/TRT work. Single process (world_size=1) checks the
# dispatch/combine math; a 2-rank run checks the all_to_all path.
#   world_size=1 : python moe_ep_a2a_wrapper.py
#   world_size=2 : torchrun --nproc_per_node=2 moe_ep_a2a_wrapper.py
# ---------------------------------------------------------------------------
def run_self_test():
    import os

    import torch.distributed as dist

    ws = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if ws > 1:
        dist.init_process_group(backend="gloo")
        md.set_config(ws, rank, dist.group.WORLD.group_name)
    else:
        # single-process: still need group_name set; use a 1-rank gloo group
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29555")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo")
        md.set_config(1, 0, dist.group.WORLD.group_name)

    torch.manual_seed(0)
    H, INTER, E, K = 32, 64, 8, 2
    B, S = 1, 8 * ws  # T divisible by ws, and enough tokens per expert
    # CAP_FACTOR=4.0 -> cap = Tl (worst case: every token on one expert) -> drops
    # impossible. Discriminator: if the ws=2 error collapses to ~1e-7 at 4.0, the
    # residual error at 1.25 was capacity drops, not an exchange-layout bug.
    cap_factor = float(os.environ.get("CAP_FACTOR", "1.25"))
    ref = RefMoE(H, INTER, E, K).eval()
    with torch.no_grad():
        wrapped = shard_moe_block(ref, ws, rank, capacity_factor=cap_factor).eval()
        hs = torch.randn(B, S, H)
        ref_out = ref(hs)
        wrap_out, _ = wrapped(hs)
        max_err = (ref_out - wrap_out).abs().max().item()
    print(
        f"[rank {rank}] cap_factor={cap_factor} max_abs_err = {max_err:.3e}  "
        f"(capacity may drop tokens; expect ~0 when cap >= actual load)"
    )
    if ws > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    run_self_test()
