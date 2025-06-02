import time
import torch
import torch.nn as nn
import torch_tensorrt
from transformers import AutoModel
from transformers.models.qwen2 import modeling_qwen2 as mq


# ---------------------------------------------------------------------------
# Patch Qwen2Attention to use export-friendly SDPA (no flash-attention kernels)
# ---------------------------------------------------------------------------

def patched_qwen2_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_value=None,
    cache_position=None,
    **kwargs,
):
    """Simplified scaled-dot-product attention (fp16 friendly, exportable)."""

    B, S, _ = hidden_states.shape

    # linear projections
    q = self.q_proj(hidden_states)
    k = self.k_proj(hidden_states)
    v = self.v_proj(hidden_states)

    # reshape to (B, nH, S, dH)
    q = q.view(B, S, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
    k = k.view(B, S, self.config.num_key_value_heads, self.head_dim)
    k = k.repeat_interleave(self.num_key_value_groups, dim=2).transpose(1, 2)
    v = v.view(B, S, self.config.num_key_value_heads, self.head_dim)
    v = v.repeat_interleave(self.num_key_value_groups, dim=2).transpose(1, 2)

    # rotary embedding
    cos, sin = position_embeddings
    q, k = mq.apply_rotary_pos_emb(q, k, cos, sin)

    # scaled dot-product
    attn = (q @ k.transpose(-2, -1)) * self.scaling
    if attention_mask is not None:
        # attention_mask is (B, S) float; convert to additive mask (B,1,1,S)
        mask = (1.0 - attention_mask[:, None, None, :]).to(q.dtype) * (-65504.0)
        attn = attn + mask

    attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    if self.training and self.attention_dropout > 0:
        attn = torch.nn.functional.dropout(attn, p=self.attention_dropout, training=True)

    ctx = attn @ v  # (B, nH, S, dH)
    ctx = ctx.transpose(1, 2).contiguous().view(B, S, -1)
    ctx = self.o_proj(ctx)

    return ctx, None

# apply patch
mq.Qwen2Attention.forward = patched_qwen2_attention_forward


# ----------------------------------------------------------------------------
# 1) Load Eagle2-2B and extract language model
# ----------------------------------------------------------------------------
device = torch.device("cuda:0")

model = (
    AutoModel.from_pretrained(
        "nvidia/Eagle2-2B", trust_remote_code=True, torch_dtype=torch.float16
    )
    .to(device)
    .eval()
)
llm = model.language_model

# llm.config._attn_implementation = "sdpa"

# ----------------------------------------------------------------------------
# 2) Minimal wrapper: forward only calls llm
# ----------------------------------------------------------------------------
class EagleLMWrapper(nn.Module):
    def __init__(self, llm_module: nn.Module):
        super().__init__()
        self.llm = llm_module

    @torch.no_grad()
    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        # Call the language model, disable kv-cache
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

wrapper = EagleLMWrapper(llm).to(device).eval()

# ----------------------------------------------------------------------------
# 3) Prepare dummy inputs outside wrapper
# ----------------------------------------------------------------------------
# Use sequence length pattern 8*K-3 to satisfy guard (e.g., 13, 21, ...)
batch_size = 2
seq_len = 13  # 8*2 - 3
vocab_size = llm.config.vocab_size

# Random token IDs
dummy_input_ids = torch.randint(
    0,
    vocab_size,
    (batch_size, seq_len),
    device=device,
    dtype=torch.long,
)

# Compute input embeddings
with torch.no_grad():
    dummy_inputs_embeds = llm.get_input_embeddings()(dummy_input_ids)

# 2D attention mask (batch, seq_len) float16 ones (no mask)
dummy_attention_mask = torch.ones(
    (batch_size, seq_len), dtype=torch.float16, device=device
)

# ----------------------------------------------------------------------------
# 4) Export with dynamic shapes
# ----------------------------------------------------------------------------
# Dynamic seq = 8*K - 3
B = torch.export.Dim("batch", min=1, max=4)
S = torch.export.Dim("seq",   min=1, max=64)
seq_sym = 8 * S - 3

dynamic_shapes = {
    "inputs_embeds":  {0: B, 1: seq_sym},
    "attention_mask": {0: B, 1: seq_sym},
}

# use torch.export.export instead of draft_export for stable tracing
with torch.inference_mode():
    exported = torch.export.export(
        wrapper,
        args=(dummy_inputs_embeds, dummy_attention_mask),
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )

# ----------------------------------------------------------------------------
# 5) Compile with Torch-TensorRT
# ----------------------------------------------------------------------------
trt_wrapper = torch_tensorrt.dynamo.compile(
    exported,
    inputs=[dummy_inputs_embeds, dummy_attention_mask],
    enabled_precisions={torch.float16},
    device=device,
    truncate_double=True,
)

# ----------------------------------------------------------------------------
# 6) Validate outputs
# ----------------------------------------------------------------------------
def compare_outputs():
    with torch.inference_mode():
        ref = wrapper(dummy_inputs_embeds, dummy_attention_mask)
        pred = trt_wrapper(dummy_inputs_embeds, dummy_attention_mask)

    # Diff metrics
    max_err = (ref - pred).abs().max().item()
    mean_err = (ref - pred).abs().mean().item()
    # Cosine similarity batch-wise
    ref_flat = ref.flatten(1).float()
    pred_flat = pred.flatten(1).float()
    cos_sim = torch.nn.functional.cosine_similarity(ref_flat, pred_flat, dim=1).mean().item()

    print(f"LLM max abs diff : {max_err:.6f}")
    print(f"LLM mean abs diff: {mean_err:.6f}")
    print(f"LLM cos sim     : {cos_sim:.6f}")

# Run comparison
compare_outputs()
