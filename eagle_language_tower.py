import time
import torch
import torch.nn as nn
import torch_tensorrt
from transformers import AutoModel
from transformers.models.qwen2 import modeling_qwen2 as mq

# ----------------------------------------------------------------------------
# 1) Load Eagle2-2B and extract language model
# ----------------------------------------------------------------------------
device = torch.device("cuda:0")

model = (
    AutoModel.from_pretrained(
        "nvidia/Eagle2-1B", trust_remote_code=True, torch_dtype=torch.float16
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
        self.llm = llm_module          # SDPA 경로 유지

    @torch.no_grad()
    def forward(self, inputs_embeds, attention_mask=None):
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
# dummy_attention_mask = torch.ones(
#     (batch_size, seq_len), dtype=torch.bool, device=device
# )
dummy_attention_mask = None


# ----------------------------------------------------------------------------
# 4) Export with dynamic shapes
# ----------------------------------------------------------------------------
B = torch.export.Dim("batch", min=1, max=4)
S = torch.export.Dim("seq",   min=1, max=2048)

dynamic_shapes = {
    "inputs_embeds":  {0: B, 1: S},
    # "attention_mask": {0: B, 1: S},   # ← 마스크를 쓸 경우에만
}

# use torch.export.export instead of draft_export for stable tracing
with torch.inference_mode():
    exported = torch.export.export(
        wrapper,
        args=(dummy_inputs_embeds,),
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )

# ----------------------------------------------------------------------------
# 5) Compile with Torch-TensorRT
# ----------------------------------------------------------------------------
trt_wrapper = torch_tensorrt.dynamo.compile(
    exported,
    inputs=[dummy_inputs_embeds], # dummy_attention_mask],
    enabled_precisions={torch.float32},
    device=device,
    truncate_double=True,
    disable_tf32=True,
    use_explicit_typing=True,
    use_fp32_acc=True,
)

# ----------------------------------------------------------------------------
# 6) Validate outputs
# ----------------------------------------------------------------------------
def compare_outputs():
    with torch.inference_mode():
        ref = wrapper(dummy_inputs_embeds) # dummy_attention_mask)
        pred = trt_wrapper(dummy_inputs_embeds) #, dummy_attention_mask)

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
