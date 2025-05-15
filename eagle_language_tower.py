import time
import torch
import torch.nn as nn
import torch_tensorrt
from transformers import AutoModel, AutoTokenizer, GenerationMixin
from transformers.models.qwen2 import modeling_qwen2 as mq
from transformers.modeling_outputs import CausalLMOutputWithPast
import types

mq.ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = mq.ALL_ATTENTION_FUNCTIONS["sdpa"]

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

# -----------------------------------------------------------------------------
# Patch: ensure attention_mask dtype is bool/float for Qwen2 during generation
# -----------------------------------------------------------------------------

def _patch_attention_mask_dtype(lm):
    """Monkey-patch `lm.forward` so that any `attention_mask` with integer dtype is
    converted to `bool`. This avoids the runtime error you observed during
    generation without needing an extra wrapper."""

    # Fetch the *unbound* original forward (so we can pass `self` manually once).
    orig_forward = lm.__class__.forward

    def forward_patched(self, *args, **kwargs):
        attn_mask = kwargs.get("attention_mask", None)
        if attn_mask is not None: # and attn_mask.dtype not in (torch.bool, torch.float16, torch.float32):
            kwargs["attention_mask"] = attn_mask.to(torch.bool)
        return orig_forward(self, *args, **kwargs)

    lm.forward = types.MethodType(forward_patched, lm)

# Apply the patch to language model
_patch_attention_mask_dtype(llm)

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

# ----------------------------------------------------------------------------
# 7) Generation-capable wrappers & comparison
# ----------------------------------------------------------------------------

class EagleGenWrapper(nn.Module, GenerationMixin):
    """Wraps the original HF language model to expose GenerationMixin while keeping the
    interface identical to the base model. This is essentially a thin pass-through
    that only adds a few attributes that GenerationMixin expects (config,
    generation_config, etc.)."""

    def __init__(self, lm_module: nn.Module):
        super().__init__()
        self.lm = lm_module
        # Attributes expected by GenerationMixin
        self.config = lm_module.config
        self.main_input_name = lm_module.main_input_name
        self.generation_config = getattr(lm_module, "generation_config", None)
        # Expose device attribute expected by GenerationMixin
        self.device = next(lm_module.parameters()).device
        # Keep embedding layers so that GenerationMixin utility methods work.
        self._input_embeddings = lm_module.get_input_embeddings()
        self._output_embeddings = lm_module.get_output_embeddings()
        # Some helper functions that GenerationMixin tries to call if they exist.
        if hasattr(lm_module, "_apply_logits_warper"):
            self._apply_logits_warper = lm_module._apply_logits_warper
        if hasattr(lm_module, "_prepare_attention_mask_for_generation"):
            self._prepare_attention_mask_for_generation = (
                lm_module._prepare_attention_mask_for_generation
            )

    # ----- Interfaces required by GenerationMixin -----
    def get_input_embeddings(self):
        return self._input_embeddings

    def get_output_embeddings(self):
        return self._output_embeddings

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # Defer to underlying model implementation so we keep identical behaviour.
        return self.lm.prepare_inputs_for_generation(*args, **kwargs)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):
        # Convert mask dtype if necessary (HF generation often produces int64 masks)
        if attention_mask is not None: #  and attention_mask.dtype not in (torch.bool, torch.float16, torch.float32):
            attention_mask = attention_mask.to(torch.bool)

        return self.lm(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )


class EagleTRTGenWrapper(nn.Module, GenerationMixin):
    """Wraps the TensorRT-compiled module so it can be used with .generate(). The
    compiled module (``trt_wrapper``) only understands (inputs_embeds,
    attention_mask) and returns raw logits. This wrapper adds the necessary glue
    to convert token IDs -> embeddings and to expose the expected attributes for
    GenerationMixin."""

    def __init__(self, trt_module: nn.Module, original_lm: nn.Module, device="cuda:0"):
        super().__init__()
        self.trt_lm = trt_module
        self._orig_lm = original_lm  # for helper / utility methods
        self.device = torch.device(device)

        # ----- Attributes GenerationMixin expects -----
        self.config = original_lm.config
        self.main_input_name = original_lm.main_input_name
        self.generation_config = getattr(original_lm, "generation_config", None)
        self._input_embeddings = original_lm.get_input_embeddings()
        self._output_embeddings = original_lm.get_output_embeddings()
        if hasattr(original_lm, "_apply_logits_warper"):
            self._apply_logits_warper = original_lm._apply_logits_warper
        if hasattr(original_lm, "_prepare_attention_mask_for_generation"):
            self._prepare_attention_mask_for_generation = (
                original_lm._prepare_attention_mask_for_generation
            )

    def get_input_embeddings(self):
        return self._input_embeddings

    def get_output_embeddings(self):
        return self._output_embeddings

    def prepare_inputs_for_generation(self, *args, **kwargs):
        # Re-use the helper from the original HF model.
        return self._orig_lm.prepare_inputs_for_generation(*args, **kwargs)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):
        # GenerationMixin may supply either ``input_ids`` or ``inputs_embeds``.
        if inputs_embeds is None and input_ids is not None:
            # Convert token IDs to embeddings using original embedding layer.
            inputs_embeds = self._input_embeddings(input_ids).to(dtype=torch.float16, device=self.device)
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(dtype=torch.float16, device=self.device)
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")

        # The compiled TensorRT graph was exported with *only* ``inputs_embeds`` as
        # input, so we must not pass the attention mask tensor even if the caller
        # supplied one. Masking could be baked into the graph or omitted during
        # export. Here we simply ignore it.

        logits = self.trt_lm(inputs_embeds)
        return CausalLMOutputWithPast(logits=logits, past_key_values=None)


# ----------------------------------------------------------------------------
# 8) Compare generation outputs between Torch and TensorRT versions
# ----------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained("nvidia/Eagle2-1B", trust_remote_code=True)

def compare_generation_outputs(prompt: str = "Hello", max_new_tokens: int = 32):
    # Torch (HF) model: use it directly without wrapper
    hf_model = llm  # already on correct device / dtype
    trt_model = EagleTRTGenWrapper(trt_wrapper, llm, device=device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    gen_args = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=False,  # KV-cache off for deterministic comparison / TRT compat
    )

    with torch.inference_mode():
        t0 = time.time()
        torch_out = hf_model.generate(**inputs, **gen_args)
        t1 = time.time()
        torch_out_wo_attention_mask = hf_model.generate(inputs.input_ids, **gen_args)
        t2 = time.time()
        trt_out = trt_model.generate(**inputs, **gen_args)
        t3 = time.time()
        trt_out_wo_attention_mask = trt_model.generate(inputs.input_ids, **gen_args)
        t4 = time.time()
    decoded_torch = tokenizer.decode(torch_out[0], skip_special_tokens=True)
    decoded_torch_wo_attention_mask = tokenizer.decode(torch_out_wo_attention_mask[0], skip_special_tokens=True)
    decoded_trt = tokenizer.decode(trt_out[0], skip_special_tokens=True)
    decoded_trt_wo_attention_mask = tokenizer.decode(trt_out_wo_attention_mask[0], skip_special_tokens=True)


    print("\nTorch output (%.2f s):" % (t1 - t0))
    print(decoded_torch)
    print("\nTorch output (%.2f s):" % (t2 - t1))
    print(decoded_torch_wo_attention_mask)
    print("\nTensorRT output (%.2f s):" % (t3 - t2))
    print(decoded_trt)
    print("\nTensorRT output (%.2f s):" % (t4 - t3))
    print(decoded_trt_wo_attention_mask)

    identical = torch.equal(torch_out, trt_out)
    print(f"\nToken sequences identical: {identical}")


# Execute comparison when running as script
if __name__ == "__main__":

    # from transformers import AutoTokenizer

    # prompt = "Hey, are you conscious? Can you talk to me?"
    # inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # # Generate
    # generate_ids = model.language_model.generate(inputs.input_ids, max_length=30)
    # tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    # "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."

    compare_generation_outputs(prompt="What is your name?", max_new_tokens=32)
