"""
.. _end_to_end_llm_generation_example:

End-to-End LLM Generation with Plugin (KV Cache Enabled)
=========================================================

This example demonstrates a full generation loop (Prefill + Decode) using
a Qwen2 model where attention is replaced by the TensorRT AttentionPlugin.
It validates that the plugin correctly handles KV cache updates across multiple steps
and produces identical text generation results compared to PyTorch.

**Plugin Library:**

This example uses a custom TensorRT plugin shared library (``libNvInfer_edgellm_plugin.so``) 
that replaces the standard transformer attention operations and RoPE (Rotary Position Embedding) 
computations with optimized CUDA kernels. The plugin automatically selects between FMHA 
(context processing) and XQA (token generation) kernels based on sequence length.

Plugin source code and build instructions (internal access only):

https://gitlab-master.nvidia.com/hoonkyungc/tensorrt-edgellm/-/blob/torchtrt-plugin-build/README_TORCHTRT_PLUGIN.md

**Key Features:**
1. **KV Cache Support**: Uses persistent KV cache tensors passed through the model.
2. **Multi-step Generation**: Implements a custom generation loop.
3. **Meaningful Input**: Generates text from a real prompt.
4. **Plugin Integration**: Replaces Qwen2Attention with PluginAttention.
5. **Text-Level Validation**: Verifies generated text matches PyTorch output exactly.

"""

import ctypes
import os
import torch
import torch.nn as nn
import torch_tensorrt
import numpy as np
import tensorrt as trt

# Enable plugin debug logging
os.environ["EDGELLM_DEBUG_PLUGIN"] = "1"

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from torch_tensorrt.dynamo.conversion import ConversionContext, dynamo_tensorrt_converter
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor
from typing import Tuple, Optional, List

# %%
# Configuration & Setup
# ---------------------

print("Initializing CUDA context...")
DEVICE = torch.device("cuda:0")
_ = torch.zeros(1, device=DEVICE)

# Load the plugin
PLUGIN_PATH = "/develop/TensorRT/tensorrt-edgellm/build/libNvInfer_edgellm_plugin.so"
ctypes.CDLL(PLUGIN_PATH)
print(f"Loaded plugin: {PLUGIN_PATH}")

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_SEQ_LEN = 128
DTYPE = torch.float16

# %%
# Plugin Registration & Converter
# -------------------------------

def get_plugin_rope_cache(rotary_emb, max_seq_len, head_dim, device):
    # Use inv_freq from the model's rotary embedding to ensure exact match
    inv_freq = rotary_emb.inv_freq.to(device).float()
    attention_scaling = rotary_emb.attention_scaling
    
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    
    # Qwen2RotaryEmbedding creates emb = cat(freqs, freqs) -> [c, c], [s, s]
    # We need [c, s] for the plugin (half dimension each)
    
    # Calculate cos/sin based on freqs (half dim)
    cos_half = freqs.cos() * attention_scaling
    sin_half = freqs.sin() * attention_scaling
    
    rope = torch.cat([cos_half, sin_half], dim=-1)
    return rope.unsqueeze(0)

def register_plugin_op():
    if hasattr(torch.ops.xqa, "attn"):
        return

    @torch.library.custom_op("xqa::attn", mutates_args=())
    def attn(qkv: torch.Tensor, kv: torch.Tensor, ctx_len: torch.Tensor, rope: torch.Tensor, nq: int, nkv: int, d: int) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = qkv.shape[0]
        seq_len = qkv.shape[1]
        attn_out = torch.zeros(batch_size, seq_len, nq, d, dtype=qkv.dtype, device=qkv.device)
        updated_kv = kv.clone()
        # Return zeros for output, and clone KV for functional behavior simulation
        return attn_out, updated_kv

    @torch.library.register_fake("xqa::attn")
    def _(qkv, kv, ctx_len, rope, nq, nkv, d):
        batch_size = qkv.shape[0]
        seq_len = qkv.shape[1]
        attn_out = torch.empty(batch_size, seq_len, nq, d, dtype=qkv.dtype, device=qkv.device)
        updated_kv = kv.clone()
        return attn_out, updated_kv

register_plugin_op()

@dynamo_tensorrt_converter(torch.ops.xqa.attn.default, supports_dynamic_shapes=True)
def convert_attn(ctx: ConversionContext, target, args, kwargs, name):
    qkv, kv, ctx_len, rope, nq, nkv, d = args[:7]
    
    creator = trt.get_plugin_registry().get_plugin_creator("AttentionPlugin", "1", "")
    if creator is None:
        raise RuntimeError("AttentionPlugin not found!")
    
    # Parameters for Qwen2.5-0.5B
    # nq=14, nkv=2, d=64
    
    field_list = [
        trt.PluginField("num_q_heads", np.array([nq], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("num_kv_heads", np.array([nkv], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("head_size", np.array([d], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("max_batch_size", np.array([4], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("kv_cache_capacity", np.array([MAX_SEQ_LEN], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("enable_tree_attention", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("enable_reuse_kv_cache", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32),
        # Enable internal copy to support functional PyTorch usage
        trt.PluginField("enable_kv_cache_copy", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32),
    ]
    
    fields = trt.PluginFieldCollection(field_list)
    plugin = creator.create_plugin(name, fields)
    
    inputs = [get_trt_tensor(ctx, i, f"{name}_i{idx}") if not isinstance(i, trt.ITensor) else i 
              for idx, i in enumerate([qkv, kv, ctx_len, rope])]

    # Reshape ctx_len to 1D if needed
    if len(inputs[2].shape) == 2 and inputs[2].shape[1] == 1:
        shuffle_layer = ctx.net.add_shuffle(inputs[2])
        shuffle_layer.reshape_dims = (inputs[2].shape[0],)
        inputs[2] = shuffle_layer.get_output(0)

    layer = ctx.net.add_plugin_v2(inputs, plugin)
    return layer.get_output(0), layer.get_output(1)

# %%
# Plugin Attention Module
# -----------------------

class PluginAttention(nn.Module):
    def __init__(self, original_attn, config, layer_idx, rope_cache):
        super().__init__()
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj
        
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        # Register RoPE cache as buffer
        self.register_buffer("rope_cache", rope_cache)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[torch.Tensor] = None,
        ctx_len: Optional[torch.Tensor] = None,
        **kwargs
    ):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        qkv = torch.cat([q, k, v], dim=-1)
        
        # ctx_len must be provided
        if ctx_len is None:
             # Fallback for export trace if not provided (should ideally be provided)
             # Assuming simple batch=1 case for fallback
             ctx_len = torch.tensor([seq_len], dtype=torch.int32, device=hidden_states.device).expand(batch_size)

        # Ensure RoPE is FP32
        rope_fp32 = self.rope_cache.float()
        
        # Plugin call
        # kv_cache input must be provided
        if past_key_value is None:
            raise ValueError("past_key_value (KV cache tensor) must be provided")
            
        attn_out, updated_kv = torch.ops.xqa.attn.default(
            qkv, past_key_value, ctx_len, rope_fp32,
            self.num_heads, self.num_key_value_heads, self.head_dim
        )
        
        attn_out = attn_out.reshape(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_out)
        
        # Return output and updated KV cache (as the second element, replacing present_key_value)
        return output, updated_kv

def replace_attention(model, config):
    # Generate RoPE cache using model's parameters
    rotary_emb = model.model.rotary_emb
    head_dim = config.hidden_size // config.num_attention_heads
    rope_cache = get_plugin_rope_cache(rotary_emb, MAX_SEQ_LEN, head_dim, DEVICE)
    
    for i, layer in enumerate(model.model.layers):
        layer.self_attn = PluginAttention(layer.self_attn, config, i, rope_cache)
    return model

# %%
# Model Wrapper for Export
# ------------------------

class Qwen2Wrapper(nn.Module):
    """
    Wraps Qwen2ForCausalLM to explicitly accept list of KV cache tensors
    and return list of updated KV cache tensors.
    Unrolls the model forward pass to bypass transformers' Cache object checks.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model # Qwen2ForCausalLM
        
    def forward(self, input_ids, position_ids, kv_caches: List[torch.Tensor], ctx_len: torch.Tensor):
        # Extract components
        transformer = self.model.model
        embed_tokens = transformer.embed_tokens
        layers = transformer.layers
        norm = transformer.norm
        lm_head = self.model.lm_head
        
        # Embeddings
        inputs_embeds = embed_tokens(input_ids)
        hidden_states = inputs_embeds
        
        new_kv_caches = []
        
        # Layers
        for i, layer in enumerate(layers):
            past_key_value = kv_caches[i]
            
            # Unroll Qwen2DecoderLayer logic to capture updated_kv from PluginAttention
            # Qwen2DecoderLayer generally does:
            # 1. input_layernorm
            # 2. self_attn
            # 3. residual
            # 4. post_attention_layernorm
            # 5. mlp
            # 6. residual
            
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            
            # Self Attention (PluginAttention)
            # Returns (hidden_states, updated_kv)
            hidden_states, updated_kv = layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=past_key_value,
                ctx_len=ctx_len
            )
            
            hidden_states = residual + hidden_states
            
            # Fully Connected
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
            
            new_kv_caches.append(updated_kv) 
            
        # Final Norm
        hidden_states = norm(hidden_states)
        
        # LM Head
        logits = lm_head(hidden_states)
        
        return logits, new_kv_caches

# %%
# Export & Compile
# ----------------

def compile_model(model, input_ids, position_ids, kv_caches, ctx_len):
    print("Exporting model...")
    
    # Dynamic shapes setup
    # seq_len can be 1 (decode) or >1 (prefill)
    seq_len_dim = torch.export.Dim("seq_len", min=1, max=MAX_SEQ_LEN)
    
    # Define dynamic shapes for inputs
    # input_ids: [batch, seq_len]
    # position_ids: [batch, seq_len]
    # kv_caches: list of [batch, 2, num_kv, capacity, head_dim]
    # ctx_len: [batch] - assume batch size 1 for simplicity or match batch dim
    
    # kv_caches is a list of tensors with static shapes (fixed capacity)
    kv_cache_dynamics = [{}] * len(kv_caches)
    
    dynamic_shapes = {
        "input_ids": {1: seq_len_dim},
        "position_ids": {1: seq_len_dim},
        "kv_caches": kv_cache_dynamics,
        "ctx_len": {} # Assuming fixed batch size=1 for this test
    }
    
    # We might need strict=False
    ep = torch.export.export(
        model,
        args=(input_ids, position_ids, kv_caches, ctx_len),
        dynamic_shapes=dynamic_shapes,
        strict=False
    )
    
    print("Compiling with torch_tensorrt...")
    trt_model = torch_tensorrt.dynamo.compile(
        ep,
        inputs=[input_ids, position_ids, kv_caches, ctx_len],
        enabled_precisions={torch.float32}, # Using explicit typing
        use_explicit_typing=True,
        use_fp32_acc=True,
        device=DEVICE,
        disable_tf32=True,
        min_block_size=1
    )
    
    return trt_model

# %%
# Generation Utils
# ----------------

def generate(model_func, tokenizer, prompt, max_new_tokens=20):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    # Initialize KV caches
    # [batch, 2, num_kv_heads, capacity, head_dim]
    config = AutoConfig.from_pretrained(MODEL_NAME)
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    
    kv_caches = [
        torch.zeros(1, 2, num_kv_heads, MAX_SEQ_LEN, head_dim, dtype=DTYPE, device=DEVICE)
        for _ in range(num_layers)
    ]
    
    print(f"\nPrompt: '{prompt}'")
    print("Generating...")
    
    generated_ids = input_ids
    
    # 1. Prefill
    print(f"\n[Step 1: Prefill] Input seq_len={seq_len} (>1) -> Expecting FMHA Kernel")
    ctx_len = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE)
    logits, kv_caches = model_func(input_ids, position_ids, kv_caches, ctx_len)

    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
    generated_ids = torch.cat([generated_ids, next_token], dim=1)
    
    # 2. Decode
    cur_pos = seq_len
    print(f"\n[Step 2: Decode] Starting generation loop (seq_len=1) -> Expecting XQA Kernel")
    
    for _ in range(max_new_tokens - 1):
        input_ids = next_token
        position_ids = torch.tensor([[cur_pos]], dtype=torch.long, device=DEVICE)
        
        # For decode, input seq_len is 1, but context length is total length
        ctx_len = torch.tensor([cur_pos + 1], dtype=torch.int32, device=DEVICE)
        
        logits, kv_caches = model_func(input_ids, position_ids, kv_caches, ctx_len)

        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        cur_pos += 1
        
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

def generate_reference(model, tokenizer, prompt, max_new_tokens=20):
    """
    Mimic run_llm.py's generation logic (greedy, no cache, full context re-eval)
    to ensure PyTorch reference output matches run_llm.py.
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    generated_ids = input_ids
    
    for _ in range(max_new_tokens):
        # Create position_ids manually
        current_seq_len = generated_ids.shape[1]
        position_ids = torch.arange(current_seq_len, dtype=torch.long, device=DEVICE).unsqueeze(0)
        
        # Forward pass (full context, no cache)
        outputs = model(generated_ids, position_ids=position_ids, use_cache=False)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# %%
# Main Execution
# --------------

if __name__ == "__main__":
    torch.manual_seed(42)
    
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    
    # We use full model layers this time
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=DTYPE,
        attn_implementation="sdpa"
    ).to(DEVICE)
    
    # 1. PyTorch Reference Generation
    print("\n=== PyTorch Reference Generation ===")
    prompt = "What is parallel programming?"
    
    with torch.no_grad():
        pyt_text = generate_reference(model, tokenizer, prompt, max_new_tokens=30)
    
    print(f"PyTorch Output: {pyt_text}")
    
    # 2. Prepare for Plugin
    print("\n=== Preparing Plugin Model ===")
    model = replace_attention(model, config)
    wrapper = Qwen2Wrapper(model)
    
    # 3. Compile
    # Create dummy inputs for compilation
    dummy_input_ids = torch.tensor([[1, 2, 3]], device=DEVICE)
    dummy_pos_ids = torch.tensor([[0, 1, 2]], device=DEVICE)
    dummy_ctx_len = torch.tensor([3], dtype=torch.int32, device=DEVICE)
    dummy_kvs = [
        torch.zeros(1, 2, config.num_key_value_heads, MAX_SEQ_LEN, config.hidden_size // config.num_attention_heads, 
                    dtype=DTYPE, device=DEVICE)
        for _ in range(config.num_hidden_layers)
    ]
    
    trt_model = compile_model(wrapper, dummy_input_ids, dummy_pos_ids, dummy_kvs, dummy_ctx_len)
    
    # 4. TensorRT Generation
    trt_text = generate(trt_model, tokenizer, prompt, max_new_tokens=30)
    print("\n=== TensorRT Plugin Generation ===")
    print(f"TensorRT Output: {trt_text}")
    
    # 5. Comparison
    print("\n=== Comparison ===")
    if pyt_text == trt_text:
        print("SUCCESS: Outputs match exactly!")
    else:
        print("FAILURE: Outputs differ.")
        print(f"PyTorch:  {pyt_text}")
        print(f"TensorRT: {trt_text}")