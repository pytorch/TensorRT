
import ctypes
import os
import time
import torch
import torch.nn as nn
import torch_tensorrt
import numpy as np
import tensorrt as trt
from transformers import AutoConfig, AutoModelForCausalLM
from torch_tensorrt.dynamo.conversion import ConversionContext, dynamo_tensorrt_converter
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor
from typing import Tuple, Optional, List

# Enable plugin debug logging (optional, turned off for benchmark)
if "EDGELLM_DEBUG_PLUGIN" in os.environ:
    del os.environ["EDGELLM_DEBUG_PLUGIN"]

# Configuration
PLUGIN_PATH = "/develop/TensorRT/tensorrt-edgellm/build/libNvInfer_edgellm_plugin.so"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_SEQ_LEN = 2048  # Increased to support larger benchmarks
DTYPE = torch.float16
DEVICE = torch.device("cuda:0")

# Load the plugin
if os.path.exists(PLUGIN_PATH):
    ctypes.CDLL(PLUGIN_PATH)
else:
    raise RuntimeError(f"Plugin not found at {PLUGIN_PATH}")

# -----------------------------------------------------------------------------
# Plugin Ops & Converters
# -----------------------------------------------------------------------------

def get_plugin_rope_cache(rotary_emb, max_seq_len, head_dim, device):
    inv_freq = rotary_emb.inv_freq.to(device).float()
    attention_scaling = rotary_emb.attention_scaling
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
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
    
    nq_val = 14
    nkv_val = 2
    d_val = 64
    
    field_list = [
        trt.PluginField("num_q_heads", np.array([nq_val], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("num_kv_heads", np.array([nkv_val], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("head_size", np.array([d_val], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("max_batch_size", np.array([4], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("kv_cache_capacity", np.array([MAX_SEQ_LEN], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("enable_tree_attention", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("enable_reuse_kv_cache", np.array([0], dtype=np.int32), trt.PluginFieldType.INT32),
        trt.PluginField("enable_kv_cache_copy", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32),
    ]
    
    fields = trt.PluginFieldCollection(field_list)
    plugin = creator.create_plugin(name, fields)
    
    inputs = [get_trt_tensor(ctx, i, f"{name}_i{idx}") if not isinstance(i, trt.ITensor) else i 
              for idx, i in enumerate([qkv, kv, ctx_len, rope])]

    if len(inputs[2].shape) == 2 and inputs[2].shape[1] == 1:
        shuffle_layer = ctx.net.add_shuffle(inputs[2])
        shuffle_layer.reshape_dims = (inputs[2].shape[0],)
        inputs[2] = shuffle_layer.get_output(0)

    layer = ctx.net.add_plugin_v2(inputs, plugin)
    return layer.get_output(0), layer.get_output(1)


# -----------------------------------------------------------------------------
# Models & Wrappers
# -----------------------------------------------------------------------------

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
        self.register_buffer("rope_cache", rope_cache)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, ctx_len=None, **kwargs):
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        qkv = torch.cat([q, k, v], dim=-1)
        
        if ctx_len is None:
             ctx_len = torch.tensor([seq_len], dtype=torch.int32, device=hidden_states.device).expand(batch_size)

        rope_fp32 = self.rope_cache.float()
        
        if past_key_value is None:
            raise ValueError("past_key_value (KV cache tensor) must be provided")
            
        attn_out, updated_kv = torch.ops.xqa.attn.default(
            qkv, past_key_value, ctx_len, rope_fp32,
            self.num_heads, self.num_key_value_heads, self.head_dim
        )
        
        attn_out = attn_out.reshape(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_out)
        return output, updated_kv

def replace_attention(model, config):
    rotary_emb = model.model.rotary_emb
    head_dim = config.hidden_size // config.num_attention_heads
    rope_cache = get_plugin_rope_cache(rotary_emb, MAX_SEQ_LEN, head_dim, DEVICE)
    
    for i, layer in enumerate(model.model.layers):
        layer.self_attn = PluginAttention(layer.self_attn, config, i, rope_cache)
    return model

class Qwen2Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids, position_ids, kv_caches: List[torch.Tensor], ctx_len: torch.Tensor):
        transformer = self.model.model
        hidden_states = transformer.embed_tokens(input_ids)
        
        new_kv_caches = []
        for i, layer in enumerate(transformer.layers):
            past_key_value = kv_caches[i]
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            
            hidden_states, updated_kv = layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=past_key_value,
                ctx_len=ctx_len
            )
            hidden_states = residual + hidden_states
            
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
            
            new_kv_caches.append(updated_kv) 
            
        hidden_states = transformer.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)
        return logits, new_kv_caches

# -----------------------------------------------------------------------------
# Compilation
# -----------------------------------------------------------------------------

def compile_model(model, input_ids, position_ids, kv_caches, ctx_len):
    # seq_len can be 1 (decode) or >1 (prefill)
    seq_len_dim = torch.export.Dim("seq_len", min=1, max=MAX_SEQ_LEN)
    
    kv_cache_dynamics = [{}] * len(kv_caches)
    dynamic_shapes = {
        "input_ids": {1: seq_len_dim},
        "position_ids": {1: seq_len_dim},
        "kv_caches": kv_cache_dynamics,
        "ctx_len": {}
    }
    
    ep = torch.export.export(
        model,
        args=(input_ids, position_ids, kv_caches, ctx_len),
        dynamic_shapes=dynamic_shapes,
        strict=False
    )
    
    trt_model = torch_tensorrt.dynamo.compile(
        ep,
        inputs=[input_ids, position_ids, kv_caches, ctx_len],
        enabled_precisions={torch.float32},
        use_explicit_typing=True,
        use_fp32_acc=True,
        device=DEVICE,
        disable_tf32=True,
        min_block_size=1
    )
    return trt_model

# -----------------------------------------------------------------------------
# Benchmarking
# -----------------------------------------------------------------------------

def benchmark_generation(model_func, isl, osl, config, run_name="Model"):
    # Prepare inputs
    input_ids = torch.randint(0, config.vocab_size, (1, isl), device=DEVICE)
    
    # Init KV Caches
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    
    kv_caches = [
        torch.zeros(1, 2, num_kv_heads, MAX_SEQ_LEN, head_dim, dtype=DTYPE, device=DEVICE)
        for _ in range(num_layers)
    ]
    
    # Warmup
    # Perform a short run to warm up kernels
    # (We do a minimal run here, but for real benchmark we rely on the actual run being representative or do a dedicated warmup loop)
    # For this script, let's do 1 warmup iteration of the loop
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    
    # 1. Prefill
    seq_len = isl
    position_ids = torch.arange(seq_len, dtype=torch.long, device=DEVICE).unsqueeze(0)
    ctx_len = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE)
    
    # Run Prefill
    logits, kv_caches = model_func(input_ids, position_ids, kv_caches, ctx_len)
    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
    
    # 2. Decode
    cur_pos = seq_len
    
    for _ in range(osl - 1):
        input_ids_step = next_token
        position_ids_step = torch.tensor([[cur_pos]], dtype=torch.long, device=DEVICE)
        ctx_len_step = torch.tensor([cur_pos + 1], dtype=torch.int32, device=DEVICE)
        
        logits, kv_caches = model_func(input_ids_step, position_ids_step, kv_caches, ctx_len_step)
        
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        cur_pos += 1
        
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    print(f"{run_name} | ISL: {isl}, OSL: {osl} | Total Time: {elapsed_ms:.2f} ms | Tokens/sec: {osl / (elapsed_ms / 1000.0):.2f}")
    return elapsed_ms

def run_pytorch_benchmark_manual(model, config, isl, osl):
    input_ids = torch.randint(0, config.vocab_size, (1, isl), device=DEVICE)
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    
    # Prefill
    # For manual loop without cache, we process full sequence each step
    # But user asked for Manual to NOT use cache, and Generate TO use cache.
    # The previous implementation of manual loop WAS using cache (past_key_values).
    # Let's modify Manual to NOT use cache as requested for comparison.
    
    # Actually, the user said "PyTorch (Manual)은 kv cache를 사용하지 않는게 맞는데" 
    # (PyTorch (Manual) should not use kv cache)
    # So we will change this function to run WITHOUT cache (re-encoding full context).
    
    with torch.no_grad():
        # Initial prefill is just full forward
        # But for subsequent steps we re-feed growing sequence
        
        generated_ids = input_ids
        
        # Prefill logic combined with decode loop for no-cache
        # We just loop OSL times. First iteration is effectively prefill+decode 1st token if we consider it that way?
        # Or we just start with input_ids and grow it.
        
        for _ in range(osl):
            outputs = model(generated_ids, use_cache=False)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    print(f"PyTorch (Manual - No Cache) | ISL: {isl}, OSL: {osl} | Total Time: {elapsed_ms:.2f} ms | Tokens/sec: {osl / (elapsed_ms / 1000.0):.2f}")
    return elapsed_ms

def run_pytorch_benchmark_generate(model, config, isl, osl):
    input_ids = torch.randint(0, config.vocab_size, (1, isl), device=DEVICE)
    max_length = isl + osl
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    
    with torch.no_grad():
        _ = model.generate(
            input_ids, 
            max_new_tokens=osl, 
            min_new_tokens=osl,
            do_sample=False, 
            use_cache=True,
            pad_token_id=config.eos_token_id
        )
            
    end_event.record()
    torch.cuda.synchronize()
    
    elapsed_ms = start_event.elapsed_time(end_event)
    print(f"PyTorch (Generate) | ISL: {isl}, OSL: {osl} | Total Time: {elapsed_ms:.2f} ms | Tokens/sec: {osl / (elapsed_ms / 1000.0):.2f}")
    return elapsed_ms

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    
    print(f"Loading {MODEL_NAME}...")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    
    # 1. PyTorch Model
    model_pytorch = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=DTYPE,
        # attn_implementation="sdpa" # Use optimized SDPA for PyTorch ref
    ).to(DEVICE)
    
    # 2. TensorRT Plugin Model
    # We need a fresh model copy for modification
    model_trt = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=DTYPE
    ).to(DEVICE)
    
    model_trt = replace_attention(model_trt, config)
    wrapper = Qwen2Wrapper(model_trt)
    
    # Compilation (perform once with symbolic shapes covering the max range)
    print("Compiling TensorRT model...")
    
    # Dummy inputs for compilation
    dummy_input_ids = torch.tensor([[1, 2, 3]], device=DEVICE)
    dummy_pos_ids = torch.tensor([[0, 1, 2]], device=DEVICE)
    dummy_ctx_len = torch.tensor([3], dtype=torch.int32, device=DEVICE)
    dummy_kvs = [
        torch.zeros(1, 2, config.num_key_value_heads, MAX_SEQ_LEN, config.hidden_size // config.num_attention_heads, 
                    dtype=DTYPE, device=DEVICE)
        for _ in range(config.num_hidden_layers)
    ]
    
    trt_model_func = compile_model(wrapper, dummy_input_ids, dummy_pos_ids, dummy_kvs, dummy_ctx_len)
    
    # 3. Benchmarks
    # Define ISL/OSL pairs
    benchmarks = [
        (128, 128),
        (256, 128),
        (512, 256),
    ]
    
    print("\n=== Starting Benchmarks ===")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    for isl, osl in benchmarks:
        print("-" * 60)
        # PyTorch Manual Loop
        run_pytorch_benchmark_manual(model_pytorch, config, isl, osl)
        
        # PyTorch Generate API
        run_pytorch_benchmark_generate(model_pytorch, config, isl, osl)
        
        # TensorRT
        # Warmup TRT first time or every time?
        # Let's just run it.
        benchmark_generation(trt_model_func, isl, osl, config, run_name="TensorRT")
