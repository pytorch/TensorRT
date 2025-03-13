from transformers import LlamaConfig, LlamaForCausalLM
import torch
import torch_tensorrt


# 1. Create a custom config with 1 layer
config = LlamaConfig(
    vocab_size=32000,
    hidden_size=4096,        # LLaMA2-7B dimensions
    intermediate_size=11008, # FFN hidden_dim = 4 * 4096 * 0.7 (SwiGLU scaling)
    num_hidden_layers=1,     # Only 1 decoder layer
    num_attention_heads=32,
    max_position_embeddings=4096,
    use_cache=False,         # Disable KV caching for export
)

# 2. Initialize model (random weights)
with torch.no_grad():
    model = (LlamaForCausalLM(config).eval().half())

# 3. Export with static shapes
input_ids = torch.randint(0, 32000, (1, 64))  # Static [batch=1, seq=64]
exported = torch.export.export(
    model,
    (input_ids,),
    dynamic_shapes=None,  # Fully static
)

# Test forward pass
input_ids = torch.randint(0, 32000, (1, 64))
output = model(input_ids)
print(output)  # Should be [1, 64, 32000]

# Export validation
# print(exported.graph_module.code)

DEVICE = torch.device("cuda:0")

with torch_tensorrt.logging.debug():
    trt_model = torch_tensorrt.dynamo.compile(
        exported,
        inputs=[input_ids],
        enabled_precisions={torch.float32, torch.float16},
        truncate_double=True,
        device=DEVICE,
        disable_tf32=True,
        use_explicit_typing=False,
        use_fp32_acc=True,
        # debug=True,
    )

input_ids = input_ids.to(DEVICE)

res = trt_model.forward(input_ids)
print(res)