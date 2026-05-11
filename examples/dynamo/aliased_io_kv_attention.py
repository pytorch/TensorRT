"""
.. _aliased_io_kv_attention_example:

Streaming attention with a static KV cache
============================================

A realistic single-layer transformer attention block with a static-shape
KV cache held as module buffers. This is the canonical PyTorch pattern
for streaming decoder inference: at each step the module takes a single
token's hidden state, projects K and V, writes them into a fixed-size
cache, and attends over the cache.

Compared to the simpler aliased-I/O examples this one exercises:

* ``LayerNorm``, ``Linear`` projections, ``scaled_dot_product_attention``
* multi-head reshapes / transposes around the cache writes
* the ``register_buffer`` + slice-write pattern for both K and V

The compiled engine emits two ``IKVCacheUpdateLayer`` ops (one each for
K and V) with aliased outputs. The C++ runtime writes the new K/V
directly into the buffer storage; the next step's attention reads the
updated cache without any copy.
"""

# %%
# Imports
# -------
import torch
import torch.nn.functional as F
import torch_tensorrt
from torch.export import export

# %%
# Single-layer attention block
# ----------------------------
# The model takes one timestep at a time and uses a compile-time
# constant ``write_pos`` for the cache slot. In a real generation loop
# you'd vary ``write_pos`` per step; a few practical recipes:
#
# * Recompile once per ``write_pos`` (cheap with engine caching).
# * Bake ``write_pos`` into the model's state via a buffer and increment
#   it inside ``forward`` (requires an extra in-place op pattern we
#   support separately).
# * Use the lower-level ``user_inputs`` flow and pass ``write_pos`` as
#   an integer argument to a wrapper that selects between pre-compiled
#   engines.


class StaticKVAttention(torch.nn.Module):
    def __init__(self, batch=1, max_seq=64, n_heads=4, head_dim=16, write_pos=3):
        super().__init__()
        self.batch = batch
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.hidden = n_heads * head_dim
        self.write_pos = write_pos

        # Static KV cache — fixed shape across the whole generation.
        self.register_buffer("cache_k", torch.zeros(batch, n_heads, max_seq, head_dim))
        self.register_buffer("cache_v", torch.zeros(batch, n_heads, max_seq, head_dim))

        self.norm = torch.nn.LayerNorm(self.hidden)
        self.q_proj = torch.nn.Linear(self.hidden, self.hidden, bias=False)
        self.k_proj = torch.nn.Linear(self.hidden, self.hidden, bias=False)
        self.v_proj = torch.nn.Linear(self.hidden, self.hidden, bias=False)
        self.o_proj = torch.nn.Linear(self.hidden, self.hidden, bias=False)

    def forward(self, hidden_states):
        # hidden_states: [B, 1, H]
        B = hidden_states.shape[0]
        h = self.norm(hidden_states)

        q = self.q_proj(h).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)

        # In-place writes into the static KV cache. These two lines are
        # what trigger the KV-cache aliasing fast path.
        self.cache_k[:, :, self.write_pos : self.write_pos + 1, :] = k
        self.cache_v[:, :, self.write_pos : self.write_pos + 1, :] = v

        attn_out = F.scaled_dot_product_attention(q, self.cache_k, self.cache_v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, self.hidden)
        return self.o_proj(attn_out)


# %%
# Compile
# -------
torch.manual_seed(0)
model = StaticKVAttention().cuda()
hidden = torch.randn(1, 1, model.hidden, device="cuda")

ep = export(model, (hidden.clone(),))
compiled = torch_tensorrt.compile(
    ep,
    ir="dynamo",
    inputs=[hidden.clone()],
    enabled_precisions={torch.float32},
    min_block_size=1,
    use_python_runtime=False,
)


# %%
# Inspect aliasing
# ----------------
for _, mod in compiled.named_modules():
    if hasattr(mod, "aliased_io") and mod.aliased_io:
        print(f"Engine input bindings: {list(mod.input_binding_names)}")
        print(f"Engine output bindings: {list(mod.output_binding_names)}")
        for out, (inp, kind) in mod.aliased_io.items():
            print(f"  {out}  <-aliased->  {inp}  (kind={kind})")


# %%
# Numerical check against eager
# ------------------------------
eager_model = StaticKVAttention().cuda()
eager_model.load_state_dict(model.state_dict())
eager_out = eager_model(hidden.clone())

# Reset the compiled cache to match eager's fresh state
compiled.cache_k.zero_()
compiled.cache_v.zero_()
compiled_out = compiled(hidden.clone())
compiled_val = compiled_out[0] if isinstance(compiled_out, tuple) else compiled_out

print(f"\nmax output diff: {(compiled_val - eager_out).abs().max().item():.6f}")
print(
    f"cache_k matches eager: {torch.allclose(compiled.cache_k, eager_model.cache_k, atol=1e-4)}"
)
print(
    f"cache_v matches eager: {torch.allclose(compiled.cache_v, eager_model.cache_v, atol=1e-4)}"
)


# %%
# Streaming inference loop
# ------------------------
# Each call writes a new K/V at the compiled ``write_pos`` slot. In a
# real decoder you'd rotate the write position per step; this example
# just demonstrates that the cache state persists.

compiled.cache_k.zero_()
compiled.cache_v.zero_()
for step in range(3):
    h = torch.randn(1, 1, model.hidden, device="cuda")
    out = compiled(h)
    print(
        f"step {step}: cache_k.norm()={compiled.cache_k.norm().item():.4f}, "
        f"out.norm()={(out[0] if isinstance(out, tuple) else out).norm().item():.4f}"
    )
