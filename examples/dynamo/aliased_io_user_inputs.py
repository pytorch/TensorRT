"""
.. _aliased_io_user_inputs_example:

In-place aliased I/O: caller-owned tensors
============================================

This example shows the simplest in-place pattern: the caller owns a buffer
(e.g. a KV cache), passes it into the compiled module on every call, and
the TensorRT engine mutates it in place. No fresh allocation per call, no
post-engine copy.

Mechanically:

* The model writes into a slice of one of its inputs:
  ``cache[:, :, t:t+1, :] = update``.
* The converter recognizes the ``slice_scatter`` pattern and emits a
  ``IKVCacheUpdateLayer`` whose output is aliased to the cache input.
* The C++ runtime binds the aliased output to the input's ``data_ptr``,
  skipping ``at::empty``. The engine writes through that pointer directly
  into the caller's tensor storage.
* The user-facing return tuple contains only the model's explicit outputs
  — the aliased "mutation output" is invisible to the caller.

Constraints (from the TensorRT operator):

* Cache shape must be 4-D ``[batch, heads, max_seq, head_dim]`` and fully
  static.
* Write dimension must be ``2`` (the sequence axis).
* ``write_start + update_len <= max_seq``.

If the constraints aren't met the converter silently falls back to a
regular scatter and aliasing isn't established (correctness is still
preserved, but no in-place benefit).
"""

# %%
# Imports
# -------
import torch
import torch_tensorrt
from torch.export import export

# %%
# Model
# -----
# A trivial "step" that writes one timestep into the cache and returns a
# summary statistic. The interesting line is ``cache[:, :, 3:4, :] = update``
# — that single slice-write is what triggers the KV-cache fast path.


class KVStep(torch.nn.Module):
    def forward(self, cache, update):
        cache[:, :, 3:4, :] = update
        return cache.sum()


# %%
# Compile
# -------
B, H, S_MAX, D = 1, 4, 16, 8
cache_proto = torch.zeros(B, H, S_MAX, D, device="cuda")
update_proto = torch.ones(B, H, 1, D, device="cuda")

ep = export(KVStep().cuda(), (cache_proto.clone(), update_proto.clone()))
compiled = torch_tensorrt.compile(
    ep,
    ir="dynamo",
    inputs=[cache_proto.clone(), update_proto.clone()],
    enabled_precisions={torch.float32},
    min_block_size=1,
    use_python_runtime=False,  # aliased I/O requires the C++ runtime
)

# %%
# Verify aliasing was established
# --------------------------------
# Each compiled engine carries an ``aliased_io`` map of output binding
# name -> ``(input_binding_name, kind)``. ``kind`` is "kv_cache_update"
# when TensorRT itself enforces the alias (via ``IKVCacheUpdateLayer``).

for _, mod in compiled.named_modules():
    if hasattr(mod, "aliased_io") and mod.aliased_io:
        for out, (inp, kind) in mod.aliased_io.items():
            print(f"  {out}  <-aliased->  {inp}  (kind={kind})")


# %%
# Run: cache mutates in place
# ---------------------------
# The caller owns ``cache``. After ``compiled(cache, update)`` returns,
# ``cache`` has been mutated; ``id(cache)`` and ``cache.data_ptr()`` are
# the same as before the call. The return value is just the model's
# explicit output (the sum).

cache = torch.zeros(B, H, S_MAX, D, device="cuda")
update = torch.ones(B, H, 1, D, device="cuda") * 7.0
cache_id_before = id(cache)
cache_ptr_before = cache.data_ptr()

returned = compiled(cache, update)

print(f"\nreturned: {returned.item()}  (expected {7.0 * H * D})")
print(f"cache.sum(): {cache.sum().item()}  (expected {7.0 * H * D})")
print(f"id preserved: {id(cache) == cache_id_before}")
print(f"data_ptr preserved: {cache.data_ptr() == cache_ptr_before}")


# %%
# Streaming: repeated calls accumulate state
# -------------------------------------------
# Because the caller's cache tensor identity is preserved and the engine
# writes in place, each call sees the result of the previous one.

cache = torch.zeros(B, H, S_MAX, D, device="cuda")
for step, scale in enumerate([1.0, 5.0, 0.0]):
    compiled(cache, torch.ones(B, H, 1, D, device="cuda") * scale)
    print(f"step {step}: cache.sum() = {cache.sum().item()}")
