"""
.. _aliased_io_buffers_example:

In-place aliased I/O: module-owned buffers
==================================================

This is the PyTorch-canonical pattern for streaming inference: the cache
lives inside the model via ``register_buffer``. The user simply calls
``model(x)`` — no need to thread the cache through manually.

How it flows through the compile pipeline:

* ``torch.export`` captures ``cache`` as a ``BUFFER`` input in the
  ``graph_signature``; mutations to it become ``BUFFER_MUTATION``
  output specs.
* ``ExportedProgram.module()`` rewrites the buffer into a ``get_attr``
  node plus a trailing ``aten.copy_(get_attr_buffer, slice_scatter_result)``
  that represents the mutation.
* Torch-TensorRT's ``lift_mutated_buffers`` pre-compile pass detects that
  trailing ``copy_``, lifts the ``get_attr`` to a ``placeholder``, and
  rebuilds the GraphModule so the engine treats the buffer as a regular
  input binding.
* The slice-scatter converter sees the cache as a network input and emits
  ``IKVCacheUpdateLayer`` with aliased I/O.
* The compiled result's lifted-buffer placeholders are rewritten in place
  to ``get_attr`` reads from registered ``nn.Module`` buffers (via
  ``inline_lifted_buffers_into_gm``). The buffer state lives on the
  compiled module itself; ``forward`` takes only user inputs. Because the
  result is a plain ``fx.GraphModule`` with buffers, it serializes through
  ``torch_tensorrt.save`` / ``torch.export`` without any external wrapper.

The net effect: the engine writes through the buffer's storage in place,
and the next call sees the updated state. No copy-back, no allocation
per call.
"""

# %%
# Imports
# -------
import torch
import torch_tensorrt
from torch.export import export

# %%
# Model with a buffer-backed KV cache
# -----------------------------------
# Two caches (K and V), both held as buffers. Each forward call writes one
# timestep into position 3 of each cache.


class StreamingKV(torch.nn.Module):
    def __init__(self, b=1, h=4, s_max=16, d=8):
        super().__init__()
        self.register_buffer("cache_k", torch.zeros(b, h, s_max, d))
        self.register_buffer("cache_v", torch.zeros(b, h, s_max, d))

    def forward(self, x_k, x_v):
        self.cache_k[:, :, 3:4, :] = x_k
        self.cache_v[:, :, 3:4, :] = x_v
        return self.cache_k.sum() + self.cache_v.sum()


# %%
# Compile
# -------
model = StreamingKV().cuda()
x_k = torch.ones(1, 4, 1, 8, device="cuda") * 3.0
x_v = torch.ones(1, 4, 1, 8, device="cuda") * 5.0

ep = export(model, (x_k.clone(), x_v.clone()))

# Show how torch.export sees the model:
print("graph_signature.input_specs:")
for s in ep.graph_signature.input_specs:
    print(f"  {s}")
print("graph_signature.output_specs:")
for s in ep.graph_signature.output_specs:
    print(f"  {s}")

compiled = torch_tensorrt.compile(
    ep,
    ir="dynamo",
    inputs=[x_k.clone(), x_v.clone()],
    enabled_precisions={torch.float32},
    min_block_size=1,
    use_python_runtime=False,
)


# %%
# Verify aliasing was established for both caches
# ------------------------------------------------
for _, mod in compiled.named_modules():
    if hasattr(mod, "aliased_io") and mod.aliased_io:
        print(f"\nEngine input bindings: {list(mod.input_binding_names)}")
        for out, (inp, kind) in mod.aliased_io.items():
            print(f"  {out}  <-aliased->  {inp}  (kind={kind})")


# %%
# Run: the compiled module owns the cache buffers
# ------------------------------------------------
# The user calls ``compiled(x_k, x_v)`` — same signature as the original
# model. The buffers are owned by the wrapping module and threaded into
# the engine automatically. After the call, ``compiled.cache_k`` and
# ``compiled.cache_v`` reflect the mutation.

returned = compiled(x_k, x_v)
returned_val = returned[0] if isinstance(returned, tuple) else returned

# Eager reference for comparison.
eager = StreamingKV().cuda()
eager_returned = eager(x_k.clone(), x_v.clone())

print(f"\nreturn matches eager:    {torch.allclose(returned_val, eager_returned)}")
print(f"cache_k matches eager:   {torch.allclose(compiled.cache_k, eager.cache_k)}")
print(f"cache_v matches eager:   {torch.allclose(compiled.cache_v, eager.cache_v)}")


# %%
# Streaming: module-held state persists across calls
# ---------------------------------------------------
# Reset the cache and step through three updates. Each call mutates the
# module's buffer state in place; the next call sees the updated value.

compiled.cache_k.zero_()
compiled.cache_v.zero_()
for step, val in enumerate([1.0, 5.0, 0.0]):
    x = torch.ones(1, 4, 1, 8, device="cuda") * val
    compiled(x, x)
    print(
        f"step {step}: cache_k.sum()={compiled.cache_k.sum().item():.1f}, "
        f"cache_v.sum()={compiled.cache_v.sum().item():.1f}"
    )
