"""
.. _shared_dynamic_dims:

Sharing Dynamic Dimensions Across Inputs
==========================================================

When a model takes multiple inputs whose dynamic axes must be **equal at
runtime** — for example, HuggingFace-style encoders where ``input_ids`` and
``attention_mask`` are both shaped ``[batch, seq_len]`` — naively assigning an
independent dynamic dimension to each input causes ``torch.export`` to raise a
``ConstraintViolationError``.  The exporter detects that the two independent
symbols are forced equal by the model's forward pass (e.g. a broadcast) and
rejects the export.

``torch_tensorrt.Input(shared_dims={axis: name})`` solves this: axes that share
the same name across inputs are exported as a single ``torch.export.Dim``, so
the equality constraint is satisfied automatically.  All dynamic-shape intent
lives on the ``Input`` objects — no separate ``dynamic_shapes`` argument or
``torch.export`` knowledge is required at the call site.
"""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import torch
import torch.nn as nn
import torch_tensorrt
from torch_tensorrt.dynamo.utils import COSINE_THRESHOLD, cosine_similarity

# %%
# Define a HuggingFace-style encoder whose two inputs share the batch axis.
# The ``embed * mask`` broadcast forces ``input_ids.shape[0] ==
# attention_mask.shape[0]`` at every forward call — exactly the pattern that
# triggers ``ConstraintViolationError`` when the batch axis is exported as two
# independent ``Dim`` objects.


class SharedDimEncoder(nn.Module):
    def __init__(self, vocab: int = 1024, hidden: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        x = self.embed(input_ids)  # [B, S, hidden]
        mask = attention_mask.unsqueeze(-1).to(x.dtype)  # [B, S, 1]
        return self.proj(x * mask)  # [B, S, hidden]


model = SharedDimEncoder().cuda().eval()

# %%
# Without ``shared_dims`` — raises ``ConstraintViolationError``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Using independent ``Input`` objects like this would fail at export time:
#
# .. code-block:: python
#
#     inputs = [
#         torch_tensorrt.Input(min_shape=(1,16), opt_shape=(4,16), max_shape=(8,16), dtype=torch.int64),
#         torch_tensorrt.Input(min_shape=(1,16), opt_shape=(4,16), max_shape=(8,16), dtype=torch.int64),
#     ]
#     # torch.export mints independent symbols s0, s1 for the batch axis of
#     # each input.  The broadcast forces Eq(s0, s1), which the exporter rejects.
#
# %%
# With ``shared_dims`` — correct approach (positional inputs)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Annotate the batch axis (axis 0) with the same name ``"B"`` on both inputs.
# Torch-TensorRT creates a single shared ``torch.export.Dim("B")`` for that
# axis so the equality constraint is satisfied up front.

inputs = [
    torch_tensorrt.Input(
        min_shape=(1, 16),
        opt_shape=(4, 16),
        max_shape=(8, 16),
        dtype=torch.int64,
        name="input_ids",
        shared_dims={0: "B"},
    ),
    torch_tensorrt.Input(
        min_shape=(1, 16),
        opt_shape=(4, 16),
        max_shape=(8, 16),
        dtype=torch.int64,
        name="attention_mask",
        shared_dims={0: "B"},
    ),
]

trt_model = torch_tensorrt.compile(
    model,
    ir="dynamo",
    inputs=inputs,
    min_block_size=1,
    cache_built_engines=False,
    reuse_cached_engines=False,
)

# %%
# Verify correctness at multiple batch sizes within the declared range.

for batch_size in (4, 2, 1):
    ids = torch.randint(0, 1024, (batch_size, 16), dtype=torch.int64, device="cuda")
    mask = torch.ones((batch_size, 16), dtype=torch.int64, device="cuda")

    with torch.no_grad():
        ref = model(ids, mask)
        out = trt_model(ids, mask)

    cos_sim = cosine_similarity(ref, out)
    assert (
        cos_sim > COSINE_THRESHOLD
    ), f"Numerical mismatch at batch_size={batch_size}: cos_sim={cos_sim:.4f}"
    print(f"batch_size={batch_size}  cos_sim={cos_sim:.6f}  ✓")

# %%
# With ``shared_dims`` — kwarg inputs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The same feature works with ``kwarg_inputs``, which is the natural form for
# HuggingFace models whose ``forward`` signature uses keyword arguments.

kwarg_inputs = {
    "input_ids": torch_tensorrt.Input(
        min_shape=(1, 16),
        opt_shape=(4, 16),
        max_shape=(8, 16),
        dtype=torch.int64,
        name="input_ids",
        shared_dims={0: "B"},
    ),
    "attention_mask": torch_tensorrt.Input(
        min_shape=(1, 16),
        opt_shape=(4, 16),
        max_shape=(8, 16),
        dtype=torch.int64,
        name="attention_mask",
        shared_dims={0: "B"},
    ),
}

trt_model_kwargs = torch_tensorrt.compile(
    model,
    ir="dynamo",
    kwarg_inputs=kwarg_inputs,
    min_block_size=1,
    cache_built_engines=False,
    reuse_cached_engines=False,
)

ids = torch.randint(0, 1024, (4, 16), dtype=torch.int64, device="cuda")
mask = torch.ones((4, 16), dtype=torch.int64, device="cuda")

with torch.no_grad():
    ref = model(input_ids=ids, attention_mask=mask)
    out = trt_model_kwargs(input_ids=ids, attention_mask=mask)

cos_sim = cosine_similarity(ref, out)
assert cos_sim > COSINE_THRESHOLD, f"kwarg path mismatch: cos_sim={cos_sim:.4f}"
print(f"kwarg_inputs path  cos_sim={cos_sim:.6f}  ✓")

# %%
# Sharing multiple axes
# ^^^^^^^^^^^^^^^^^^^^^^
#
# If both batch and sequence length are dynamic and must be shared, annotate
# both axes on each input:
#
# .. code-block:: python
#
#     shared_dims={0: "B", 1: "S"}
#
# The same name on the same axis across different inputs produces one shared
# ``Dim``; different names on different axes produce independent ``Dim``\s.

print("\nAll checks passed.")
