"""
.. _torch_export_alpamayo_fp8_woq:

Compiling Alpamayo 1.5 with TorchAO FP8 weight-only quantization
================================================================

This example loads
`NVIDIA Alpamayo 1.5 10B <https://huggingface.co/nvidia/Alpamayo-1.5-10B>`_,
extracts its diffusion expert path, applies TorchAO
``Float8WeightOnlyConfig``, and compiles it with the Torch-TensorRT Dynamo
backend.

Alpamayo's complete trajectory rollout contains Hugging Face autoregressive
generation, mutable KV caches, and a Python-controlled diffusion loop. Those
parts cannot be represented by one static ``torch.export`` graph. This example
therefore compiles the cache-independent expert graph:
``action_in_proj -> expert -> action_out_proj``. Integrating a compiled expert
with the VLM prefix KV cache is outside the scope of this example.

Weight-only FP8 keeps activations in BF16 and stores Linear weights as FP8 plus
per-channel scales. Torch-TensorRT maps the exported
``torchao.dequantize_affine`` operations to TensorRT ``IDequantizeLayer``.
No calibration dataset is required.

You need access to the gated Alpamayo checkpoint on Hugging Face.

.. code-block:: bash

    pip install torchao transformers accelerate einops hydra-core
    pip install git+https://github.com/NVlabs/alpamayo1.5.git

    export ALPAMAYO_MODEL_ID=nvidia/Alpamayo-1.5-10B
    python torch_export_alpamayo_fp8_woq.py

On Blackwell, set Myelin prologue-fusion flags to encourage DQ+GEMM fusion::

    export __LUNOWUD='-log:level=1 -log:dump=on -trace:use_id=on -mlir:prologue_fusion=1 -mlir:fusion_profit_threshold=0.01'

"""

# %%
# Imports
# -------
# This example lives in ``examples/dynamo/torchao/``. Move that directory off
# the front of ``sys.path`` so ``import torchao`` resolves the PyPI package
# instead of this folder.

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

_EXAMPLE_DIR = str(Path(__file__).resolve().parent)
if sys.path and Path(sys.path[0]).resolve() == Path(_EXAMPLE_DIR):
    sys.path.pop(0)

import torch
import torch_tensorrt
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from torchao.quantization import Float8WeightOnlyConfig, quantize_

sys.path.insert(0, _EXAMPLE_DIR)
from utils import exclude_dq_from_constant_folding, pre_process_model_for_export

DEVICE = "cuda:0"
MODEL_ID = os.environ.get("ALPAMAYO_MODEL_ID", "nvidia/Alpamayo-1.5-10B")
BATCH_SIZE = int(os.environ.get("ALPAMAYO_BATCH_SIZE", "1"))


# %%
# Define an export-friendly expert wrapper
# ----------------------------------------
# The Alpamayo expert consumes projected noisy actions and predicts the
# flow-matching velocity used by each diffusion step. The production rollout
# also passes the VLM prefix KV cache. Omitting that mutable cache gives us a
# static graph suitable for demonstrating TorchAO FP8 WOQ export and TensorRT
# compilation.


class AlpamayoExpertWrapper(torch.nn.Module):
    """Cache-independent Alpamayo diffusion expert."""

    def __init__(
        self,
        action_in_proj: torch.nn.Module,
        expert: torch.nn.Module,
        action_out_proj: torch.nn.Module,
        action_dims: tuple[int, ...],
        non_causal_attention: bool,
    ) -> None:
        super().__init__()
        self.action_in_proj = action_in_proj
        self.expert = expert
        self.action_out_proj = action_out_proj
        self.action_dims = action_dims
        self.non_causal_attention = non_causal_attention

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = noisy_action.shape[0]
        num_action_tokens = self.action_dims[0]

        inputs_embeds = self.action_in_proj(noisy_action, timestep)
        if inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.view(
                batch_size,
                num_action_tokens,
                -1,
            )

        forward_kwargs = {}
        if self.non_causal_attention:
            forward_kwargs["is_causal"] = False

        hidden_states = self.expert(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=False,
            return_dict=False,
            **forward_kwargs,
        )[0]
        return self.action_out_proj(hidden_states).view(
            batch_size,
            *self.action_dims,
        )


# %%
# Load Alpamayo and extract the expert path
# ------------------------------------------
# Only the action projections and diffusion expert are retained. The VLM is
# deleted before compilation to release its memory.

print(f"Loading {MODEL_ID} ...")
model = (
    Alpamayo1_5.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    .to(DEVICE)
    .eval()
)

action_dims = tuple(model.action_space.get_action_space_dims())
expert_wrapper = (
    AlpamayoExpertWrapper(
        action_in_proj=model.action_in_proj,
        expert=model.expert,
        action_out_proj=model.action_out_proj,
        action_dims=action_dims,
        non_causal_attention=model.config.expert_non_causal_attention,
    )
    .to(DEVICE)
    .eval()
)

model.action_in_proj = None
model.expert = None
model.action_out_proj = None
del model
gc.collect()
torch.cuda.empty_cache()

# %%
# Quantize the expert with TorchAO
# --------------------------------
# FP8 weight-only quantization is calibration-free. Activations remain BF16.

quantize_(expert_wrapper, Float8WeightOnlyConfig())

# %%
# Create fixed-shape example inputs
# ---------------------------------
# Alpamayo uses three-component temporal/height/width RoPE position IDs.

num_action_tokens, action_width = action_dims
noisy_action = torch.randn(
    BATCH_SIZE,
    num_action_tokens,
    action_width,
    dtype=torch.bfloat16,
    device=DEVICE,
)
timestep = torch.rand(
    BATCH_SIZE,
    1,
    1,
    dtype=torch.bfloat16,
    device=DEVICE,
)
position_ids = (
    torch.arange(num_action_tokens, device=DEVICE)
    .view(1, 1, num_action_tokens)
    .expand(3, BATCH_SIZE, num_action_tokens)
    .clone()
)
example_inputs = (noisy_action, timestep, position_ids)

# %%
# Eager TorchAO reference
# -----------------------

with torch.no_grad():
    eager_output = expert_wrapper(*example_inputs)

# %%
# Export the quantized expert
# ---------------------------
# ``exclude_dq_from_constant_folding`` preserves explicit FP8 DQ operations for
# TensorRT. ``strict=False`` accommodates the Transformers expert module.

processed_expert = pre_process_model_for_export(expert_wrapper)
with exclude_dq_from_constant_folding():
    exported_program = torch.export.export(
        processed_expert,
        example_inputs,
        strict=False,
    )

# %%
# Compile with Torch-TensorRT
# ---------------------------

compiled_expert = torch_tensorrt.dynamo.compile(
    exported_program,
    inputs=list(example_inputs),
    truncate_double=True,
    min_block_size=1,
    require_full_compilation=True,
    immutable_weights=False,
    offload_module_to_cpu=True,
)

del exported_program, processed_expert, expert_wrapper
gc.collect()
torch.cuda.empty_cache()

# %%
# Compare eager and TensorRT outputs
# ----------------------------------

with torch.no_grad():
    trt_output = compiled_expert(*example_inputs)
if isinstance(trt_output, (tuple, list)):
    trt_output = trt_output[0]
trt_output = trt_output.to(device=eager_output.device, dtype=eager_output.dtype)

absolute_difference = (trt_output.float() - eager_output.float()).abs()
print(f"output shape:     {tuple(trt_output.shape)}")
print(f"max |Δoutput|:    {absolute_difference.max().item():.6g}")
print(f"mean |Δoutput|:   {absolute_difference.mean().item():.6g}")
