"""
.. _torch_export_alpamayo_modelopt_fp8:

Compiling Alpamayo 1.5 with ModelOpt FP8 quantization
=====================================================

This example loads
`NVIDIA Alpamayo 1.5 10B <https://huggingface.co/nvidia/Alpamayo-1.5-10B>`_,
extracts its diffusion expert path, applies NVIDIA ModelOpt FP8 post-training
quantization, and compiles the result with the Torch-TensorRT Dynamo backend.

Alpamayo's complete trajectory rollout contains Hugging Face autoregressive
generation, mutable KV caches, and a Python-controlled diffusion loop. Those
parts cannot be represented by one static ``torch.export`` graph. This example
therefore compiles the cache-independent expert graph:
``action_in_proj -> expert -> action_out_proj``.

The calibration loop below uses synthetic tensors so the example is
self-contained. It validates the ModelOpt-to-Torch-TensorRT Q/DQ workflow, but
it is not an accuracy recipe. For representative PhysicalAI calibration,
checkpoint evaluation, and full vision/language/action Edge export, see
:ref:`alpamayo_fp8_edge_exporter` and the
`official Alpamayo ModelOpt recipe
<https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo1_5_quant>`_.

Requirements:

* NVIDIA GPU with FP8 support
* ``nvidia-modelopt[hf]>=0.44.0``
* ``transformers`` and the Alpamayo 1.5 package
* a compatible Torch-TensorRT installation
* access to the gated Alpamayo checkpoint

.. code-block:: bash

    pip install "nvidia-modelopt[hf]>=0.44.0" transformers accelerate einops hydra-core
    pip install git+https://github.com/NVlabs/alpamayo1.5.git

    export ALPAMAYO_MODEL_ID=nvidia/Alpamayo-1.5-10B
    export ALPAMAYO_MODELOPT_CALIB_STEPS=8
    python torch_export_alpamayo_modelopt_fp8.py

"""

# %%
# Imports
# -------

from __future__ import annotations

import gc
import os

import modelopt.torch.quantization as mtq
import torch
import torch_tensorrt
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
from modelopt.torch.quantization.utils import export_torch_mode

DEVICE = "cuda:0"
MODEL_ID = os.environ.get("ALPAMAYO_MODEL_ID", "nvidia/Alpamayo-1.5-10B")
BATCH_SIZE = int(os.environ.get("ALPAMAYO_BATCH_SIZE", "1"))
CALIBRATION_STEPS = int(os.environ.get("ALPAMAYO_MODELOPT_CALIB_STEPS", "8"))


# %%
# Define the expert graph
# -----------------------


class AlpamayoExpert(torch.nn.Module):
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
        return self.action_out_proj(hidden_states).reshape(
            batch_size,
            *self.action_dims,
        )


# %%
# Load Alpamayo
# -------------
# Retain only the action projections and diffusion expert so the VLM does not
# consume memory during calibration and TensorRT compilation.

print(f"Loading {MODEL_ID} ...")
model = (
    Alpamayo1_5.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        attn_implementation="eager",
    )
    .to(DEVICE)
    .eval()
)

action_dims = tuple(model.action_space.get_action_space_dims())
expert_model = (
    AlpamayoExpert(
        model.action_in_proj,
        model.expert,
        model.action_out_proj,
        action_dims,
        model.config.expert_non_causal_attention,
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
# Create example inputs
# ---------------------

num_action_tokens, action_width = action_dims
noisy_action = torch.randn(
    BATCH_SIZE,
    num_action_tokens,
    action_width,
    device=DEVICE,
    dtype=torch.float16,
)
timestep = torch.rand(
    BATCH_SIZE,
    1,
    1,
    device=DEVICE,
    dtype=torch.float16,
)
position_ids = (
    torch.arange(num_action_tokens, device=DEVICE)
    .view(1, 1, num_action_tokens)
    .expand(3, BATCH_SIZE, num_action_tokens)
    .clone()
)
example_inputs = (noisy_action, timestep, position_ids)


# %%
# Calibrate ModelOpt FP8
# ----------------------
# ``FP8_DEFAULT_CFG`` quantizes eligible Linear inputs and weights. The
# synthetic loop exercises the complete isolated expert graph. Production
# calibration should use representative PhysicalAI clips as described in the
# user guide.


def calibration_loop(module: torch.nn.Module) -> None:
    generator = torch.Generator(device=DEVICE).manual_seed(1234)
    with torch.no_grad():
        for _ in range(CALIBRATION_STEPS):
            calib_action = torch.randn(
                noisy_action.shape,
                generator=generator,
                device=DEVICE,
                dtype=noisy_action.dtype,
            )
            calib_timestep = torch.rand(
                timestep.shape,
                generator=generator,
                device=DEVICE,
                dtype=timestep.dtype,
            )
            module(calib_action, calib_timestep, position_ids)


mtq.quantize(
    expert_model,
    mtq.FP8_DEFAULT_CFG,
    forward_loop=calibration_loop,
)
mtq.print_quant_summary(expert_model)


# %%
# Capture the quantized eager reference
# -------------------------------------

with torch.no_grad():
    eager_output = expert_model(*example_inputs)


# %%
# Export ModelOpt Q/DQ operations
# -------------------------------
# ``export_torch_mode`` lowers ModelOpt TensorQuantizers to explicit
# ``torch.ops.tensorrt`` Q/DQ operations that Torch-TensorRT converters
# recognize. Keep weights in fake-quant form until after export.

with export_torch_mode():
    exported_program = torch.export.export(
        expert_model,
        example_inputs,
        strict=False,
    )


# %%
# Compile with Torch-TensorRT
# ---------------------------

compiled_expert = torch_tensorrt.dynamo.compile(
    exported_program,
    arg_inputs=list(example_inputs),
    truncate_double=True,
    min_block_size=1,
    require_full_compilation=True,
    immutable_weights=False,
    offload_module_to_cpu=True,
)

del exported_program, expert_model
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
close = torch.isclose(
    trt_output.float(),
    eager_output.float(),
    rtol=1e-2,
    atol=1e-2,
)
print(f"output shape:     {tuple(trt_output.shape)}")
print(f"max |Δoutput|:    {absolute_difference.max().item():.6g}")
print(f"mean |Δoutput|:   {absolute_difference.mean().item():.6g}")
print(f"close elements:   {close.float().mean().item() * 100:.2f}%")
