"""
.. _torch_export_flux_nvfp4_woq:

Compiling FLUX.1-dev with TorchAO NVFP4 weight-only quantization
================================================================

This example quantizes the transformer of
`FLUX.1-dev <https://huggingface.co/black-forest-labs/FLUX.1-dev>`_ to NVFP4
with TorchAO (packed FP4 E2M1 + FP8 block scales), then compiles it with the
Torch-TensorRT Dynamo backend.

Weight-only NVFP4 keeps activations in BF16 and stores Linear weights as
packed FP4 plus swizzled FP8 block scales and a global scale. After export,
Torch-TensorRT maps ``dequantize_nvfp4`` to two-level TensorRT
``IDequantizeLayer`` so the engine can keep an FP4 weight constant.

This is **not** native FP4 Tensor Core MMA. Layers whose last two dims are
not divisible by 16 are left in BF16.

You need access to FLUX.1-dev on Hugging Face. Quantization is done
layer-by-layer on GPU so the rest of the pipeline can stay on CPU. Compile
with ``immutable_weights=True``. On 32GB cards, park the text encoders during
compile and generate from precomputed prompt embeddings so T5 and the TRT
execution context are not resident together.

.. code-block:: bash

    pip install torchao diffusers transformers accelerate sentencepiece protobuf

"""

# %%
# Imports
# -------
# This example lives in examples/dynamo/torchao/. Move that directory off
# the front of sys.path so import torchao resolves the PyPI package
# instead of this folder.

import gc
import os
import sys
from pathlib import Path

_EXAMPLE_DIR = str(Path(__file__).resolve().parent)
if sys.path and Path(sys.path[0]).resolve() == Path(_EXAMPLE_DIR):
    sys.path.pop(0)

import torch
import torch_tensorrt
from diffusers import FluxPipeline

sys.path.insert(0, _EXAMPLE_DIR)
from nvfp4_utils import pre_process_model_for_export, quantize_linear_nvfp4

DEVICE = "cuda:0"
MODEL_ID = os.environ.get("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-dev")
PROMPT = (
    "Baroque style, a lavish palace interior with ornate gilded ceilings, "
    "intricate tapestries, and dramatic lighting over a grand staircase."
)
MAX_SEQUENCE_LENGTH = 512

# %%
# Load FLUX.1-dev and quantize the transformer
# --------------------------------------------
# Only the transformer is quantized and compiled. Text encoders and the VAE stay
# BF16. Load on CPU, then quantize each Linear on GPU individually.

pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)
config = pipe.transformer.config
cache_context = getattr(pipe.transformer, "cache_context", None)

quantize_linear_nvfp4(pipe.transformer)
pipe.transformer = pre_process_model_for_export(pipe.transformer)

# %%
# Encode the prompt, then free the text encoders
# ----------------------------------------------
# The NVFP4 engine's execution context can reserve tens of GB. Compute
# embeddings once, then drop CLIP/T5 so they are not resident during compile
# or generate.

pipe.to(DEVICE)
with torch.no_grad():
    prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
        prompt=PROMPT,
        prompt_2=PROMPT,
        device=torch.device(DEVICE),
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )
pipe.text_encoder = None
pipe.text_encoder_2 = None
pipe.vae.to("cpu")
gc.collect()
torch.cuda.empty_cache()

backbone = pipe.transformer.to(DEVICE)

# %%
# Export the quantized transformer
# --------------------------------
# Dummy inputs match the Flux transformer signature used by the BF16 / FP8 /
# INT4 Flux examples. Torch-TensorRT constant folding keeps
# ``dequantize_nvfp4`` in the graph.

dummy_inputs = {
    "hidden_states": torch.randn(1, 4096, 64, dtype=torch.bfloat16, device=DEVICE),
    "encoder_hidden_states": torch.randn(
        1, 512, 4096, dtype=torch.bfloat16, device=DEVICE
    ),
    "pooled_projections": torch.randn(1, 768, dtype=torch.bfloat16, device=DEVICE),
    "timestep": torch.randn(1, dtype=torch.bfloat16, device=DEVICE),
    "guidance": torch.randn(1, dtype=torch.float32, device=DEVICE),
    "img_ids": torch.randn(4096, 3, dtype=torch.bfloat16, device=DEVICE),
    "txt_ids": torch.randn(512, 3, dtype=torch.bfloat16, device=DEVICE),
    "joint_attention_kwargs": {},
    "return_dict": False,
}

exp_program = torch.export.export(
    backbone,
    args=(),
    kwargs=dummy_inputs,
    strict=True,
)

# %%
# Compile with Torch-TensorRT
# ---------------------------
# .. note::
#    Compilation of the 12B transformer takes on the order of 20–30 minutes.
#    ``immutable_weights=True`` keeps the packed FP4 constants from being
#    refit as dense high-precision weights.

trt_gm = torch_tensorrt.dynamo.compile(
    exp_program,
    inputs=dummy_inputs,
    truncate_double=True,
    min_block_size=1,
    use_explicit_typing=True,
    require_full_compilation=True,
    immutable_weights=True,
    offload_module_to_cpu=True,
)

# %%
# Swap the compiled transformer into the pipeline
# -----------------------------------------------
# Wrap the engine so Diffusers still sees a CUDA module after the text encoders
# are dropped, and so TRT outputs are moved back to GPU if they land on CPU.


class _TRTTransformerAdapter(torch.nn.Module):
    def __init__(self, compiled, config, cache_context) -> None:
        super().__init__()
        self._compiled = compiled
        self.config = config
        self.cache_context = cache_context
        self.register_buffer(
            "_device_anchor", torch.empty(0, device=DEVICE), persistent=False
        )

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    def forward(self, *args, **kwargs):
        def _to_cuda(x):
            return x.to(DEVICE) if isinstance(x, torch.Tensor) else x

        args = tuple(_to_cuda(a) for a in args)
        kwargs = {k: _to_cuda(v) for k, v in kwargs.items()}
        out = self._compiled(*args, **kwargs)
        if isinstance(out, (tuple, list)):
            return type(out)(_to_cuda(o) for o in out)
        return _to_cuda(out)


pipe.transformer = _TRTTransformerAdapter(trt_gm, config, cache_context)
pipe.vae.to(DEVICE)
del exp_program, backbone, trt_gm
gc.collect()
torch.cuda.empty_cache()

# %%
# Generate an image
# -----------------


def generate_image(pipe, image_name):
    seed = 42
    with torch.no_grad():
        image = pipe(
            prompt_embeds=prompt_embeds.to(DEVICE),
            pooled_prompt_embeds=pooled_prompt_embeds.to(DEVICE),
            output_type="pil",
            num_inference_steps=20,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
        image.save(f"{image_name}.png")
        print(f"Image generated using {image_name} model saved as {image_name}.png")


generate_image(pipe, "flux_nvfp4_woq")
