"""
.. _torch_export_flux_fp8_woq:

Compiling FLUX.1-dev with TorchAO FP8 weight-only quantization
==============================================================

This example quantizes the ``transformer`` of
`FLUX.1-dev <https://huggingface.co/black-forest-labs/FLUX.1-dev>`_ with TorchAO
``Float8WeightOnlyConfig``, then compiles it with the Torch-TensorRT Dynamo
backend.

Weight-only FP8 keeps activations in BF16 and stores Linear weights as FP8 plus
per-channel scales. After export, Torch-TensorRT maps ``dequantize_affine`` to
TensorRT ``IDequantizeLayer`` so the engine can keep an FP8 weight constant
instead of folding DQ into a dense BF16 GEMM.

You need access to FLUX.1-dev on Hugging Face and a GPU with enough memory to
load and compile the 12B transformer (the unquantized Flux compile path wants
>80GB; FP8 WOQ reduces the weight footprint).

.. code-block:: bash

    pip install torchao diffusers transformers accelerate sentencepiece protobuf

On Blackwell, set Myelin prologue-fusion flags to encourage DQ+GEMM fusion::

    export __LUNOWUD='-log:level=1 -log:dump=on -trace:use_id=on -mlir:prologue_fusion=1 -mlir:fusion_profit_threshold=0.01'

"""

# %%
# Imports
# -------
# This example lives in ``examples/dynamo/torchao/``. Move that directory off
# the front of ``sys.path`` so ``import torchao`` resolves the PyPI package
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
from torchao.quantization import Float8WeightOnlyConfig, quantize_

sys.path.insert(0, _EXAMPLE_DIR)
from utils import exclude_dq_from_constant_folding, pre_process_model_for_export

DEVICE = "cuda:0"
MODEL_ID = os.environ.get("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-dev")

# %%
# Load FLUX.1-dev and quantize the transformer
# --------------------------------------------
# Only the transformer is quantized and compiled. Text encoders and the VAE stay
# BF16. ``Float8WeightOnlyConfig`` is weight-only, so no calibration dataset is
# required.

pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)
config = pipe.transformer.config
backbone = pipe.transformer.to(DEVICE)

quantize_(backbone, Float8WeightOnlyConfig())
backbone = pre_process_model_for_export(backbone)

# %%
# Export the quantized transformer
# --------------------------------
# Dummy inputs match the Flux transformer signature used by the BF16 Flux
# example. ``exclude_dq_from_constant_folding`` keeps ``dequantize_affine`` in
# the graph during ``torch.export``.

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

with exclude_dq_from_constant_folding():
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
#    Compilation of the 12B transformer takes on the order of 20–30 minutes on
#    an H100. ``offload_module_to_cpu`` frees PyTorch weights after they are
#    ingested by TensorRT.

trt_gm = torch_tensorrt.dynamo.compile(
    exp_program,
    inputs=dummy_inputs,
    truncate_double=True,
    min_block_size=1,
    use_explicit_typing=True,
    require_full_compilation=True,
    immutable_weights=False,
    offload_module_to_cpu=True,
)

# %%
# Swap the compiled transformer into the pipeline
# -----------------------------------------------

pipe.transformer = None
pipe.to(DEVICE)
pipe.transformer = trt_gm
pipe.transformer.config = config
trt_gm.device = torch.device("cuda")
del exp_program, backbone
gc.collect()
torch.cuda.empty_cache()

# %%
# Generate an image
# -----------------


def generate_image(pipe, prompt, image_name):
    seed = 42
    with torch.no_grad():
        image = pipe(
            prompt,
            output_type="pil",
            num_inference_steps=20,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
        image.save(f"{image_name}.png")
        print(f"Image generated using {image_name} model saved as {image_name}.png")


generate_image(
    pipe,
    [
        "Baroque style, a lavish palace interior with ornate gilded ceilings, "
        "intricate tapestries, and dramatic lighting over a grand staircase."
    ],
    "flux_fp8_woq",
)
