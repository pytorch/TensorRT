"""
.. _torch_export_flux_dev:

Compiling FLUX.1-dev model using the Torch-TensorRT dynamo backend
===================================================================

This example illustrates the state of the art model `FLUX.1-dev <https://huggingface.co/black-forest-labs/FLUX.1-dev>`_ optimized using
Torch-TensorRT.

**FLUX.1 [dev]** is a 12 billion parameter rectified flow transformer capable of generating images from text descriptions. It is an open-weight, guidance-distilled model for non-commercial applications.

To run this demo, you need to have access to Flux model (request for access if you do not have it already on the `FLUX.1-dev <https://huggingface.co/black-forest-labs/FLUX.1-dev>`_ page) and install the following dependencies

.. code-block:: python

    pip install sentencepiece=="0.2.0" transformers=="4.48.2" accelerate=="1.3.0" diffusers=="0.32.2" protobuf=="5.29.3"

There are different components of the ``FLUX.1-dev`` pipeline such as ``transformer``, ``vae``, ``text_encoder``, ``tokenizer`` and ``scheduler``. In this example,
we demonstrate optimizing the ``transformer`` component of the model (which typically consumes >95% of the e2e diffusion latency)
"""

import register_sdpa  # Register SDPA as a standalone operator

# %%
# Import the following libraries
# -----------------------------
import torch
import torch_tensorrt
from diffusers import FluxPipeline
from torch.export._trace import _export

# %%
# Define the FLUX-1.dev model
# -----------------------------
# Load the ``FLUX-1.dev`` pretrained pipeline using ``FluxPipeline`` class.
# ``FluxPipeline`` includes different components such as ``transformer``, ``vae``, ``text_encoder``, ``tokenizer`` and ``scheduler`` necessary
# to generate an image. We load the weights in ``FP16`` precision using ``torch_dtype`` argument
DEVICE = "cuda:0"
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16,
)

# Store the config and transformer backbone
config = pipe.transformer.config
backbone = pipe.transformer.to(DEVICE)

# %%
# Export the backbone using torch.export
# --------------------------------------------------
# Define the dummy inputs and their respective dynamic shapes. We export the transformer backbone with dynamic shapes with a ``batch_size=2``
# due to `0/1 specialization <https://docs.google.com/document/d/16VPOa3d-Liikf48teAOmxLc92rgvJdfosIy-yoT38Io/edit?fbclid=IwAR3HNwmmexcitV0pbZm_x1a4ykdXZ9th_eJWK-3hBtVgKnrkmemz6Pm5jRQ&tab=t.0#heading=h.ez923tomjvyk>`_
batch_size = 2
BATCH = torch.export.Dim("batch", min=1, max=2)
SEQ_LEN = torch.export.Dim("seq_len", min=1, max=512)
# This particular min, max values for img_id input are recommended by torch dynamo during the export of the model.
# To see this recommendation, you can try exporting using min=1, max=4096
IMG_ID = torch.export.Dim("img_id", min=3586, max=4096)
dynamic_shapes = {
    "hidden_states": {0: BATCH},
    "encoder_hidden_states": {0: BATCH, 1: SEQ_LEN},
    "pooled_projections": {0: BATCH},
    "timestep": {0: BATCH},
    "txt_ids": {0: SEQ_LEN},
    "img_ids": {0: IMG_ID},
    "guidance": {0: BATCH},
    "joint_attention_kwargs": {},
    "return_dict": None,
}
# The guidance factor is of type torch.float32
dummy_inputs = {
    "hidden_states": torch.randn((batch_size, 4096, 64), dtype=torch.float16).to(
        DEVICE
    ),
    "encoder_hidden_states": torch.randn(
        (batch_size, 512, 4096), dtype=torch.float16
    ).to(DEVICE),
    "pooled_projections": torch.randn((batch_size, 768), dtype=torch.float16).to(
        DEVICE
    ),
    "timestep": torch.tensor([1.0, 1.0], dtype=torch.float16).to(DEVICE),
    "txt_ids": torch.randn((512, 3), dtype=torch.float16).to(DEVICE),
    "img_ids": torch.randn((4096, 3), dtype=torch.float16).to(DEVICE),
    "guidance": torch.tensor([1.0, 1.0], dtype=torch.float32).to(DEVICE),
    "joint_attention_kwargs": {},
    "return_dict": False,
}
# This will create an exported program which is going to be compiled with Torch-TensorRT
ep = _export(
    backbone,
    args=(),
    kwargs=dummy_inputs,
    dynamic_shapes=dynamic_shapes,
    strict=False,
    prefer_deferred_runtime_asserts_over_guards=True,
)

# %%
# Torch-TensorRT compilation
# ---------------------------
# .. note::
#    The compilation requires a GPU with high memory (> 80GB) since TensorRT is storing the weights in FP32 precision. This is a known issue and will be resolved in the future.
#
#
# We enable ``FP32`` matmul accumulation using ``use_fp32_acc=True`` to ensure accuracy is preserved by introducing cast to ``FP32`` nodes.
# We also enable explicit typing to ensure TensorRT respects the datatypes set by the user which is a requirement for FP32 matmul accumulation.
# Since this is a 12 billion parameter model, it takes around 20-30 min to compile on H100 GPU. The model is completely convertible and results in
# a single TensorRT engine.
trt_gm = torch_tensorrt.dynamo.compile(
    ep,
    inputs=dummy_inputs,
    enabled_precisions={torch.float32},
    truncate_double=True,
    min_block_size=1,
    use_fp32_acc=True,
    use_explicit_typing=True,
    immutable_weights=False,
    offload_module_to_cpu=True,
)

# %%
# Post Processing
# ---------------------------
# Release the GPU memory occupied by the exported program and the pipe.transformer
# Set the transformer in the Flux pipeline to the Torch-TRT compiled model
pipe.transformer = None
pipe.to(DEVICE)
pipe.transformer = trt_gm
del ep
torch.cuda.empty_cache()
pipe.transformer.config = config
trt_gm.device = torch.device("cuda")
# %%
# Image generation using prompt
# ---------------------------
# Provide a prompt and the file name of the image to be generated. Here we use the
# prompt ``A golden retriever holding a sign to code``.


# Function which generates images from the flux pipeline
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


generate_image(pipe, ["A golden retriever holding a sign to code"], "dog_code")

# %%
# The generated image is as shown below
#
# .. image:: dog_code.png
#
