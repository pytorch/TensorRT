"""
.. _torch_compile_stable_diffusion:

Compiling Stable Diffusion model using the `torch.compile` backend
======================================================

This interactive script is intended as a sample of the Torch-TensorRT workflow with `torch.compile` on a Stable Diffusion model. A sample output is featured below:

.. image:: /tutorials/images/majestic_castle.png
   :width: 512px
   :height: 512px
   :scale: 50 %
   :align: right
"""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import torch
import torch_tensorrt
from diffusers import DiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda:0"

# Instantiate Stable Diffusion Pipeline with FP16 weights
pipe = DiffusionPipeline.from_pretrained(
    model_id, revision="fp16", torch_dtype=torch.float16
)
pipe = pipe.to(device)

backend = "torch_tensorrt"

# Optimize the UNet portion with Torch-TensorRT
pipe.unet = torch.compile(
    pipe.unet,
    backend=backend,
    options={
        "truncate_long_and_double": True,
        "enabled_precisions": {torch.float32, torch.float16},
    },
    dynamic=False,
)

# %%
# Inference
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
with torch.no_grad():
    prompt = "a majestic castle in the clouds"
    image = pipe(prompt).images[0]

    image.save("images/majestic_castle.png")
    image.show()
