"""
.. _refit_engine_example:

Refit  TenorRT Graph Module with Torch-TensorRT
===================================================================

We are going to demonstrate how a compiled TensorRT Graph Module can be refitted with updated weights.

In many cases, we frequently update the weights of models, such as applying various LoRA to Stable Diffusion or constant A/B testing of AI products.
That poses challenges for TensorRT inference optimizations, as compiling the TensorRT engines takes significant time, making repetitive compilation highly inefficient.
Torch-TensorRT supports refitting TensorRT graph modules without re-compiling the engine, considerably accelerating the workflow.

In this tutorial, we are going to walk through
1. Compiling a PyTorch model to a TensorRT Graph Module
2. Save and load a graph module
3. Refit the graph module
"""

# %%
# Standard Workflow
# -----------------------------

# %%
# Imports and model definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import numpy as np
import torch
import torch_tensorrt as torch_trt
import torchvision.models as models
from diffusers import DiffusionPipeline

np.random.seed(0)
torch.manual_seed(1)


# %%
# Compile the module for the first time and save it.
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
kwargs = {
    "use_python": False,
    "enabled_precisions": {torch.float32},
    "make_refitable": True,
}

model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda:0"

# Instantiate Stable Diffusion Pipeline with FP16 weigh ts
prompt = "portrait of a woman standing, shuimobysim, wuchangshuo, best quality"
negative = "(worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, skin spots, acnes, skin blemishes, age spot, glans, (watermark:2),"

pipe = DiffusionPipeline.from_pretrained(
    model_id, revision="fp16", torch_dtype=torch.float16, safety_checker=None
)
pipe.to(device)
backend = "torch_tensorrt" 

pipe.unet = torch_trt.MutableTorchTensorRTModule(pipe.unet, **kwargs)
image = pipe(prompt, negative_prompt=negative, num_inference_steps=30).images[0]
image.save('./without_LoRA.jpg')


pipe.load_lora_weights("/opt/torch_tensorrt/moxin.safetensors", adapter_name="lora1")
pipe.set_adapters(["lora1"], adapter_weights=[1])
pipe.fuse_lora(['lora1'], 1)
pipe.unload_lora_weights()


# Check the output
image = pipe(prompt, negative_prompt=negative, num_inference_steps=30).images[0]
image.save('./with_LoRA.jpg')


