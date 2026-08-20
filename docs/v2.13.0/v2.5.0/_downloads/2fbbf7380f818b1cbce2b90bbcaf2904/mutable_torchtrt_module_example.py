"""
.. _mutable_torchtrt_module_example:

Mutable Torch TensorRT Module
===================================================================

We are going to demonstrate how we can easily use Mutable Torch TensorRT Module to compile, interact, and modify the TensorRT Graph Module.

Compiling a Torch-TensorRT module is straightforward, but modifying the compiled module can be challenging, especially when it comes to maintaining the state and connection between the PyTorch module and the corresponding Torch-TensorRT module.
In Ahead-of-Time (AoT) scenarios, integrating Torch TensorRT with complex pipelines, such as the Hugging Face Stable Diffusion pipeline, becomes even more difficult.
The Mutable Torch TensorRT Module is designed to address these challenges, making interaction with the Torch-TensorRT module easier than ever.

In this tutorial, we are going to walk through
1. Sample workflow of Mutable Torch TensorRT Module with ResNet 18
2. Save a Mutable Torch TensorRT Module
3. Integration with Huggingface pipeline in LoRA use case
"""

import numpy as np
import torch
import torch_tensorrt as torch_trt
import torchvision.models as models

np.random.seed(5)
torch.manual_seed(5)
inputs = [torch.rand((1, 3, 224, 224)).to("cuda")]

# %%
# Initialize the Mutable Torch TensorRT Module with settings.
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
settings = {
    "use_python": False,
    "enabled_precisions": {torch.float32},
    "make_refittable": True,
}

model = models.resnet18(pretrained=True).eval().to("cuda")
mutable_module = torch_trt.MutableTorchTensorRTModule(model, **settings)
# You can use the mutable module just like the original pytorch module. The compilation happens while you first call the mutable module.
mutable_module(*inputs)

# %%
# Make modifications to the mutable module.
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# %%
# Making changes to mutable module can trigger refit or re-compilation. For example, loading a different state_dict and setting new weight values will trigger refit, and adding a module to the model will trigger re-compilation.
model2 = models.resnet18(pretrained=False).eval().to("cuda")
mutable_module.load_state_dict(model2.state_dict())


# Check the output
# The refit happens while you call the mutable module again.
expected_outputs, refitted_outputs = model2(*inputs), mutable_module(*inputs)
for expected_output, refitted_output in zip(expected_outputs, refitted_outputs):
    assert torch.allclose(
        expected_output, refitted_output, 1e-2, 1e-2
    ), "Refit Result is not correct. Refit failed"

print("Refit successfully!")

# %%
# Saving Mutable Torch TensorRT Module
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Currently, saving is only enabled for C++ runtime, not python runtime.
torch_trt.MutableTorchTensorRTModule.save(mutable_module, "mutable_module.pkl")
reload = torch_trt.MutableTorchTensorRTModule.load("mutable_module.pkl")

# %%
# Stable Diffusion with Huggingface
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# The LoRA checkpoint is from https://civitai.com/models/12597/moxin

from diffusers import DiffusionPipeline

with torch.no_grad():
    settings = {
        "use_python_runtime": True,
        "enabled_precisions": {torch.float16},
        "debug": True,
        "make_refittable": True,
    }

    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda:0"

    prompt = "house in forest, shuimobysim, wuchangshuo, best quality"
    negative = "(worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, out of focus, cloudy, (watermark:2),"

    pipe = DiffusionPipeline.from_pretrained(
        model_id, revision="fp16", torch_dtype=torch.float16
    )
    pipe.to(device)

    # The only extra line you need
    pipe.unet = torch_trt.MutableTorchTensorRTModule(pipe.unet, **settings)

    image = pipe(prompt, negative_prompt=negative, num_inference_steps=30).images[0]
    image.save("./without_LoRA_mutable.jpg")

    # Standard Huggingface LoRA loading procedure
    pipe.load_lora_weights(
        "stablediffusionapi/load_lora_embeddings",
        weight_name="moxin.safetensors",
        adapter_name="lora1",
    )
    pipe.set_adapters(["lora1"], adapter_weights=[1])
    pipe.fuse_lora()
    pipe.unload_lora_weights()

    # Refit triggered
    image = pipe(prompt, negative_prompt=negative, num_inference_steps=30).images[0]
    image.save("./with_LoRA_mutable.jpg")
