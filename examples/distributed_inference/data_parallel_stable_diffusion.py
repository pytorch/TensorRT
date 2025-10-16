"""
.. _data_parallel_stable_diffusion:

Torch-TensorRT Distributed Inference
======================================================

This interactive script is intended as a sample of distributed inference using data
parallelism using Accelerate
library with the Torch-TensorRT workflow on Stable Diffusion model.

"""

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import torch
import torch_tensorrt
from accelerate import PartialState
from diffusers import DiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"

# Instantiate Stable Diffusion Pipeline with FP16 weights
pipe = DiffusionPipeline.from_pretrained(
    model_id, revision="fp16", torch_dtype=torch.float16
)

distributed_state = PartialState()
pipe = pipe.to(distributed_state.device)

backend = "torch_tensorrt"

# Optimize the UNet portion with Torch-TensorRT
pipe.unet = torch.compile(  # %%
    # Inference
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Assume there are 2 processes (2 devices)
    pipe.unet,
    backend=backend,
    options={
        "truncate_long_and_double": True,
        "precision": torch.float16,
        "use_python_runtime": True,
    },
    dynamic=False,
)
torch_tensorrt.runtime.set_multi_device_safe_mode(True)


# %%
# Inference
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Assume there are 2 processes (2 devices)
with distributed_state.split_between_processes(["a dog", "a cat"]) as prompt:
    result = pipe(prompt).images[0]
    result.save(f"result_{distributed_state.process_index}.png")
