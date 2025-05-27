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
    4. Usage of dynamic shape with Mutable Torch TensorRT Module
"""

# %%
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
    "immutable_weights": False,
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

# Currently, saving is only enabled when "use_python_runtime" = False in settings
torch_trt.MutableTorchTensorRTModule.save(mutable_module, "mutable_module.pkl")
reload = torch_trt.MutableTorchTensorRTModule.load("mutable_module.pkl")

# %%
# Stable Diffusion with Huggingface
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from diffusers import DiffusionPipeline

with torch.no_grad():
    settings = {
        "use_python_runtime": True,
        "enabled_precisions": {torch.float16},
        "debug": True,
        "immutable_weights": False,
    }

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    device = "cuda:0"

    prompt = "cinematic photo elsa, police uniform <lora:princess_xl_v2:0.8>, . 35mm photograph, film, bokeh, professional, 4k, highly detailed"
    negative = "drawing, painting, crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly, nude"

    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to(device)

    # The only extra line you need
    pipe.unet = torch_trt.MutableTorchTensorRTModule(pipe.unet, **settings)
    BATCH = torch.export.Dim("BATCH", min=2, max=24)
    _HEIGHT = torch.export.Dim("_HEIGHT", min=16, max=32)
    _WIDTH = torch.export.Dim("_WIDTH", min=16, max=32)
    HEIGHT = 4 * _HEIGHT
    WIDTH = 4 * _WIDTH
    args_dynamic_shapes = ({0: BATCH, 2: HEIGHT, 3: WIDTH}, {})
    kwargs_dynamic_shapes = {
        "encoder_hidden_states": {0: BATCH},
        "added_cond_kwargs": {
            "text_embeds": {0: BATCH},
            "time_ids": {0: BATCH},
        },
        "return_dict": False,
    }
    pipe.unet.set_expected_dynamic_shape_range(
        args_dynamic_shapes, kwargs_dynamic_shapes
    )
    image = pipe(
        prompt,
        negative_prompt=negative,
        num_inference_steps=30,
        height=1024,
        width=768,
        num_images_per_prompt=2,
    ).images[0]
    image.save("./without_LoRA_mutable.jpg")

    # Standard Huggingface LoRA loading procedure
    pipe.load_lora_weights(
        "stablediffusionapi/load_lora_embeddings",
        weight_name="all-disney-princess-xl-lo.safetensors",
        adapter_name="lora1",
    )
    pipe.set_adapters(["lora1"], adapter_weights=[1])
    pipe.fuse_lora()
    pipe.unload_lora_weights()

    # Refit triggered
    image = pipe(
        prompt,
        negative_prompt=negative,
        num_inference_steps=30,
        height=1024,
        width=1024,
        num_images_per_prompt=1,
    ).images[0]
    image.save("./with_LoRA_mutable.jpg")


# %%
# Use Mutable Torch TensorRT module with dynamic shape
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# When adding dynamic shape hint to MutableTorchTensorRTModule, The shape hint should EXACTLY follow the semantics of arg_inputs and kwarg_inputs passed to the forward function
# and should not omit any entries (except None in the kwarg_inputs). If there is a nested dict/list in the input, the dynamic shape for that entry should also be an nested dict/list.
# If the dynamic shape is not required for an input, an empty dictionary should be given as the shape hint for that input.
# Note that you should exclude keyword arguments with value None as those will be filtered out.


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, c={}):
        x = torch.matmul(a, b)
        x = torch.matmul(c["a"], c["b"].T)
        print(c["b"][0])
        x = 2 * c["b"]
        return x


device = "cuda:0"
model = Model().eval().to(device)
inputs = (torch.rand(10, 3).to(device), torch.rand(3, 30).to(device))
kwargs = {
    "c": {"a": torch.rand(10, 30).to(device), "b": torch.rand(10, 30).to(device)},
}
dim_0 = torch.export.Dim("dim", min=1, max=50)
dim_1 = torch.export.Dim("dim", min=1, max=50)
dim_2 = torch.export.Dim("dim2", min=1, max=50)
args_dynamic_shapes = ({1: dim_1}, {0: dim_0})
kwarg_dynamic_shapes = {
    "c": {
        "a": {},
        "b": {0: dim_2},
    },  # a's shape does not change so we give it an empty dict
}
# Export the model first with custom dynamic shape constraints
model = torch_trt.MutableTorchTensorRTModule(model, debug=True, min_block_size=1)
model.set_expected_dynamic_shape_range(args_dynamic_shapes, kwarg_dynamic_shapes)
# Compile
model(*inputs, **kwargs)
# Change input shape
inputs_2 = (torch.rand(10, 5).to(device), torch.rand(10, 30).to(device))
kwargs_2 = {
    "c": {"a": torch.rand(10, 30).to(device), "b": torch.rand(5, 30).to(device)},
}
# Run without recompiling
model(*inputs_2, **kwargs_2)

# %%
# Use Mutable Torch TensorRT module with persistent cache
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Leveraging engine caching, we are able to shortcut the engine compilation and save much time.
import os

from torch_tensorrt.dynamo._defaults import TIMING_CACHE_PATH

model = models.resnet18(pretrained=True).eval().to("cuda")

times = []
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

example_inputs = (torch.randn((100, 3, 224, 224)).to("cuda"),)
model = torch_trt.MutableTorchTensorRTModule(
    model,
    use_python_runtime=True,
    enabled_precisions={torch.float},
    debug=True,
    min_block_size=1,
    immutable_weights=False,
    cache_built_engines=True,
    reuse_cached_engines=True,
    engine_cache_size=1 << 30,  # 1GB
)


def remove_timing_cache(path=TIMING_CACHE_PATH):
    if os.path.exists(path):
        os.remove(path)


remove_timing_cache()

for i in range(4):
    inputs = [torch.rand((100 + i, 3, 224, 224)).to("cuda")]

    start.record()
    model(*inputs)  # Recompile
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))

print("----------------dynamo_compile----------------")
print("Without engine caching, used:", times[0], "ms")
print("With engine caching used:", times[1], "ms")
print("With engine caching used:", times[2], "ms")
print("With engine caching used:", times[3], "ms")
