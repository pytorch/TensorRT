# https://huggingface.co/black-forest-labs/FLUX.1-schnell
import torch
from diffusers import FluxPipeline
import torch_tensorrt

device = "cuda:0"
backend = "torch_tensorrt"

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16, device_map="balanced", max_memory={0: "32GB"})

# pipe = pipe.to(device)
pipe.reset_device_map()
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# Optimize the transformer portion with Torch-TensorRT
pipe.transformer = torch.compile(
    pipe.transformer,
    backend=backend,
    options={
        "truncate_long_and_double": True,
        "enabled_precisions": {torch.float16},
        # "use_fp32_acc": True,
    },
    dynamic=False,
)

# pipe.transformer.config['num_layers'] = 5
# pipe.transformer.config.num_layers = 5

prompt = "A cat holding a sign that says hello world"

with torch_tensorrt.logging.debug():
    image = pipe(
        prompt,
        guidance_scale=0.0,
        num_inference_steps=4,
        max_sequence_length=128,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]

image.save("images/flux-schnell.png")