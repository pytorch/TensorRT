from time import time

import register_sdpa
import torch
import torch_tensorrt
from diffusers import FluxPipeline

for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(i).name)

DEVICE = "cuda:0"
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
)
pipe.to(DEVICE).to(torch.bfloat16)
backbone = pipe.transformer


batch_size = 2
BATCH = torch.export.Dim("batch", min=1, max=8)

# This particular min, max values for img_id input are recommended by torch dynamo during the export of the model.
# To see this recommendation, you can try exporting using min=1, max=4096
dynamic_shapes = {
    "hidden_states": {0: BATCH},
    "encoder_hidden_states": {0: BATCH},
    "pooled_projections": {0: BATCH},
    "timestep": {0: BATCH},
    "txt_ids": {},
    "img_ids": {},
    "guidance": {0: BATCH},
    "joint_attention_kwargs": {},
    "return_dict": None,
}

settings = {
    "strict": False,
    "allow_complex_guards_as_runtime_asserts": True,
    # "enabled_precisions": {torch.float16},
    use_explicit_typing: True,
    "truncate_double": True,
    "min_block_size": 1,
    "debug": False,
    # "use_python_runtime": True,
    "immutable_weights": False,
    "offload_module_to_cpu": True,
}


def generate_image(prompt, inference_step, batch_size=1, benchmark=False, iterations=1):

    start = time()
    for i in range(iterations):
        image = pipe(
            prompt,
            output_type="pil",
            num_inference_steps=inference_step,
            num_images_per_prompt=batch_size,
        ).images
    end = time()
    if benchmark:
        print(f"Batch Size: {batch_size}")
        print("Time Elapse for", iterations, "iterations:", end - start)
        print(
            "Average Latency Per Step:",
            (end - start) / inference_step / iterations / batch_size,
        )
    return image


pipe.to(torch.bfloat16)
torch.cuda.empty_cache()
# Warmup
generate_image(["Test"], 20)
print("Benchmark Original PyTorch Module Latency (bfloat16)")
for batch_size in range(1, 3):
    generate_image(["Test"], 20, batch_size=batch_size, benchmark=True, iterations=3)

pipe.to(torch.float16)
print("Benchmark Original PyTorch Module Latency (float16)")
for batch_size in range(1, 3):
    generate_image(["Test"], 20, batch_size=batch_size, benchmark=True, iterations=3)

trt_gm = torch_tensorrt.MutableTorchTensorRTModule(backbone, **settings)
trt_gm.set_expected_dynamic_shape_range((), dynamic_shapes)
pipe.transformer = trt_gm

start = time()
generate_image(["Test"], 2, batch_size=2)
end = time()
print("Time Elapse compilation:", end - start)
print()
print("Benchmark TRT Accelerated Latency")
for batch_size in range(1, 3):
    generate_image(["Test"], 20, batch_size=batch_size, benchmark=True, iterations=3)
torch.cuda.empty_cache()
