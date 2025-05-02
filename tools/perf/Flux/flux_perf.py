from time import time

import torch
import torch_tensorrt
from diffusers import FluxPipeline

for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(i).name)

DEVICE = "cuda:0"
# pipe = FluxPipeline.from_pretrained(
#     "black-forest-labs/FLUX.1-dev",
#     torch_dtype=torch.float32,
# )
pipe.to(DEVICE).to(torch.float32)
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
    "enabled_precisions": {torch.float32},
    "truncate_double": True,
    "min_block_size": 1,
    "use_fp32_acc": True,
    "use_explicit_typing": True,
    "debug": False,
    "use_python_runtime": True,
    "immutable_weights": False,
}


def generate_image(prompt, inference_step, batch_size=2, benchmark=False, iterations=1):

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
        print("Time Elapse for", iterations, "iterations:", end - start)
        print(
            "Average Latency Per Step:",
            (end - start) / inference_step / iterations / batchsize,
        )
    return image


generate_image(["Test"], 2)
print("Benchmark Original PyTorch Module Latency (float32)")
generate_image(["Test"], 50, benchmark=True, iterations=3)

pipe.to(torch.float16)
print("Benchmark Original PyTorch Module Latency (float16)")
generate_image(["Test"], 50, benchmark=True, iterations=3)


trt_gm = torch_tensorrt.MutableTorchTensorRTModule(backbone, **settings)
trt_gm.set_expected_dynamic_shape_range((), dynamic_shapes)
pipe.transformer = trt_gm

start = time()
generate_image(["Test"], 2)
end = time()
print("Time Elapse compilation:", end - start)
print()
print("Benchmark TRT Accelerated Latency")
generate_image(["Test"], 50, benchmark=True, iterations=3)
torch.cuda.empty_cache()


with torch_tensorrt.runtime.enable_cudagraphs(trt_gm):
    generate_image(["Test"], 2)
    print("Benchmark TRT Accelerated Latency with Cuda Graph")
    generate_image(["Test"], 50, benchmark=True, iterations=3)
