import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import torch
import torch_tensorrt
from diffusers import FluxPipeline
from modelopt.torch.quantization.utils import export_torch_mode

# from onnx_utils.export import generate_dummy_inputs
from torch.export._trace import _export


def generate_image(pipe, prompt, image_name):
    seed = 42
    image = pipe(
        prompt,
        output_type="pil",
        num_inference_steps=20,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]
    image.save(f"{image_name}.png")
    print(f"Image generated using {image_name} model saved as {image_name}.png")


device = "cuda"
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16,
)

pipe.to(device)
backbone = pipe.transformer

# Restore FP8 weights
mto.restore(backbone, "./schnell_fp8.pt")

# dummy_inputs = generate_dummy_inputs("flux-dev", "cuda", True)
batch_size = 1
BATCH = torch.export.Dim("batch", min=1, max=2)
SEQ_LEN = torch.export.Dim("seq_len", min=1, max=256)
dynamic_shapes = (
    {0: BATCH},
    {0: BATCH, 1: SEQ_LEN},
    {0: BATCH},
    {0: BATCH},
    {0: BATCH},
    {0: BATCH, 1: SEQ_LEN},
)

dummy_inputs = (
    torch.randn((batch_size, 4096, 64), dtype=torch.float16).to(device),
    torch.randn((batch_size, 256, 4096), dtype=torch.float16).to(device),
    torch.randn((batch_size, 768), dtype=torch.float16).to(device),
    torch.tensor([1.0, 1.0], dtype=torch.float16).to(device),
    torch.randn((batch_size, 4096, 3), dtype=torch.float16).to(device),
    torch.randn((batch_size, 256, 3), dtype=torch.float16).to(device),
)
with export_torch_mode():
    ep = _export(
        backbone,
        dummy_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=False,
        allow_complex_guards_as_runtime_asserts=True,
    )

with torch_tensorrt.logging.debug():
    trt_gm = torch_tensorrt.dynamo.compile(
        ep,
        inputs=dummy_inputs,
        enabled_precisions={torch.float8_e4m3fn, torch.float16},
        truncate_double=True,
        dryrun=True,
        debug=True,
    )


backbone.to("cpu")
config = pipe.transformer.config
pipe.transformer = trt_gm
pipe.transformer.config = config

# Generate an image
generate_image(pipe, "A cat holding a sign that says hello world", "flux-dev")
