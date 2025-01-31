# import modelopt.torch.opt as mto
# import modelopt.torch.quantization as mtq
# from modelopt.torch.quantization.utils import export_torch_mode
import torch
import torch_tensorrt
from diffusers import (
    DiffusionPipeline,
    FluxPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionPipeline,
)

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
breakpoint()
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16,
)

breakpoint()
pipe.to(device)
pipe.to(torch.float16)
backbone = pipe.transformer

# mto.restore(backbone, "./schnell_fp8.pt")

# dummy_inputs = generate_dummy_inputs("flux-dev", "cuda", True)
batch_size = 2
BATCH = torch.export.Dim("batch", min=1, max=2)
SEQ_LEN = torch.export.Dim("seq_len", min=1, max=512)
IMG_ID = torch.export.Dim("img_id", min=3586, max=4096)
# dynamic_shapes = (
#            {0: BATCH},
#            {0: BATCH},
#            {0: BATCH},
#            {0: BATCH},
#            {0: BATCH, 1: SEQ_LEN},
#            {0: BATCH, 1: SEQ_LEN},
#            {0: BATCH},
#            {}
#        )
#
# dummy_inputs = (
#            torch.randn((batch_size, 4096, 64), dtype=torch.float16).to(device),
#            torch.tensor([1.0, 1.0], dtype=torch.float16).to(device),
#            torch.tensor([1.0, 1.0], dtype=torch.float16).to(device),
#            torch.randn((batch_size, 768), dtype=torch.float16).to(device),
#            torch.randn((batch_size, 512, 4096), dtype=torch.float16).to(device),
#            torch.randn((batch_size, 512, 3), dtype=torch.float16).to(device),
#            torch.randn((batch_size, 4096, 3), dtype=torch.float16).to(device),
#        )

dynamic_shapes = {
    "hidden_states": {0: BATCH},
    "encoder_hidden_states": {0: BATCH, 1: SEQ_LEN},
    "pooled_projections": {0: BATCH},
    "timestep": {0: BATCH},
    "txt_ids": {0: BATCH, 1: SEQ_LEN},
    "img_ids": {0: BATCH, 1: IMG_ID},
    "guidance": {0: BATCH},
    # "joint_attention_kwargs": {},
    # "return_dict": {}
}

dummy_inputs = {
    "hidden_states": torch.randn((batch_size, 4096, 64), dtype=torch.float16).to(
        device
    ),
    "encoder_hidden_states": torch.randn(
        (batch_size, 512, 4096), dtype=torch.float16
    ).to(device),
    "pooled_projections": torch.randn((batch_size, 768), dtype=torch.float16).to(
        device
    ),
    "timestep": torch.tensor([1.0, 1.0], dtype=torch.float16).to(device),
    "txt_ids": torch.randn((batch_size, 512, 3), dtype=torch.float16).to(device),
    "img_ids": torch.randn((batch_size, 4096, 3), dtype=torch.float16).to(device),
    "guidance": torch.tensor([1.0, 1.0], dtype=torch.float16).to(device),
    # "joint_attention_kwargs": {},
    # "return_dict": torch.tensor(False)
}
with export_torch_mode():
    ep = _export(
        backbone,
        args=(),
        kwargs=dummy_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=False,
        allow_complex_guards_as_runtime_asserts=True,
    )

# breakpoint()
with torch_tensorrt.logging.debug():
    trt_gm = torch_tensorrt.dynamo.compile(
        ep,
        inputs=dummy_inputs,
        enabled_precisions={torch.float16},
        truncate_double=True,
        dryrun=False,
        min_block_size=1,
        debug=True,
    )

breakpoint()
backbone.to("cpu")
config = pipe.transformer.config
pipe.transformer = trt_gm
pipe.transformer.config = config

# Generate an image
generate_image(pipe, "A cat holding a sign that says hello world", "flux-dev")
