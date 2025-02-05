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
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16,
)

pipe.to(device).to(torch.float16)
config = pipe.transformer.config
# from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
# pipe.transformer = FluxTransformer2DModel(patch_size=1, in_channels=64, num_layers=1, num_single_layers=1, guidance_embeds=True).to("cuda:0").to(torch.float16)
backbone = pipe.transformer
# generate_image(pipe, ["A cat holding a sign that says hello world"], "flux-dev")
# breakpoint()
# mto.restore(backbone, "./schnell_fp8.pt")

batch_size = 2
BATCH = torch.export.Dim("batch", min=1, max=2)
SEQ_LEN = torch.export.Dim("seq_len", min=1, max=512)
IMG_ID = torch.export.Dim("img_id", min=3586, max=4096)


dynamic_shapes = {
    "hidden_states": {0: BATCH},
    "encoder_hidden_states": {0: BATCH, 1: SEQ_LEN},
    "pooled_projections": {0: BATCH},
    "timestep": {0: BATCH},
    "txt_ids": {0: SEQ_LEN},
    "img_ids": {0: IMG_ID},
    "guidance": {0: BATCH},
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
    "txt_ids": torch.randn((512, 3), dtype=torch.float16).to(device),
    "img_ids": torch.randn((4096, 3), dtype=torch.float16).to(device),
    "guidance": torch.tensor([1.0, 1.0], dtype=torch.float32).to(device),
}
# with export_torch_mode():
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
        enabled_precisions={torch.float32},
        truncate_double=True,
        dryrun=False,
        min_block_size=1,
        # use_python_runtime=True,
        debug=True,
        use_fp32_acc=True,
        use_explicit_typing=True,
    )
# breakpoint()
# out_pyt = backbone(**dummy_inputs)
# out_trt = trt_gm(**dummy_inputs)
breakpoint()


class TRTModule(torch.nn.Module):
    def __init__(self, trt_mod):
        super(TRTModule, self).__init__()
        self.trt_mod = trt_mod

    def __call__(self, *args, **kwargs):
        # breakpoint()
        kwargs.pop("joint_attention_kwargs")
        kwargs.pop("return_dict")

        return self.trt_mod(**kwargs)


backbone.to("cpu")
pipe.transformer = TRTModule(trt_gm)
pipe.transformer.config = config

# Generate an image
generate_image(pipe, ["A cat holding a sign that says hello world"], "flux-dev")
