# %%
# Import the following libraries
# -----------------------------
import re

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import torch
import torch_tensorrt
from diffusers import FluxPipeline
from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from modelopt.torch.quantization.utils import export_torch_mode
from torch.export._trace import _export
from transformers import AutoModelForCausalLM

# %%
DEVICE = "cuda:0"
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float32,
)
pipe.transformer = FluxTransformer2DModel(
    num_layers=1, num_single_layers=1, guidance_embeds=True
)

pipe.to(DEVICE).to(torch.float32)
# Store the config and transformer backbone
config = pipe.transformer.config
# global backbone
backbone = pipe.transformer
backbone.eval()


def filter_func(name):
    pattern = re.compile(
        r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding|pos_embed|time_text_embed|context_embedder|norm_out|x_embedder).*"
    )
    return pattern.match(name) is not None


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


generate_image(pipe, ["A golden retriever holding a sign to code"], "dog_code")

# %%
# Quantization


def do_calibrate(
    pipe,
    prompt: str,
) -> None:
    """
    Run calibration steps on the pipeline using the given prompts.
    """
    image = pipe(
        prompt,
        output_type="pil",
        num_inference_steps=20,
        generator=torch.Generator("cuda").manual_seed(0),
    ).images[0]


def forward_loop(mod):
    # Switch the pipeline's backbone, run calibration
    pipe.transformer = mod
    do_calibrate(
        pipe=pipe,
        prompt="test",
    )


ptq_config = mtq.FP8_DEFAULT_CFG
backbone = mtq.quantize(backbone, ptq_config, forward_loop)
mtq.disable_quantizer(backbone, filter_func)


# %%
# Export the backbone using torch.export
# --------------------------------------------------
# Define the dummy inputs and their respective dynamic shapes. We export the transformer backbone with dynamic shapes with a ``batch_size=2``
# due to `0/1 specialization <https://docs.google.com/document/d/16VPOa3d-Liikf48teAOmxLc92rgvJdfosIy-yoT38Io/edit?fbclid=IwAR3HNwmmexcitV0pbZm_x1a4ykdXZ9th_eJWK-3hBtVgKnrkmemz6Pm5jRQ&tab=t.0#heading=h.ez923tomjvyk>`_

batch_size = 2
BATCH = torch.export.Dim("batch", min=1, max=2)
SEQ_LEN = torch.export.Dim("seq_len", min=1, max=512)
# This particular min, max values for img_id input are recommended by torch dynamo during the export of the model.
# To see this recommendation, you can try exporting using min=1, max=4096
IMG_ID = torch.export.Dim("img_id", min=3586, max=4096)
dynamic_shapes = {
    "hidden_states": {0: BATCH},
    "encoder_hidden_states": {0: BATCH, 1: SEQ_LEN},
    "pooled_projections": {0: BATCH},
    "timestep": {0: BATCH},
    "txt_ids": {0: SEQ_LEN},
    "img_ids": {0: IMG_ID},
    "guidance": {0: BATCH},
    "joint_attention_kwargs": {},
    "return_dict": None,
}
# The guidance factor is of type torch.float32
dummy_inputs = {
    "hidden_states": torch.randn((batch_size, 4096, 64), dtype=torch.float32).to(
        DEVICE
    ),
    "encoder_hidden_states": torch.randn(
        (batch_size, 512, 4096), dtype=torch.float32
    ).to(DEVICE),
    "pooled_projections": torch.randn((batch_size, 768), dtype=torch.float32).to(
        DEVICE
    ),
    "timestep": torch.tensor([1.0, 1.0], dtype=torch.float32).to(DEVICE),
    "txt_ids": torch.randn((512, 3), dtype=torch.float32).to(DEVICE),
    "img_ids": torch.randn((4096, 3), dtype=torch.float32).to(DEVICE),
    "guidance": torch.tensor([1.0, 1.0], dtype=torch.float32).to(DEVICE),
    "joint_attention_kwargs": {},
    "return_dict": False,
}

# This will create an exported program which is going to be compiled with Torch-TensorRT
with export_torch_mode():
    ep = _export(
        backbone,
        args=(),
        kwargs=dummy_inputs,
        dynamic_shapes=dynamic_shapes,
        strict=False,
        allow_complex_guards_as_runtime_asserts=True,
    )

with torch_tensorrt.logging.debug():
    trt_gm = torch_tensorrt.dynamo.compile(
        ep,
        inputs=dummy_inputs,
        enabled_precisions={torch.float8_e4m3fn},
        truncate_double=True,
        min_block_size=1,
        debug=False,
        use_python_runtime=True,
        immutable_weights=True,
        offload_module_to_cpu=True,
    )


del ep
pipe.transformer = trt_gm
pipe.transformer.config = config


# %%
trt_gm.device = torch.device(DEVICE)
# Function which generates images from the flux pipeline

for _ in range(2):
    generate_image(pipe, ["A golden retriever holding a sign to code"], "dog_code")

# For this dummy model, the fp16 engine size is around 1GB, fp32 engine size is around 2GB
