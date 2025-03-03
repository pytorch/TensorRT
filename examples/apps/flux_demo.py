from typing import Any

import gradio as gr
import torch
import torch_tensorrt
from diffusers import FluxPipeline
from torch.export._trace import _export

# %%
# Define the FLUX-1.dev model
# -----------------------------
# Load the ``FLUX-1.dev`` pretrained pipeline using ``FluxPipeline`` class.
# ``FluxPipeline`` includes different components such as ``transformer``, ``vae``, ``text_encoder``, ``tokenizer`` and ``scheduler`` necessary
# to generate an image. We load the weights in ``FP16`` precision using ``torch_dtype`` argument
DEVICE = "cuda:0"
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16,
)
pipe.to(DEVICE).to(torch.float16)
# Store the config and transformer backbone
config = pipe.transformer.config
backbone = pipe.transformer


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

dummy_inputs = {
    "hidden_states": torch.randn((batch_size, 4096, 64), dtype=torch.float16).to(
        DEVICE
    ),
    "encoder_hidden_states": torch.randn(
        (batch_size, 512, 4096), dtype=torch.float16
    ).to(DEVICE),
    "pooled_projections": torch.randn((batch_size, 768), dtype=torch.float16).to(
        DEVICE
    ),
    "timestep": torch.tensor([1.0, 1.0], dtype=torch.float16).to(DEVICE),
    "txt_ids": torch.randn((512, 3), dtype=torch.float16).to(DEVICE),
    "img_ids": torch.randn((4096, 3), dtype=torch.float16).to(DEVICE),
    "guidance": torch.tensor([1.0, 1.0], dtype=torch.float32).to(DEVICE),
    "joint_attention_kwargs": {},
    "return_dict": False,
}
# This will create an exported program which is going to be compiled with Torch-TensorRT
ep = _export(
    backbone,
    args=(),
    kwargs=dummy_inputs,
    dynamic_shapes=dynamic_shapes,
    strict=False,
    allow_complex_guards_as_runtime_asserts=True,
)

trt_gm = torch_tensorrt.dynamo.compile(
    ep,
    inputs=dummy_inputs,
    enabled_precisions={torch.float32},
    truncate_double=True,
    min_block_size=1,
    use_fp32_acc=True,
    use_explicit_typing=True,
    debug=False,
    use_python_runtime=True,
)
backbone.to("cpu")
del ep
pipe.transformer = trt_gm
pipe.transformer.config = config
trt_gm.device = torch.device("cuda")
torch.cuda.empty_cache()


def generate_image(prompt: str, inference_step: int) -> Any:
    """Generate image from text prompt using Stable Diffusion."""
    image = pipe(
        prompt,
        output_type="pil",
        num_inference_steps=inference_step,
        generator=torch.Generator("cuda"),
    ).images[0]
    return image


def model_change(model: str) -> None:
    if model == "Torch Model":
        pipe.transformer = backbone
        backbone.to(DEVICE)
    else:
        backbone.to("cpu")
        pipe.transformer = trt_gm
        torch.cuda.empty_cache()


# Create Gradio interface
with gr.Blocks(title="Flux Demo with Torch-TensorRT") as demo:
    gr.Markdown("# Flux Image Generation Demo Accelerated by Torch-TensorRT")

    with gr.Row():
        with gr.Column():
            # Input components
            prompt_input = gr.Textbox(
                label="Prompt", placeholder="Enter your prompt here...", lines=3
            )
            model_dropdown = gr.Dropdown(
                choices=["Torch Model", "Torch-TensorRT Accelerated Model"],
                value="Torch-TensorRT Accelerated Model",
                label="Model Variant",
            )

            lora_upload = gr.File(
                label="Upload LoRA weights (.safetensors)", file_types=[".safetensors"]
            )
            num_steps = gr.Slider(
                minimum=20, maximum=100, value=20, step=1, label="Inference Steps"
            )

            generate_btn = gr.Button("Generate Image")

        with gr.Column():
            # Output component
            output_image = gr.Image(label="Generated Image")

    # Connect the button to the generation function
    model_dropdown.change(model_change, inputs=[model_dropdown])
    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt_input,
            num_steps,
        ],
        outputs=output_image,
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch()
