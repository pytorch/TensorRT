import argparse
import re
import time

import gradio as gr
import modelopt.torch.quantization as mtq
import torch
import torch_tensorrt
from diffusers import FluxPipeline

parser = argparse.ArgumentParser(
    description="Run Flux quantization with different dtypes"
)

parser.add_argument(
    "--dtype",
    choices=["fp8", "int8", "fp16"],
    default="fp16",
    help="Select the data type to use (fp8 or int8 or fp16)",
)
args = parser.parse_args()
# Update enabled precisions based on dtype argument

if args.dtype == "fp8":
    enabled_precisions = {torch.float8_e4m3fn, torch.float16}
    ptq_config = mtq.FP8_DEFAULT_CFG
elif args.dtype == "int8":
    enabled_precisions = {torch.int8, torch.float16}
    ptq_config = mtq.INT8_DEFAULT_CFG
    ptq_config["quant_cfg"]["*weight_quantizer"]["axis"] = None
elif args.dtype == "fp16":
    enabled_precisions = {torch.float16}
print(f"\nUsing {args.dtype}")


DEVICE = "cuda:0"
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16,
)


pipe.to(DEVICE).to(torch.float16)
backbone = pipe.transformer
backbone.eval()


def filter_func(name):
    pattern = re.compile(
        r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding|pos_embed|time_text_embed|context_embedder|norm_out|x_embedder).*"
    )
    return pattern.match(name) is not None


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


if args.dtype != "fp16":
    backbone = mtq.quantize(backbone, ptq_config, forward_loop)
    mtq.disable_quantizer(backbone, filter_func)

batch_size = 2

BATCH = torch.export.Dim("batch", min=1, max=8)
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
    "enabled_precisions": enabled_precisions,
    "truncate_double": True,
    "min_block_size": 1,
    "debug": False,
    "use_python_runtime": True,
    "immutable_weights": False,
    "offload_module_to_cpu": True,
}

trt_gm = torch_tensorrt.MutableTorchTensorRTModule(backbone, **settings)
trt_gm.set_expected_dynamic_shape_range((), dynamic_shapes)
pipe.transformer = trt_gm


def generate_image(prompt, inference_step, batch_size=2):
    start_time = time.time()
    image = pipe(
        prompt,
        output_type="pil",
        num_inference_steps=inference_step,
        num_images_per_prompt=batch_size,
    ).images
    end_time = time.time()
    return image, end_time - start_time


generate_image(["Test"], 2)
torch.cuda.empty_cache()


def model_change(model):
    if model == "Torch Model":
        pipe.transformer = backbone
        backbone.to(DEVICE)
    else:
        backbone.to("cpu")
        pipe.transformer = trt_gm
        torch.cuda.empty_cache()


def load_lora(path):

    pipe.load_lora_weights(
        path,
        adapter_name="lora1",
    )
    pipe.set_adapters(["lora1"], adapter_weights=[1])
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    print("LoRA loaded! Begin refitting")
    generate_image(["Test"], 2)
    print("Refitting Finished!")


load_lora("/home/TensorRT/examples/apps/NGRVNG.safetensors")


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

            lora_upload_path = gr.Textbox(
                label="LoRA Path",
                placeholder="Enter the LoRA checkpoint path here",
                value="/home/TensorRT/examples/apps/NGRVNG.safetensors",
                lines=2,
            )
            num_steps = gr.Slider(
                minimum=20, maximum=100, value=20, step=1, label="Inference Steps"
            )
            batch_size = gr.Slider(
                minimum=1, maximum=8, value=1, step=1, label="Batch Size"
            )

            generate_btn = gr.Button("Generate Image")
            load_lora_btn = gr.Button("Load LoRA")

        with gr.Column():
            # Output component
            output_image = gr.Gallery(label="Generated Image")
            time_taken = gr.Textbox(
                label="Generation Time (seconds)", interactive=False
            )

    # Connect the button to the generation function
    model_dropdown.change(model_change, inputs=[model_dropdown])
    load_lora_btn.click(
        fn=load_lora,
        inputs=[
            lora_upload_path,
        ],
    )

    # Update generate button click to include time output
    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt_input,
            num_steps,
            batch_size,
        ],
        outputs=[output_image, time_taken],
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch()
