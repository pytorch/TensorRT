import argparse
import gc
import os
import re
import sys
import time

import gradio as gr
import modelopt.torch.quantization as mtq
import torch
import torch_tensorrt
from accelerate.hooks import remove_hook_from_module
from diffusers import FluxPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

# Register SDPA as a standalone operator. Converter and lowering pass are defined in register_sdpa.py
sys.path.append(os.path.join(os.path.dirname(__file__), "../dynamo"))
from register_sdpa import *

DEVICE = "cuda:0"


def print_peak_memory(stage, debug=False):
    if debug:
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
        print(
            f"Peak memory allocated during {stage}: {peak_memory=}GB {peak_reserved=}GB"
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()


def compile_model(
    args,
) -> tuple[
    FluxPipeline,
    FluxTransformer2DModel,
    Union[torch_tensorrt.MutableTorchTensorRTModule, torch.fx.GraphModule],
]:
    original_dtype = torch.float16
    use_explicit_typing = args.use_explicit_typing
    if args.dtype == "fp8":
        enabled_precisions = {torch.float8_e4m3fn, torch.float16}
        ptq_config = mtq.FP8_DEFAULT_CFG
    elif args.dtype == "fp4":
        enabled_precisions = {torch.float32}
        if args.fp4_mha:
            ptq_config = mtq.NVFP4_FP8_MHA_CONFIG
        else:
            ptq_config = mtq.NVFP4_DEFAULT_CFG
        # fp4 requires explicit typing
        use_explicit_typing = True
    elif args.dtype == "int8":
        enabled_precisions = {torch.int8, torch.float16}
        ptq_config = mtq.INT8_DEFAULT_CFG
        # ptq_config["quant_cfg"]["*weight_quantizer"]["axis"] = None

    elif args.dtype == "fp16":
        enabled_precisions = {torch.float16}
    elif args.dtype == "bf16":
        enabled_precisions = {torch.bfloat16}
        original_dtype = torch.bfloat16

    print(f"\nUsing {args.dtype}")

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=original_dtype,
    ).to(original_dtype)

    if args.low_vram_mode:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(DEVICE)

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
            prompt="a dog running in a park",
        )

    if args.dtype not in ["fp16", "bf16"]:
        print_peak_memory("before mtq.quantize()", args.debug)
        backbone = mtq.quantize(backbone, ptq_config, forward_loop)
        mtq.disable_quantizer(backbone, filter_func)
        print_peak_memory("after mtq.quantize()", args.debug)

    batch_size = 2 if args.dynamic_shapes else 1
    if args.dynamic_shapes:
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
    else:
        dynamic_shapes = None

    settings = {
        "strict": False,
        "allow_complex_guards_as_runtime_asserts": True,
        "enabled_precisions": enabled_precisions,
        "truncate_double": True,
        "min_block_size": 1,
        "use_python_runtime": False,
        "immutable_weights": False,
        "offload_module_to_cpu": args.low_vram_mode,
        "debug": args.debug,
        "use_explicit_typing": use_explicit_typing,
    }
    if args.low_vram_mode:
        pipe.remove_all_hooks()
        pipe.enable_sequential_cpu_offload()
        remove_hook_from_module(pipe.transformer, recurse=True)
        pipe.transformer.to(DEVICE)

    print_peak_memory("before TRT", args.debug)
    if args.use_torch_dynamo_compile:
        with torch.no_grad():
            with export_torch_mode():
                dummy_inputs = {
                    "hidden_states": torch.randn(
                        (batch_size, 4096, 64), dtype=original_dtype
                    ).to(DEVICE),
                    "encoder_hidden_states": torch.randn(
                        (batch_size, 512, 4096), dtype=original_dtype
                    ).to(DEVICE),
                    "pooled_projections": torch.randn(
                        (batch_size, 768), dtype=original_dtype
                    ).to(DEVICE),
                    "timestep": torch.tensor(
                        [1.0] * batch_size, dtype=original_dtype
                    ).to(DEVICE),
                    "txt_ids": torch.randn((512, 3), dtype=original_dtype).to(DEVICE),
                    "img_ids": torch.randn((4096, 3), dtype=original_dtype).to(DEVICE),
                    "guidance": torch.tensor(
                        [1.0] * batch_size, dtype=torch.float32
                    ).to(DEVICE),
                    "joint_attention_kwargs": {},
                }
                ep = torch.export.export(
                    backbone,
                    args=(),
                    kwargs=dummy_inputs,
                    dynamic=dynamic_shapes,
                    strict=False,
                    allow_complex_guards_as_runtime_asserts=True,
                )
                trt_gm = torch_tensorrt.dynamo.compile(
                    ep, inputs=dummy_inputs, **settings
                )
    else:
        trt_gm = torch_tensorrt.MutableTorchTensorRTModule(backbone, **settings)
        if dynamic_shapes:
            trt_gm.set_expected_dynamic_shape_range((), dynamic_shapes)
    pipe.transformer = trt_gm
    print_peak_memory("after TRT", args.debug)

    image = pipe(
        "Test",
        output_type="pil",
        num_inference_steps=2,
        num_images_per_prompt=batch_size,
    ).images

    torch.cuda.empty_cache()

    if args.low_vram_mode:
        pipe.remove_all_hooks()
        pipe.to(DEVICE)

    return pipe, backbone, trt_gm


def launch_gradio(pipeline, backbone, trt_gm):

    def generate_image(prompt, inference_step, batch_size=2):
        start_time = time.time()
        image = pipeline(
            prompt,
            output_type="pil",
            num_inference_steps=inference_step,
            num_images_per_prompt=batch_size,
        ).images
        end_time = time.time()
        return image, end_time - start_time

    def model_change(model):
        if model == "Torch Model":
            pipeline.transformer = backbone
            backbone.to(DEVICE)
        else:
            backbone.to("cpu")
            pipeline.transformer = trt_gm
            torch.cuda.empty_cache()

    def load_lora(path):
        pipeline.load_lora_weights(
            path,
            adapter_name="lora1",
        )
        pipeline.set_adapters(["lora1"], adapter_weights=[1])
        pipeline.fuse_lora()
        pipeline.unload_lora_weights()
        print("LoRA loaded! Begin refitting")
        generate_image(pipeline, ["Test"], 2)
        print("Refitting Finished!")

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
                    placeholder="Enter the LoRA checkpoint path here. It could be a local path or a Hugging Face URL.",
                    value="gokaygokay/Flux-Engrave-LoRA",
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
        demo.launch()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Flux quantization with different dtypes"
    )

    parser.add_argument(
        "--dtype",
        choices=["fp8", "int8", "fp16", "fp4", "bf16"],
        default="fp16",
        help="Select the data type to use (fp8 or int8 or fp16 or fp4 or bf16)",
    )
    parser.add_argument(
        "--fp4_mha",
        action="store_true",
        help="Use FP4 MHA",
    )
    parser.add_argument(
        "--use_explicit_typing",
        action="store_true",
        help="Use explicit typing",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode",
    )
    parser.add_argument(
        "--low_vram_mode",
        action="store_true",
        help="Use low VRAM mode when you have a small GPU (<=32GB)",
    )
    parser.add_argument(
        "--dynamic_shapes",
        "-d",
        action="store_true",
        help="Use dynamic shapes",
    )
    parser.add_argument(
        "--use_torch_dynamo_compile",
        action="store_true",
        help="Use torch.dynamo.compile() to compile the model",
    )
    args = parser.parse_args()
    return args


def main(args):
    pipe, backbone, trt_gm = compile_model(args)
    launch_gradio(pipe, backbone, trt_gm)


# Launch the interface
if __name__ == "__main__":
    args = parse_args()
    main(args)
