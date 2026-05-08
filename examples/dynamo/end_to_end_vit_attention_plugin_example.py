"""
End-to-End ViT Attention Plugin Example

This mirrors the structure of end_to_end_llm_generation_example.py, but for the
visual towers used by production multimodal models:

1. load the TensorRT-Edge-LLM plugin shared library
2. register the PyTorch custom op and Torch-TensorRT converter
3. load a PyTorch reference model
4. load a second model, replace visual attention with plugin attention
5. wrap and compile the visual model
6. verify output shape and run a small latency benchmark

Supported end-to-end paths:
- Qwen2.5-VL visual tower
- Meta Llama 3.2 Vision / HuggingFace Mllama vision tower

By default this script runs both benchmarks. Use ``--model-type qwen_vl`` or
``--model-type mllama`` to run only one path.
"""

import argparse
import os
import sys

import torch
from transformers import AutoConfig, MllamaVisionModel, Qwen2_5_VLForConditionalGeneration

# Add tools/llm to path for shared plugin utilities, matching the LLM example style.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../tools/llm"))

from plugin_utils_vit import (
    ViTPluginAttention,
    ViTPluginWrapper,
    compile_vit_plugin_model,
    get_vit_plugin_config,
    load_plugin,
    measure_vit_latency,
    register_vit_plugin_op,
    replace_vit_attention_with_plugin,
    set_vit_plugin_config,
)

# Configuration
QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
MLLAMA_MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision"
DTYPE = torch.float16
DEVICE = torch.device("cuda:0")

# One image grid used for the compile example: t, h, w.
# Qwen2.5-VL visual input is already patchified as [t*h*w, patch_dim].
IMAGE_GRID_THW = (1, 8, 16)

# Load the plugin and register the op/converter path.
load_plugin()
register_vit_plugin_op()


# -----------------------------------------------------------------------------
# Backward Compatibility Exports
# -----------------------------------------------------------------------------

# Export the attention wrapper under a short name for parity with the LLM example.
PluginAttention = ViTPluginAttention

# Global config for compatibility with converter-style imports.
TARGET_CONFIG = None

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def get_qwen_vision_config(config):
    """Return the Qwen2.5-VL vision config from the top-level config."""
    if hasattr(config, "vision_config"):
        return config.vision_config
    if hasattr(config, "visual"):
        return config.visual
    raise ValueError("Cannot find Qwen-VL vision config")

def set_plugin_config_from_qwen_vl(vision_config):
    """Set ViT plugin fields from Qwen2.5-VL vision config."""
    head_dim = vision_config.hidden_size // vision_config.num_heads
    num_patches = IMAGE_GRID_THW[0] * IMAGE_GRID_THW[1] * IMAGE_GRID_THW[2]
    set_vit_plugin_config(
        num_attention_heads=vision_config.num_heads,
        head_dim=head_dim,
        num_patches=num_patches,
    )


def get_mllama_vision_config(config):
    """Return the official Llama 3.2 Vision/Mllama vision config."""
    if hasattr(config, "vision_config"):
        return config.vision_config
    if hasattr(config, "attention_heads") and hasattr(config, "max_num_tiles"):
        return config
    raise ValueError("Cannot find Mllama/Llama Vision config")


def set_plugin_config_from_mllama(vision_config):
    """Set ViT plugin fields from a Mllama/Llama Vision config."""
    num_heads = vision_config.attention_heads
    head_dim = vision_config.hidden_size // num_heads
    num_patches_per_tile = (vision_config.image_size // vision_config.patch_size) ** 2 + 1
    target_length = num_patches_per_tile + (8 - (num_patches_per_tile % 8)) % 8
    set_vit_plugin_config(
        num_attention_heads=num_heads,
        head_dim=head_dim,
        num_patches=vision_config.max_num_tiles * target_length,
    )


def create_qwen_vl_visual_inputs(vision_config):
    """Create dummy patchified Qwen2.5-VL visual inputs."""
    grid_t, grid_h, grid_w = IMAGE_GRID_THW
    num_patches = grid_t * grid_h * grid_w
    patch_dim = (
        vision_config.in_chans
        * vision_config.temporal_patch_size
        * vision_config.patch_size
        * vision_config.patch_size
    )
    pixel_values = torch.randn(
        num_patches,
        patch_dim,
        dtype=DTYPE,
        device=DEVICE,
    )
    image_grid_thw = torch.tensor([IMAGE_GRID_THW], dtype=torch.long, device=DEVICE)
    return pixel_values, image_grid_thw


def create_qwen_vl_visual_core_inputs(visual_model, pixel_values, image_grid_thw):
    """Precompute Qwen visual metadata that torch.export cannot trace."""
    with torch.no_grad():
        rotary_pos_emb = visual_model.rot_pos_emb(image_grid_thw)
        window_index, cu_window_seqlens = visual_model.get_window_index(image_grid_thw)

    window_index = window_index.to(device=DEVICE, dtype=torch.long)
    reverse_window_index = torch.argsort(window_index)

    seq_len = pixel_values.shape[0]
    attention_mask = torch.zeros(1, seq_len, seq_len, dtype=DTYPE, device=DEVICE)
    window_attention_mask = torch.full(
        (1, seq_len, seq_len),
        torch.finfo(DTYPE).min,
        dtype=DTYPE,
        device=DEVICE,
    )
    if isinstance(cu_window_seqlens, torch.Tensor):
        cu_window_seqlens = cu_window_seqlens.to(device="cpu", dtype=torch.long).tolist()
    starts = cu_window_seqlens[:-1]
    ends = cu_window_seqlens[1:]
    for start, end in zip(starts, ends):
        window_attention_mask[:, start:end, start:end] = 0

    return (
        rotary_pos_emb.to(device=DEVICE),
        attention_mask,
        window_attention_mask,
        window_index,
        reverse_window_index,
    )


def create_mllama_vision_inputs(vision_config):
    """
    Create dummy real-model inputs for MllamaVisionModel.

    HF Mllama/Llama 3.2 Vision expects:
    - pixel_values: [batch, max_num_images, max_num_tiles, channels, H, W]
    - aspect_ratio_ids: [batch, max_num_images]
    - aspect_ratio_mask: [batch, max_num_images, max_num_tiles]
    """
    batch_size = 1
    max_num_images = 1
    num_tiles = vision_config.max_num_tiles
    pixel_values = torch.randn(
        batch_size,
        max_num_images,
        num_tiles,
        vision_config.num_channels,
        vision_config.image_size,
        vision_config.image_size,
        dtype=DTYPE,
        device=DEVICE,
    )
    aspect_ratio_ids = torch.ones(
        batch_size,
        max_num_images,
        dtype=torch.long,
        device=DEVICE,
    )
    aspect_ratio_mask = torch.ones(
        batch_size,
        max_num_images,
        num_tiles,
        dtype=torch.long,
        device=DEVICE,
    )
    aspect_ratio_mask[:, :, 1:] = 0
    return pixel_values, aspect_ratio_ids, aspect_ratio_mask


def create_mllama_attention_mask(vision_config, aspect_ratio_mask):
    """
    Precompute Mllama's aspect-ratio attention mask outside torch.export.

    This mirrors HuggingFace _prepare_aspect_ratio_attention_mask but avoids
    compiling its in-place padding-mask update through TensorRT.
    """
    batch_size, max_num_images, max_num_tiles = aspect_ratio_mask.shape
    flat_mask = aspect_ratio_mask.reshape(batch_size * max_num_images, max_num_tiles)
    num_patches = (vision_config.image_size // vision_config.patch_size) ** 2 + 1
    target_length = num_patches + (8 - (num_patches % 8)) % 8

    attention_mask = flat_mask.view(
        batch_size * max_num_images,
        max_num_tiles,
        1,
        1,
    ).to(DTYPE)
    attention_mask = attention_mask.repeat(1, 1, target_length, 1)

    pad_patches = target_length - num_patches
    if pad_patches > 0:
        attention_mask[:, :, -pad_patches:] = 0

    attention_mask = 1 - attention_mask
    attention_mask = attention_mask.reshape(
        batch_size * max_num_images,
        max_num_tiles * target_length,
        1,
    )
    attention_mask = (
        attention_mask
        @ attention_mask.transpose(-1, -2)
        * torch.finfo(DTYPE).min
    )
    return attention_mask.unsqueeze(1).to(device=DEVICE, dtype=DTYPE)


def get_last_hidden_state(output):
    """Normalize HF model outputs to a tensor for verification and benchmark."""
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def benchmark_visual(model_func, inputs, run_name="TensorRT"):
    """Benchmark the compiled visual model."""
    pixel_values = inputs[0]
    if pixel_values.dim() == 2:
        input_desc = f"Patches: {pixel_values.shape[0]}"
    elif pixel_values.dim() == 6:
        input_desc = (
            f"Images: {pixel_values.shape[0]} | "
            f"Tiles: {pixel_values.shape[2]}"
        )
    else:
        input_desc = f"Input shape: {tuple(pixel_values.shape)}"

    def forward():
        with torch.no_grad():
            return get_last_hidden_state(model_func(*inputs))

    mean_ms, std_ms, median_ms = measure_vit_latency(forward)
    print(
        f"{run_name} | {input_desc} | "
        f"Mean: {mean_ms:.3f} ms | Median: {median_ms:.3f} ms | Std: {std_ms:.3f} ms"
    )
    return mean_ms


def call_qwen_visual(model_func, pixel_values, core_inputs):
    """Call a compiled Qwen visual wrapper with named inputs."""
    (
        rotary_pos_emb,
        attention_mask,
        window_attention_mask,
        window_index,
        reverse_window_index,
    ) = core_inputs
    try:
        return model_func(
            pixel_values=pixel_values,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=attention_mask,
            window_attention_mask=window_attention_mask,
            window_index=window_index,
            reverse_window_index=reverse_window_index,
        )
    except TypeError:
        return model_func(pixel_values, *core_inputs)


def call_mllama_visual(model_func, pixel_values, aspect_ratio_ids, attention_mask):
    """Call a compiled Mllama visual wrapper with named inputs."""
    try:
        return model_func(
            pixel_values=pixel_values,
            aspect_ratio_ids=aspect_ratio_ids,
            attention_mask=attention_mask,
        )
    except TypeError:
        return model_func(pixel_values, aspect_ratio_ids, attention_mask)


def verify_qwen_output(trt_model_func, visual_pytorch, pixel_values, image_grid_thw, core_inputs):
    """Compare PyTorch and TensorRT-plugin Qwen visual output."""
    print("\n=== Verifying Qwen2.5-VL Visual Output ===")

    with torch.no_grad():
        pyt_output = visual_pytorch(pixel_values, image_grid_thw)
        trt_output = call_qwen_visual(trt_model_func, pixel_values, core_inputs)

    print(f"PyTorch output shape:  {tuple(pyt_output.shape)}")
    print(f"TensorRT output shape: {tuple(trt_output.shape)}")

    if pyt_output.shape == trt_output.shape:
        print("SUCCESS: Output shapes match.")
    else:
        print("FAILURE: Output shapes differ.")
        return

    max_abs_diff = (pyt_output - trt_output).abs().max().item()
    print(f"Max absolute difference: {max_abs_diff:.6f}")
    print(
        "Note: small numerical differences are expected across PyTorch SDPA, "
        "TensorRT, and the custom CUDA plugin."
    )


def verify_mllama_output(trt_model_func, visual_pytorch, pyt_inputs, trt_inputs):
    """Compare PyTorch and TensorRT-plugin Mllama visual output."""
    print("\n=== Verifying Llama 3.2 Vision/Mllama Visual Output ===")

    with torch.no_grad():
        pyt_output = get_last_hidden_state(visual_pytorch(*pyt_inputs))
        trt_output = get_last_hidden_state(call_mllama_visual(trt_model_func, *trt_inputs))

    print(f"PyTorch output shape:  {tuple(pyt_output.shape)}")
    print(f"TensorRT output shape: {tuple(trt_output.shape)}")

    if pyt_output.shape == trt_output.shape:
        print("SUCCESS: Output shapes match.")
    else:
        print("FAILURE: Output shapes differ.")
        return

    max_abs_diff = (pyt_output - trt_output).abs().max().item()
    print(f"Max absolute difference: {max_abs_diff:.6f}")
    print(
        "Note: small numerical differences are expected across PyTorch SDPA, "
        "TensorRT, and the custom CUDA plugin."
    )


class MllamaVisionTensorOutputWrapper(torch.nn.Module):
    """Wrap MllamaVisionModel so torch.export sees a tensor output."""

    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values, aspect_ratio_ids, aspect_ratio_mask):
        output = self.vision_model(
            pixel_values,
            aspect_ratio_ids,
            aspect_ratio_mask,
        )
        return get_last_hidden_state(output)


class MllamaVisionReferenceCoreWrapper(torch.nn.Module):
    """Reference wrapper that keeps the native HF forward but returns a tensor."""

    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values, aspect_ratio_ids, aspect_ratio_mask):
        return get_last_hidden_state(
            self.vision_model(pixel_values, aspect_ratio_ids, aspect_ratio_mask)
        )


def run_qwen(model_name):
    print(f"Loading {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    vision_config = get_qwen_vision_config(config)

    globals()["TARGET_CONFIG"] = vision_config
    set_plugin_config_from_qwen_vl(vision_config)
    print(f"Plugin config: {get_vit_plugin_config()}")

    model_pytorch = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    model_pytorch.eval()

    model_trt = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    model_trt.eval()

    visual_trt = replace_vit_attention_with_plugin(model_trt.visual, vision_config)
    wrapper = ViTPluginWrapper(visual_trt, model_type="qwen_vl").eval()

    pixel_values, image_grid_thw = create_qwen_vl_visual_inputs(vision_config)
    core_inputs = create_qwen_vl_visual_core_inputs(
        model_trt.visual, pixel_values, image_grid_thw
    )

    print("Compiling TensorRT visual model...")
    qwen_inputs = (pixel_values, *core_inputs)
    qwen_kwargs = {
        "pixel_values": pixel_values,
        "rotary_pos_emb": core_inputs[0],
        "attention_mask": core_inputs[1],
        "window_attention_mask": core_inputs[2],
        "window_index": core_inputs[3],
        "reverse_window_index": core_inputs[4],
    }
    trt_model_func = compile_vit_plugin_model(
        wrapper,
        (),
        DEVICE,
        example_kwargs=qwen_kwargs,
        dynamic_shapes={name: {} for name in qwen_kwargs},
    )

    verify_qwen_output(
        trt_model_func,
        model_pytorch.visual,
        pixel_values,
        image_grid_thw,
        core_inputs,
    )

    print("\n=== Starting Qwen2.5-VL Visual Benchmark ===")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    trt_qwen_forward = lambda pixel_values, *inputs: call_qwen_visual(
        trt_model_func,
        pixel_values,
        inputs,
    )
    benchmark_visual(
        trt_qwen_forward, qwen_inputs, run_name="TensorRT"
    )


def run_mllama(model_name):
    print(f"Loading {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    vision_config = get_mllama_vision_config(config)

    globals()["TARGET_CONFIG"] = vision_config
    set_plugin_config_from_mllama(vision_config)
    print(f"Plugin config: {get_vit_plugin_config()}")

    visual_pytorch = MllamaVisionModel.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    visual_pytorch.eval()

    visual_trt = MllamaVisionModel.from_pretrained(
        model_name,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    visual_trt.eval()

    visual_trt = replace_vit_attention_with_plugin(
        visual_trt,
        vision_config,
    )
    wrapper = ViTPluginWrapper(visual_trt, model_type="mllama").eval()
    visual_pytorch_wrapper = MllamaVisionReferenceCoreWrapper(visual_pytorch).eval()

    pixel_values, aspect_ratio_ids, aspect_ratio_mask = create_mllama_vision_inputs(
        vision_config
    )
    attention_mask = create_mllama_attention_mask(vision_config, aspect_ratio_mask)
    trt_inputs = (pixel_values, aspect_ratio_ids, attention_mask)
    pyt_inputs = (pixel_values, aspect_ratio_ids, aspect_ratio_mask)
    trt_kwargs = {
        "pixel_values": pixel_values,
        "aspect_ratio_ids": aspect_ratio_ids,
        "attention_mask": attention_mask,
    }

    print("Compiling TensorRT Llama Vision/Mllama visual model...")
    trt_model_func = compile_vit_plugin_model(
        wrapper,
        (),
        DEVICE,
        example_kwargs=trt_kwargs,
        dynamic_shapes={name: {} for name in trt_kwargs},
    )

    verify_mllama_output(
        trt_model_func,
        visual_pytorch_wrapper,
        pyt_inputs,
        trt_inputs,
    )

    print("\n=== Starting Llama 3.2 Vision/Mllama Visual Benchmark ===")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    trt_mllama_forward = lambda pixel_values, aspect_ratio_ids, attention_mask: (
        call_mllama_visual(
            trt_model_func,
            pixel_values,
            aspect_ratio_ids,
            attention_mask,
        )
    )
    benchmark_visual(trt_mllama_forward, trt_inputs, run_name="TensorRT")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        choices=("qwen_vl", "mllama", "both"),
        default="both",
        help="Which real visual tower to compile and benchmark.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Override the default checkpoint for a single selected model type. "
            "Ignored when --model-type both is used."
        ),
    )
    args = parser.parse_args()

    if args.model_type == "qwen_vl":
        run_qwen(args.model_name or QWEN_MODEL_NAME)
    elif args.model_type == "mllama":
        run_mllama(args.model_name or MLLAMA_MODEL_NAME)
    else:
        if args.model_name is not None:
            print("--model-name is ignored when --model-type both is used.")
        run_qwen(QWEN_MODEL_NAME)
        run_mllama(MLLAMA_MODEL_NAME)
