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

import os
import sys

import torch
from transformers import AutoConfig, MllamaVisionModel, Qwen2_5_VLForConditionalGeneration

# Add tools/llm to path for shared plugin utilities, matching the LLM example style.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../tools/llm"))

from plugin_utils_vit import (
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
# Qwen2.5-VL visual input is already a patch-vector tensor: [t*h*w, patch_dim].
IMAGE_GRID_THW = (1, 8, 16)

# Load the plugin and register the op/converter path.
load_plugin()
register_vit_plugin_op()

# Global config for compatibility with converter-style imports.
TARGET_CONFIG = None

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def get_vision_config(config):
    """Return the vision config from a top-level or vision-only config."""
    if hasattr(config, "vision_config"):
        return config.vision_config
    if hasattr(config, "visual"):
        return config.visual
    if hasattr(config, "attention_heads") and hasattr(config, "max_num_tiles"):
        return config
    raise ValueError("Cannot find vision config")

def get_visual_num_patches(vision_config, image_grid_thw=None):
    """Return the visual-token extent used by the ViT plugin config."""
    if image_grid_thw is not None:
        grid_t, grid_h, grid_w = image_grid_thw
        return grid_t * grid_h * grid_w

    image_size = vision_config.image_size
    patch_size = vision_config.patch_size
    if isinstance(image_size, (tuple, list)):
        image_h, image_w = image_size
    else:
        image_h = image_w = image_size
    if isinstance(patch_size, (tuple, list)):
        patch_h, patch_w = patch_size
    else:
        patch_h = patch_w = patch_size

    num_patches = (image_h // patch_h) * (image_w // patch_w) + 1
    if hasattr(vision_config, "max_num_tiles"):
        target_length = num_patches + (8 - (num_patches % 8)) % 8
        return vision_config.max_num_tiles * target_length
    return num_patches

def set_plugin_config_from_vision_config(vision_config, num_patches):
    """Set ViT plugin fields from a generic vision config."""
    num_heads = (
        getattr(vision_config, "num_heads", None)
        or getattr(vision_config, "num_attention_heads", None)
        or getattr(vision_config, "attention_heads", None)
    )
    if num_heads is None:
        raise ValueError("Cannot infer number of attention heads from vision config")

    head_dim = getattr(vision_config, "head_dim", None)
    if head_dim is None:
        head_dim = vision_config.hidden_size // num_heads

    set_vit_plugin_config(
        num_attention_heads=num_heads,
        head_dim=head_dim,
        num_patches=num_patches,
    )

def create_windowed_rope_metadata(visual_model, pixel_values, image_grid_thw):
    """
    Lower windowed visual-attention metadata to raw tensors.

    Qwen-VL derives RoPE positions and window boundaries from image_grid_thw
    using Python list/index logic that is awkward for torch.export. We compute
    it once here and pass the compiled wrapper only tensor inputs.
    """
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
    for start, end in zip(cu_window_seqlens[:-1], cu_window_seqlens[1:]):
        window_attention_mask[:, start:end, start:end] = 0

    return {
        "rotary_pos_emb": rotary_pos_emb.to(device=DEVICE),
        "attention_mask": attention_mask,
        "window_attention_mask": window_attention_mask,
        "window_index": window_index,
        "reverse_window_index": reverse_window_index,
    }

def create_patch_vector_inputs(vision_config):
    """
    Create native PyTorch args for flattened patch-vector visual input.

    Input style:
    patch_vector_inputs -> [num_patches, patch_vector_dim]

    These are the public inputs for models like Qwen-VL:
    visual(pixel_values, image_grid_thw).
    """
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


def create_tiled_vision_inputs(vision_config):
    """
    Create native PyTorch args for a tiled visual input.

    Input style:
    tiled_image_inputs -> [B, images, tiles, C, H, W]

    These are the public HuggingFace Mllama/Llama Vision visual inputs.
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

def create_tiled_aspect_ratio_attention_mask(vision_config, aspect_ratio_mask):
    """
    Create an expanded additive mask from tiled aspect-ratio validity metadata.

    This lowers Mllama's compact tile-validity mask into the raw attention mask
    tensor consumed by the export-friendly plugin wrapper.

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

def create_visual_plugin_metadata(model_type, visual_model, vision_config, pytorch_args):
    """
    Create raw plugin-only metadata tensors for a visual model.

    The public PyTorch visual APIs are intentionally model-specific. This
    function lowers those public inputs into the tensor-only metadata expected
    by ViTPluginWrapper.
    """
    if model_type == "qwen_vl":
        pixel_values, image_grid_thw = pytorch_args
        return create_windowed_rope_metadata(
            visual_model,
            pixel_values,
            image_grid_thw,
        )
    if model_type == "mllama":
        _, _, aspect_ratio_mask = pytorch_args
        return {
            "attention_mask": create_tiled_aspect_ratio_attention_mask(
                vision_config,
                aspect_ratio_mask,
            )
        }
    raise ValueError(f"Unsupported visual model type: {model_type}")

def create_visual_inputs(model_type, visual_model, vision_config):
    """
    Create native PyTorch args and raw plugin kwargs for a visual model.

    Input styles this example currently covers:
    - raw_image_inputs    -> [B, C, H, W]
    - patch_vector_inputs -> [num_patches, patch_vector_dim]
    - tiled_image_inputs  -> [B, images, tiles, C, H, W]

    Returns:
    - pytorch_args: inputs for the original HuggingFace visual tower
    - plugin_kwargs: lowered raw tensors for the compiled plugin wrapper
    """
    if model_type == "qwen_vl":
        pytorch_args = create_patch_vector_inputs(vision_config)
    elif model_type == "mllama":
        pytorch_args = create_tiled_vision_inputs(vision_config)
    else:
        raise ValueError(f"Unsupported visual model type: {model_type}")

    plugin_kwargs = {
        "pixel_values": pytorch_args[0],
        **create_visual_plugin_metadata(
            model_type,
            visual_model,
            vision_config,
            pytorch_args,
        ),
    }
    if model_type == "mllama":
        plugin_kwargs["aspect_ratio_ids"] = pytorch_args[1]
    return pytorch_args, plugin_kwargs

def get_last_hidden_state(output):
    """Normalize HF model outputs to a tensor for verification and benchmark."""
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)):
        return output[0]
    return output

def benchmark_visual(visual_model, input_kwargs, run_name="TensorRT"):
    """Benchmark the compiled visual model."""
    pixel_values = input_kwargs["pixel_values"]
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
            return get_last_hidden_state(visual_model(**input_kwargs))

    mean_ms, std_ms, median_ms = measure_vit_latency(forward)
    print(
        f"{run_name} | {input_desc} | "
        f"Mean: {mean_ms:.3f} ms | Median: {median_ms:.3f} ms | Std: {std_ms:.3f} ms"
    )
    return mean_ms


def verify_visual_output(model_name, pytorch_model, pytorch_args, trt_model, trt_kwargs):
    """Compare PyTorch and TensorRT-plugin visual outputs."""
    print(f"\n=== Verifying {model_name} Visual Output ===")

    with torch.no_grad():
        pyt_output = get_last_hidden_state(pytorch_model(*pytorch_args))
        trt_output = get_last_hidden_state(trt_model(**trt_kwargs))

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


def run_qwen(model_name):
    print(f"Loading {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    vision_config = get_vision_config(config)

    globals()["TARGET_CONFIG"] = vision_config
    set_plugin_config_from_vision_config(
        vision_config,
        get_visual_num_patches(vision_config, IMAGE_GRID_THW),
    )
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

    pytorch_args, plugin_kwargs = create_visual_inputs(
        "qwen_vl",
        model_trt.visual,
        vision_config,
    )

    print("Compiling TensorRT visual model...")
    trt_visual_model = compile_vit_plugin_model(
        wrapper,
        (),
        DEVICE,
        example_kwargs=plugin_kwargs,
        dynamic_shapes={name: {} for name in plugin_kwargs},
    )

    verify_visual_output(
        "Qwen2.5-VL",
        model_pytorch.visual,
        pytorch_args,
        trt_visual_model,
        plugin_kwargs,
    )

    print("\n=== Starting Qwen2.5-VL Visual Benchmark ===")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    benchmark_visual(trt_visual_model, plugin_kwargs, run_name="TensorRT")


def run_mllama(model_name):
    print(f"Loading {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    vision_config = get_vision_config(config)

    globals()["TARGET_CONFIG"] = vision_config
    set_plugin_config_from_vision_config(
        vision_config,
        get_visual_num_patches(vision_config),
    )
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

    pytorch_args, plugin_kwargs = create_visual_inputs(
        "mllama",
        visual_trt,
        vision_config
    )

    print("Compiling TensorRT Llama Vision/Mllama visual model...")
    trt_visual_model = compile_vit_plugin_model(
        wrapper,
        (),
        DEVICE,
        example_kwargs=plugin_kwargs,
        dynamic_shapes={name: {} for name in plugin_kwargs},
    )

    verify_visual_output(
        "Llama 3.2 Vision/Mllama",
        visual_pytorch,
        pytorch_args,
        trt_visual_model,
        plugin_kwargs,
    )

    print("Starting Llama 3.2 Vision/Mllama Visual Benchmark...")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    benchmark_visual(trt_visual_model, plugin_kwargs, run_name="TensorRT")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    run_qwen(QWEN_MODEL_NAME)
    run_mllama(MLLAMA_MODEL_NAME)
