"""
.. _run_vlm:

Benchmarking VLM Inference with Torch-TensorRT
==========================================================

This script provides a framework for benchmarking the performance of Visual-Language
Models (VLMs). It optimizes the two most computationally intensive components of a
VLM—the language model and the vision model (image feature extraction)—using
the Torch-TensorRT dynamo backend.

Key Features:
- **Component-wise Optimization**: Compiles both the language and vision models
  separately with Torch-TensorRT to accelerate inference.
- **Performance Benchmarking**: Runs the model for multiple iterations to
  measure and compare inference latency against the PyTorch baseline.
- **Output Verification**: Checks for token-level consistency between the optimized
  TensorRT model and the original PyTorch model to ensure correctness.
- **KV Cache Testing**: Includes options to test inference with and without
  KV caching to evaluate its impact on performance.

This tool mirrors the style and structure of `run_llm.py`, providing a clear
workflow for VLM optimization and analysis.

Dependencies:
- For Qwen VLM models: pip install qwen-vl-utils
- For Eagle2 models: pip install flash-attn --no-build-isolation -v
"""

import argparse
import copy
import os
from types import SimpleNamespace
from contextlib import nullcontext
from typing import Any, Dict, Tuple, TypedDict

import requests
import torch
import torch_tensorrt
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor, PreTrainedModel
from transformers.models.qwen2 import modeling_qwen2 as mq
from transformers.models.siglip import modeling_siglip as ms
from utils import (
    export_llm,
    generate_mm,
    generate_mm_qwen2_5_vl,
    generate_mm_qwen2_5_vl_with_static_cache,
    generate_mm_with_static_cache,
    get_qwen_image_embeds,
    get_qwen_position_ids,
    record_stats,
)

# Import ViT plugin utilities (optional)
try:
    from plugin_utils_vit import (
        ViTPluginAttention,
        ViTPluginWrapper,
        VIT_INPUT_CONTRACT_NATIVE,
        VIT_INPUT_CONTRACT_TILED_ASPECT_RATIO,
        VIT_INPUT_CONTRACT_WINDOWED_ROPE,
        compile_vit_plugin_model,
        count_vit_plugin_attention_modules,
        get_vit_plugin_conversion_count,
        get_vit_plugin_config,
        load_plugin as load_vit_plugin,
        register_vit_plugin_op,
        replace_vit_attention_with_plugin,
        reset_vit_plugin_conversion_count,
        set_vit_plugin_config,
    )

    VIT_PLUGIN_AVAILABLE = True
except ImportError:
    VIT_PLUGIN_AVAILABLE = False

# --- WORKAROUND FOR EAGLE2 SDPA COMPILATION ---
# Eagle2's language model (Qwen2) implicitly defaults to "flash_attention_2"
# due to settings in its remote code and config.json. This prevents direct
# compilation with SDPA. To work around this without modifying the library,
ms.ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = ms.ALL_ATTENTION_FUNCTIONS["sdpa"]
mq.ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = mq.ALL_ATTENTION_FUNCTIONS["sdpa"]
# --- END WORKAROUND ---

# --- Model-specific constants for benchmark and compilation ---
# Centralizing these values improves readability and maintainability.
MODEL_CONSTANTS = {
    "nvidia/Eagle2-2B": {
        "EXAMPLE_SEQLEN": 2560,  # A fixed sequence length for creating the example tensor for TRT compilation.
        "IMAGE_TOKENS": 1792,  # Number of special tokens used to represent the image patch embeddings in the input sequence for Eagle2-2B VLM.
        "PROMPT_WRAPPER_TOKENS": 26,  # The number of special/processing tokens added by the processor's chat template in benchmark mode.
    },
    "Qwen/Qwen2.5-VL-3B-Instruct": {
        "EXAMPLE_SEQLEN": 2560,
        "IMAGE_TOKENS": 1426,
        "PROMPT_WRAPPER_TOKENS": 21,
    },
}
# --- END Model-specific constants ---

# -----------------------------------------------------------------------------#
# Model loading helpers
# -----------------------------------------------------------------------------#


def _is_qwen2_5_vl(model_name: str) -> bool:
    return "qwen2.5-vl" in model_name.lower()


def _is_eagle2(model_name: str) -> bool:
    return "eagle2" in model_name.lower()


def _patch_transformers_tied_weights_compat() -> None:
    """
    Some remote VLM classes still expose the older `_tied_weights_keys`
    metadata, while newer Transformers loading code expects
    `all_tied_weights_keys`. Add a conservative compatibility property on the
    base class before model construction.
    """
    existing_attr = getattr(PreTrainedModel, "all_tied_weights_keys", None)
    if isinstance(existing_attr, property) and existing_attr.fset is not None:
        return
    if existing_attr is not None and not isinstance(existing_attr, property):
        return

    def normalize_tied_weights_keys(value):
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        return {key: None for key in value}

    def all_tied_weights_keys(self):
        if "_all_tied_weights_keys_compat" in self.__dict__:
            return normalize_tied_weights_keys(
                self.__dict__["_all_tied_weights_keys_compat"]
            )
        return normalize_tied_weights_keys(getattr(self, "_tied_weights_keys", None))

    def set_all_tied_weights_keys(self, value):
        self.__dict__["_all_tied_weights_keys_compat"] = value

    PreTrainedModel.all_tied_weights_keys = property(
        all_tied_weights_keys, set_all_tied_weights_keys
    )


def _patch_transformers_image_utils_compat() -> None:
    """
    TODO: Eagle2 remote processor code imports image/video helper APIs that are
    not exported by the Transformers builds used in this environment. Remove
    this once Eagle pins/updates its remote code or we pin a known-compatible
    Transformers version. If this remains necessary, raise an upstream bug.
    """
    import transformers.image_utils as image_utils

    if not hasattr(image_utils, "VideoInput"):
        image_utils.VideoInput = Any
    if not hasattr(image_utils, "make_batched_videos"):
        def make_batched_videos(videos):
            if videos is None:
                return None
            if isinstance(videos, (list, tuple)):
                return list(videos)
            return [videos]

        image_utils.make_batched_videos = make_batched_videos


def _patch_transformers_fast_image_processor_compat() -> None:
    """
    TODO: Eagle2's fast image processor remote code imports docstring constants
    from Transformers that are not exported by the builds used here. Remove
    this after the Eagle remote-code / Transformers API mismatch is resolved.
    """
    import transformers.image_processing_utils_fast as fast_image_utils

    if not hasattr(fast_image_utils, "BASE_IMAGE_PROCESSOR_FAST_DOCSTRING"):
        fast_image_utils.BASE_IMAGE_PROCESSOR_FAST_DOCSTRING = ""
    if not hasattr(fast_image_utils, "BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS"):
        fast_image_utils.BASE_IMAGE_PROCESSOR_FAST_DOCSTRING_PREPROCESS = ""
    if not hasattr(fast_image_utils, "DefaultFastImageProcessorKwargs"):
        class DefaultFastImageProcessorKwargs(TypedDict, total=False):
            pass

        fast_image_utils.DefaultFastImageProcessorKwargs = DefaultFastImageProcessorKwargs


def _patch_siglip_flash_attention_compat() -> None:
    """
    Eagle2's remote config can still request FlashAttention2 for the SigLIP
    vision tower even after the parent config is patched. The SigLIP attention
    registry above maps flash_attention_2 to SDPA, so bypass only the package
    availability check that runs during model construction.
    """

    def flash_attention_check_compat(self, *args, **kwargs):
        config = getattr(self, "config", None)
        if config is not None:
            _set_config_attn_implementation(config, "sdpa")
        return True

    def get_correct_attn_implementation_compat(self, *args, **kwargs):
        config = getattr(self, "config", None)
        if config is not None:
            _set_config_attn_implementation(config, "sdpa")
        return "sdpa"

    for class_name in ("SiglipPreTrainedModel", "SiglipVisionModel"):
        cls = getattr(ms, class_name, None)
        if cls is None:
            continue
        if hasattr(cls, "get_correct_attn_implementation"):
            setattr(
                cls,
                "get_correct_attn_implementation",
                get_correct_attn_implementation_compat,
            )
        for method_name in ("_flash_attn_2_can_dispatch", "_flash_attn_can_dispatch"):
            if hasattr(cls, method_name):
                setattr(cls, method_name, flash_attention_check_compat)


def _model_loader_candidates():
    candidates = []
    for class_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        try:
            module = __import__("transformers", fromlist=[class_name])
            candidates.append(getattr(module, class_name))
        except (ImportError, AttributeError):
            pass
    candidates.append(AutoModel)
    return candidates


def _effective_attn_implementation(args: argparse.Namespace):
    if args.attn_implementation:
        return args.attn_implementation
    if args.vision_backend == "torchtrt" or _is_eagle2(args.model):
        return "sdpa"
    return None


def _set_config_attn_implementation(config, attn_implementation: str) -> None:
    visited = set()
    attn_attr_names = {
        "attn_implementation",
        "_attn_implementation",
        "_attn_implementation_internal",
    }

    def is_attn_attr(name: str) -> bool:
        return name in attn_attr_names or name.endswith("_attn_implementation")

    def set_attn_attr(config_obj, attr_name: str) -> None:
        try:
            setattr(config_obj, attr_name, attn_implementation)
        except Exception:
            pass
        try:
            config_obj.__dict__[attr_name] = attn_implementation
        except Exception:
            pass

    def visit(config_obj):
        if config_obj is None or id(config_obj) in visited:
            return
        visited.add(id(config_obj))

        if isinstance(config_obj, dict):
            for key, value in list(config_obj.items()):
                if is_attn_attr(str(key)):
                    config_obj[key] = attn_implementation
                else:
                    visit(value)
            return

        if isinstance(config_obj, (list, tuple)):
            for value in config_obj:
                visit(value)
            return

        if isinstance(config_obj, (str, bytes, int, float, bool)):
            return

        for attr_name in attn_attr_names:
            set_attn_attr(config_obj, attr_name)

        config_dict = getattr(config_obj, "__dict__", None)
        if not isinstance(config_dict, dict):
            return

        for child_name, child_value in list(config_dict.items()):
            if is_attn_attr(child_name):
                set_attn_attr(config_obj, child_name)
            else:
                visit(child_value)

    visit(config)


def _load_model_config(args: argparse.Namespace):
    config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    attn_implementation = _effective_attn_implementation(args)
    if attn_implementation:
        _set_config_attn_implementation(config, attn_implementation)
    return config


def _from_pretrained_kwargs(args: argparse.Namespace, torch_dtype: torch.dtype):
    kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": args.trust_remote_code,
        "config": _load_model_config(args),
    }
    attn_implementation = _effective_attn_implementation(args)
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    return kwargs


_VISION_MODULE_ATTRS = ("vision_model", "visual", "vision_tower", "vision_encoder")


def _is_windowed_rope_vision_module(module: torch.nn.Module) -> bool:
    return (
        hasattr(module, "patch_embed")
        and hasattr(module, "blocks")
        and hasattr(module, "rot_pos_emb")
        and hasattr(module, "get_window_index")
    )


def _is_merged_windowed_rope_vision_module(module: torch.nn.Module) -> bool:
    return _is_windowed_rope_vision_module(module) and hasattr(module, "merger")


def _is_tiled_aspect_ratio_vision_module(module: torch.nn.Module) -> bool:
    return (
        hasattr(module, "patch_embedding")
        and hasattr(module, "global_transformer")
        and hasattr(module, "pre_tile_positional_embedding")
    )


def _is_native_vit_vision_module(module: torch.nn.Module) -> bool:
    has_patch_embedding = any(
        hasattr(module, attr_name)
        for attr_name in ("patch_embed", "patch_embedding", "embeddings")
    )
    has_transformer = any(
        hasattr(module, attr_name)
        for attr_name in ("blocks", "encoder", "transformer", "global_transformer")
    )
    return has_patch_embedding and has_transformer


def _is_vision_module(module: torch.nn.Module) -> bool:
    return (
        _is_windowed_rope_vision_module(module)
        or _is_tiled_aspect_ratio_vision_module(module)
        or _is_native_vit_vision_module(module)
    )


def _contains_vision_module(module: torch.nn.Module) -> bool:
    for attr_name in _VISION_MODULE_ATTRS:
        if isinstance(getattr(module, attr_name, None), torch.nn.Module):
            return True

    for _, child in module.named_modules():
        if child is module:
            continue
        if _is_vision_module(child):
            return True
    return False


_LANGUAGE_MODULE_ATTRS = (
    "language_model",
    "text_model",
    "llm",
    "decoder",
    "model",
)


def _find_language_module(model: torch.nn.Module) -> Tuple[str, torch.nn.Module]:
    for attr_name in _LANGUAGE_MODULE_ATTRS:
        candidate = getattr(model, attr_name, None)
        if not isinstance(candidate, torch.nn.Module):
            continue
        if attr_name == "model" and _contains_vision_module(candidate):
            continue
        return attr_name, candidate

    for module_name, module in model.named_modules():
        if module is model:
            continue
        if module_name.rsplit(".", 1)[-1] not in _LANGUAGE_MODULE_ATTRS:
            continue
        if _contains_vision_module(module):
            continue
        return module_name, module

    raise ValueError(
        "Cannot find a language-model submodule. Expected a language/text "
        "model leaf module that does not also contain the vision tower."
    )


def get_language_model(model: torch.nn.Module) -> torch.nn.Module:
    _, language_model = _find_language_module(model)
    return language_model


def set_language_model(model: torch.nn.Module, language_model: torch.nn.Module) -> None:
    module_name, _ = _find_language_module(model)
    _set_module_by_name(model, module_name, language_model)


def _find_vision_module(model: torch.nn.Module) -> Tuple[str, torch.nn.Module]:
    visual = getattr(model, "visual", None)
    if isinstance(visual, torch.nn.Module):
        return "visual", visual

    for parent_attr in ("model", "language_model"):
        parent = getattr(model, parent_attr, None)
        if not isinstance(parent, torch.nn.Module):
            continue
        visual = getattr(parent, "visual", None)
        if isinstance(visual, torch.nn.Module):
            return f"{parent_attr}.visual", visual

    for module_name, module in model.named_modules():
        if module is model:
            continue
        if _is_merged_windowed_rope_vision_module(module):
            return module_name, module

    for attr_name in _VISION_MODULE_ATTRS:
        if attr_name == "visual":
            continue
        candidate = getattr(model, attr_name, None)
        if isinstance(candidate, torch.nn.Module):
            return attr_name, candidate

    for module_name, module in model.named_modules():
        if module is model:
            continue
        if _is_vision_module(module):
            return module_name, module

    raise ValueError(
        "Cannot find a vision-model submodule. Expected a Hugging Face vision "
        "tower alias or a module with ViT-like patch embedding and transformer "
        "blocks/encoder."
    )


def _set_child_module(parent: torch.nn.Module, child_name: str, child: torch.nn.Module):
    if child_name.isdigit() and isinstance(
        parent, (torch.nn.ModuleList, torch.nn.Sequential)
    ):
        parent[int(child_name)] = child
    else:
        setattr(parent, child_name, child)


def _set_module_by_name(
    model: torch.nn.Module, module_name: str, module: torch.nn.Module
) -> None:
    if not module_name:
        raise ValueError("Cannot replace the root model with a vision module.")

    path = module_name.split(".")
    parent = model
    for child_name in path[:-1]:
        parent = getattr(parent, child_name)
    _set_child_module(parent, path[-1], module)


def get_vision_model(model: torch.nn.Module) -> torch.nn.Module:
    _, vision_model = _find_vision_module(model)
    return vision_model


def set_vision_model(model: torch.nn.Module, vision_model: torch.nn.Module) -> None:
    module_name, _ = _find_vision_module(model)
    _set_module_by_name(model, module_name, vision_model)


def get_input_embedding_layer(
    model: torch.nn.Module, torch_dtype: torch.dtype, device: torch.device
) -> torch.nn.Embedding:
    language_model = get_language_model(model)
    if hasattr(language_model, "get_input_embeddings"):
        return language_model.get_input_embeddings().to(torch_dtype).to(device)
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings().to(torch_dtype).to(device)
    raise ValueError("Cannot find an input embedding layer for this VLM.")


def get_model(
    args: argparse.Namespace, device: torch.device, torch_dtype: torch.dtype
) -> Tuple[torch.nn.Module, AutoProcessor, torch.nn.Embedding]:
    """Load and configure a VLM model, processor, and input embedding layer."""
    _patch_transformers_tied_weights_compat()
    if _is_eagle2(args.model):
        _patch_siglip_flash_attention_compat()
    model_kwargs = _from_pretrained_kwargs(args, torch_dtype)
    last_error = None

    with torch.no_grad():
        for loader in _model_loader_candidates():
            try:
                model = (
                    loader.from_pretrained(args.model, **model_kwargs).eval().to(device)
                )
                break
            except (KeyError, ValueError) as exc:
                last_error = exc
        else:
            raise ValueError(
                f"Could not load '{args.model}' with available AutoModel classes."
            ) from last_error

    processor_name = args.processor or args.model
    if _is_eagle2(args.model):
        _patch_transformers_image_utils_compat()
        _patch_transformers_fast_image_processor_compat()
    processor_use_fast = True
    processor = AutoProcessor.from_pretrained(
        processor_name,
        trust_remote_code=args.trust_remote_code,
        use_fast=processor_use_fast,
    )
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    emb_layer = get_input_embedding_layer(model, torch_dtype, device)
    return model, processor, emb_layer

# -----------------------------------------------------------------------------#
# Input loading helpers
# -----------------------------------------------------------------------------#

def _get_message_content_value(content_item, keys):
    for key in keys:
        if key in content_item and content_item[key] is not None:
            return content_item[key]
    return None

def extract_vision_inputs(processor, messages):
    """
    Extract image/video payloads from chat-style messages for VLM processors.

    Some processors own this logic directly. For processors that do not, fall
    back to the common chat content schema used by Hugging Face VLM examples:
    {"type": "image", "image": ...} and {"type": "video", "video": ...}.
    """
    if hasattr(processor, "process_vision_info"):
        return processor.process_vision_info(messages)

    image_inputs = []
    video_inputs = []
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, dict):
            content = [content]
        for content_item in content:
            if not isinstance(content_item, dict):
                continue

            content_type = content_item.get("type")
            if content_type == "image":
                image = _get_message_content_value(
                    content_item, ("image", "url", "path")
                )
                if image is not None:
                    image_inputs.append(image)
            elif content_type == "video":
                video = _get_message_content_value(
                    content_item, ("video", "url", "path")
                )
                if video is not None:
                    video_inputs.append(video)

    return image_inputs or None, video_inputs or None

def load_inputs(args: argparse.Namespace, processor, device: torch.device):
    """
    Loads and constructs the input dictionary for the specified VLM model.
    """
    # Load image from local path if provided, otherwise use default URL
    if args.image_path is not None:
        # Use local image file
        image = Image.open(args.image_path)
    else:
        # Use default URL image
        url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

    if args.benchmark:
        model_constants = MODEL_CONSTANTS.get(args.model, {})
        image_tokens = model_constants.get("IMAGE_TOKENS", 0)
        wrapper_tokens = model_constants.get("PROMPT_WRAPPER_TOKENS", 0)

        prompt_len = args.isl - image_tokens - wrapper_tokens
        prompt_txt = " ".join(["token"] * max(prompt_len, 0))
    else:
        prompt_txt = args.prompt or "Describe this image."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_txt},
            ],
        }
    ]

    text = [
        processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    ]

    image_inputs, video_inputs = extract_vision_inputs(processor, messages)

    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    return inputs


# -----------------------------------------------------------------------------#
# Torch-TensorRT compilation helpers
# -----------------------------------------------------------------------------#


class _LMNoCache(torch.nn.Module):
    """
    Thin wrapper that exposes a language model via ``inputs_embeds`` without KV-cache.
    """

    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, inputs_embeds, position_ids):
        out = self.lm(inputs_embeds=inputs_embeds, position_ids=position_ids)
        return (
            out.logits
            if hasattr(out, "logits")
            else out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        )


def _compile_lm(
    language_model: torch.nn.Module,
    input_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.nn.Module:
    """
    Compile the language model component of a VLM with Torch-TensorRT
    """
    lm_wrap = _LMNoCache(language_model).to(device).eval()
    max_seq_len = input_embeds.shape[1] + args.num_tokens

    seq_len = torch.export.Dim("seq", min=1, max=max_seq_len)
    use_fp32_acc = False
    if args.precision == "FP16":
        use_fp32_acc = True

    exported_program = export_llm(
        lm_wrap,
        input_embeds,
        min_seq_len=1,
        max_seq_len=2560,
        position_ids=position_ids,
    )

    with torch_tensorrt.dynamo.Debugger() if args.debug else nullcontext():
        trt_mod = torch_tensorrt.dynamo.compile(
            exported_program,
            inputs=[input_embeds, position_ids],
            use_fp32_acc=use_fp32_acc,
            device=device,
            disable_tf32=args.disable_tf32,
            use_python_runtime=args.use_python_runtime,
            offload_module_to_cpu=args.offload_module_to_cpu,
            min_block_size=args.min_block_size,
        )
    return trt_mod


def _make_lm_compile_position_ids(
    args: argparse.Namespace,
    input_embeds: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    base_position_ids = torch.arange(
        input_embeds.shape[1], dtype=torch.long, device=device
    ).unsqueeze(0)
    base_position_ids = base_position_ids.expand(input_embeds.shape[0], -1)
    if _is_qwen2_5_vl(args.model):
        return base_position_ids.unsqueeze(0).expand(3, -1, -1).contiguous()
    return base_position_ids


def compile_lm_torchtrt(
    model: torch.nn.Module, args: argparse.Namespace, device: torch.device
) -> torch.nn.Module:
    """
    Compiles the Language Model (LLM) component of the VLM using Torch-TensorRT.
    """
    torch_dtype = {
        "FP16": torch.float16,
        "BF16": torch.bfloat16,
    }.get(args.precision, torch.float32)

    lm_model = get_language_model(model)

    model_constants = MODEL_CONSTANTS.get(
        args.model, {"EXAMPLE_SEQLEN": args.isl}
    )
    example_seq_len = model_constants["EXAMPLE_SEQLEN"]

    example_embeds = torch.randn(
        args.batch_size,
        example_seq_len,
        _get_lm_hidden_size(lm_model.config),
        dtype=torch_dtype,
        device=device,
    )

    position_ids = _make_lm_compile_position_ids(args, example_embeds, device)

    return _compile_lm(lm_model, example_embeds, position_ids, args, device)


def _compile_vision_model(
    vision_model: torch.nn.Module,
    example_pixel_values: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.nn.Module:
    """
    Compile a vision tower with Torch-TensorRT.
    """
    use_fp32_acc = False
    if args.precision == "FP16":
        use_fp32_acc = True

    with torch.inference_mode():
        exported_program = torch.export.export(
            vision_model,
            (example_pixel_values,),
            strict=False,
        )

    with torch_tensorrt.dynamo.Debugger() if args.debug else nullcontext():
        trt_mod = torch_tensorrt.dynamo.compile(
            exported_program,
            inputs=[example_pixel_values],
            use_fp32_acc=use_fp32_acc,
            device=device,
            disable_tf32=args.disable_tf32,
            use_python_runtime=args.use_python_runtime,
            offload_module_to_cpu=args.offload_module_to_cpu,
            min_block_size=args.min_block_size,
        )
    return trt_mod


def _get_vision_config(config):
    if hasattr(config, "vision_config"):
        return config.vision_config
    if hasattr(config, "visual"):
        return config.visual
    return config


def _get_config_attr(config, names):
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return None


def _get_lm_hidden_size(config):
    hidden_size = _get_config_attr(config, ("hidden_size", "n_embd", "d_model"))
    if hidden_size is not None:
        return int(hidden_size)

    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        hidden_size = _get_config_attr(
            text_config, ("hidden_size", "n_embd", "d_model")
        )
        if hidden_size is not None:
            return int(hidden_size)

    raise ValueError(
        "Cannot infer language-model hidden size from config. Expected one of "
        "config.hidden_size, config.n_embd, config.d_model, or the same fields "
        "under config.text_config."
    )


def _infer_patch_count(vision_config, pixel_values: torch.Tensor) -> int:
    if pixel_values.dim() in (2, 3):
        return int(pixel_values.shape[-2])

    image_size = _get_config_attr(vision_config, ("image_size",))
    patch_size = _get_config_attr(vision_config, ("patch_size",))
    if image_size is None or patch_size is None:
        return 0

    if isinstance(image_size, (tuple, list)):
        image_h, image_w = image_size[:2]
    else:
        image_h = image_w = image_size
    if isinstance(patch_size, (tuple, list)):
        patch_h, patch_w = patch_size[:2]
    else:
        patch_h = patch_w = patch_size
    return int((image_h // patch_h) * (image_w // patch_w) + 1)


def _set_vit_plugin_config_from_vision(vision_config, pixel_values):
    num_heads = _get_config_attr(
        vision_config, ("num_heads", "num_attention_heads", "attention_heads")
    )
    if num_heads is None:
        raise ValueError("Cannot infer ViT plugin num_attention_heads from config.")

    head_dim = _get_config_attr(vision_config, ("head_dim",))
    if head_dim is None:
        hidden_size = _get_config_attr(
            vision_config, ("hidden_size", "embed_dim", "dim")
        )
        if hidden_size is None:
            raise ValueError("Cannot infer ViT plugin hidden_size from config.")
        head_dim = int(hidden_size) // int(num_heads)

    set_vit_plugin_config(
        num_attention_heads=int(num_heads),
        head_dim=int(head_dim),
        num_patches=_infer_patch_count(vision_config, pixel_values),
    )


def _create_windowed_rope_vit_plugin_core_inputs(
    visual_model: torch.nn.Module,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
):
    with torch.no_grad():
        rotary_pos_emb = visual_model.rot_pos_emb(image_grid_thw)
        window_index, cu_window_seqlens = visual_model.get_window_index(image_grid_thw)

    window_index = window_index.to(device=device, dtype=torch.long)
    reverse_window_index = torch.argsort(window_index)

    seq_len = pixel_values.shape[0]
    attention_mask = torch.zeros(1, seq_len, seq_len, dtype=dtype, device=device)
    window_attention_mask = torch.full(
        (1, seq_len, seq_len),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )

    cu_window_seqlens = torch.as_tensor(
        cu_window_seqlens, device="cpu", dtype=torch.long
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens).tolist()
    max_window_seq_len = max(
        end - start
        for start, end in zip(cu_window_seqlens[:-1], cu_window_seqlens[1:])
    )

    for start, end in zip(cu_window_seqlens[:-1], cu_window_seqlens[1:]):
        window_attention_mask[:, start:end, start:end] = 0

    return {
        "rotary_pos_emb": rotary_pos_emb.to(device=device),
        "attention_mask": attention_mask,
        "window_attention_mask": window_attention_mask,
        "cu_window_seqlens": torch.tensor(
            cu_window_seqlens, dtype=torch.int32, device=device
        ),
        "max_window_seq_len": max_window_seq_len,
        "window_index": window_index,
        "reverse_window_index": reverse_window_index,
    }


class _VITPluginVisualAdapter(torch.nn.Module):
    """
    Preserve a vision tower's native call signature while using the compiled
    ViT plugin wrapper internally.
    """

    def __init__(
        self,
        compiled_visual,
        input_contract: str,
        core_inputs: Dict[str, Any],
        original_visual: torch.nn.Module | None = None,
    ):
        super().__init__()
        self.compiled_visual = compiled_visual
        self.input_contract = input_contract
        self.core_inputs = core_inputs
        self.original_visual = original_visual
        self.return_pooler_output = False

        if original_visual is not None:
            for attr_name in ("dtype", "spatial_merge_size", "spatial_merge_unit"):
                if hasattr(original_visual, attr_name):
                    setattr(self, attr_name, getattr(original_visual, attr_name))
            if hasattr(original_visual, "merger"):
                self.merger = original_visual.merger
                self.return_pooler_output = True

    def _call_compiled(self, positional_args, keyword_args):
        try:
            return self.compiled_visual(**keyword_args)
        except TypeError:
            return self.compiled_visual(*positional_args)

    def _compiled_kwargs(self, pixel_values, *args, **kwargs):
        if self.input_contract == VIT_INPUT_CONTRACT_WINDOWED_ROPE:
            return {"pixel_values": pixel_values, **self.core_inputs}

        if self.input_contract == VIT_INPUT_CONTRACT_TILED_ASPECT_RATIO:
            aspect_ratio_ids = _get_runtime_tensor(
                args, kwargs, self.core_inputs, ("aspect_ratio_ids",), position=0
            )
            attention_mask = _get_runtime_tensor(
                args,
                kwargs,
                self.core_inputs,
                ("attention_mask", "aspect_ratio_mask"),
                position=1,
            )
            return {
                "pixel_values": pixel_values,
                "aspect_ratio_ids": aspect_ratio_ids,
                "attention_mask": attention_mask,
            }

        if self.input_contract == VIT_INPUT_CONTRACT_NATIVE:
            return {"pixel_values": pixel_values}

        raise ValueError(f"Unsupported ViT plugin input contract: {self.input_contract}")

    def forward(self, pixel_values, *args, **kwargs):
        keyword_args = self._compiled_kwargs(pixel_values, *args, **kwargs)
        output = self._call_compiled(tuple(keyword_args.values()), keyword_args)
        if self.return_pooler_output:
            return SimpleNamespace(pooler_output=output, last_hidden_state=output)
        return output


def _get_required_input(inputs, names, purpose: str):
    for name in names:
        value = inputs.get(name)
        if isinstance(value, torch.Tensor):
            return value
    raise ValueError(
        f"ViT plugin path requires {purpose}. Expected one of: {', '.join(names)}."
    )


def _get_optional_tensor(inputs, names):
    for name in names:
        value = inputs.get(name)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _get_runtime_tensor(args, kwargs, core_inputs, names, position=None):
    for name in names:
        value = kwargs.get(name)
        if isinstance(value, torch.Tensor):
            return value
    if (
        position is not None
        and position < len(args)
        and isinstance(args[position], torch.Tensor)
    ):
        return args[position]
    for name in names:
        value = core_inputs.get(name)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _has_windowed_rope_contract(visual_model, inputs) -> bool:
    return (
        isinstance(inputs.get("image_grid_thw"), torch.Tensor)
        and hasattr(visual_model, "get_window_index")
        and hasattr(visual_model, "rot_pos_emb")
        and hasattr(visual_model, "patch_embed")
        and hasattr(visual_model, "blocks")
    )


def _has_tiled_aspect_ratio_contract(inputs) -> bool:
    return isinstance(inputs.get("aspect_ratio_ids"), torch.Tensor) and (
        _get_optional_tensor(inputs, ("aspect_ratio_mask", "attention_mask"))
        is not None
    )


def _prepare_vit_plugin_inputs(
    visual_model,
    inputs,
    pixel_values: torch.Tensor,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> Tuple[str, Dict[str, Any], int]:
    if _has_windowed_rope_contract(visual_model, inputs):
        core_inputs = _create_windowed_rope_vit_plugin_core_inputs(
            visual_model,
            pixel_values,
            inputs["image_grid_thw"],
            device,
            torch_dtype,
        )
        max_window_seq_len = core_inputs.pop("max_window_seq_len")
        return (
            VIT_INPUT_CONTRACT_WINDOWED_ROPE,
            core_inputs,
            max_window_seq_len,
        )

    if _has_tiled_aspect_ratio_contract(inputs):
        core_inputs = {
            "aspect_ratio_ids": _get_required_input(
                inputs, ("aspect_ratio_ids",), "tiled vision aspect ratio ids"
            ),
            "attention_mask": _get_required_input(
                inputs,
                ("aspect_ratio_mask", "attention_mask"),
                "tiled vision attention mask",
            ),
        }
        return (
            VIT_INPUT_CONTRACT_TILED_ASPECT_RATIO,
            core_inputs,
            0,
        )

    return (
        VIT_INPUT_CONTRACT_NATIVE,
        {},
        0,
    )


def _compile_vision_with_vit_plugin(
    model: torch.nn.Module,
    inputs,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.nn.Module:
    torch_dtype = {
        "FP16": torch.float16,
        "BF16": torch.bfloat16,
    }.get(args.precision, torch.float32)

    vision_config = _get_vision_config(model.config)
    pixel_values = inputs["pixel_values"].to(dtype=torch_dtype)

    load_vit_plugin()
    register_vit_plugin_op()
    _set_vit_plugin_config_from_vision(vision_config, pixel_values)
    print(f"ViT plugin config: {get_vit_plugin_config()}")

    visual_model = replace_vit_attention_with_plugin(
        get_vision_model(model), vision_config
    )
    plugin_module_count = count_vit_plugin_attention_modules(visual_model)
    print(f"ViT plugin attention modules inserted: {plugin_module_count}")

    input_contract, core_inputs, max_window_seq_len = _prepare_vit_plugin_inputs(
        visual_model, inputs, pixel_values, device, torch_dtype
    )
    wrapper = ViTPluginWrapper(
        visual_model,
        input_contract=input_contract,
        max_window_seq_len=max_window_seq_len,
    ).eval()

    compile_kwargs = {"pixel_values": pixel_values, **core_inputs}
    reset_vit_plugin_conversion_count()
    compiled_visual = compile_vit_plugin_model(
        wrapper,
        (),
        device,
        example_kwargs=compile_kwargs,
        dynamic_shapes={name: {} for name in compile_kwargs},
        debug=args.debug,
    )
    plugin_conversion_count = get_vit_plugin_conversion_count()
    print(f"ViT plugin TensorRT conversions: {plugin_conversion_count}")
    if plugin_conversion_count == 0:
        raise RuntimeError(
            "ViT plugin backend was requested, but no ViTAttentionPlugin nodes "
            "were lowered into the TensorRT network."
        )

    return _VITPluginVisualAdapter(
        compiled_visual, input_contract, core_inputs, original_visual=visual_model
    ).eval()


def compile_vision_torchtrt(
    model: torch.nn.Module,
    args: argparse.Namespace,
    inputs,
    device: torch.device,
) -> torch.nn.Module:
    """
    Dispatcher function for vision model compilation.
    """
    example_pixel_values = inputs["pixel_values"]
    if getattr(args, "vision_backend", "torchtrt") == "plugin":
        if not VIT_PLUGIN_AVAILABLE:
            raise RuntimeError(
                "ViT plugin vision backend requested but plugin utilities are not available."
            )
        return _compile_vision_with_vit_plugin(model, inputs, args, device)

    if _is_qwen2_5_vl(args.model):
        # TODO: Vision model compilation for Qwen2.5-VL is currently skipped.
        # The model's `get_window_index` method uses dynamic Python list operations
        # (e.g., .tolist(), .extend()) to process variable-sized image grids for
        # windowed attention. These operations are incompatible with torch.export's
        # static graph tracing, preventing successful compilation.
        return get_vision_model(model)

    try:
        return _compile_vision_model(
            get_vision_model(model), example_pixel_values, args, device
        )
    except ValueError as exc:
        raise ValueError(
            f"Cannot compile the vision tower for '{args.model}' with the generic "
            "path. Add a model adapter or use a supported architecture."
        ) from exc


# -----------------------------------------------------------------------------#
# Utility helpers
# -----------------------------------------------------------------------------#


def print_outputs(backend_name: str, gen_tokens: torch.Tensor, tokenizer):
    """Print the generated tokens from the model."""
    print(f"========= {backend_name} =========")
    print(
        f"{backend_name} model generated text: ",
        tokenizer.decode(gen_tokens[0], skip_special_tokens=True),
    )
    print("===================================")


def _extract_hidden_states(outputs):
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def _extract_vision_output(outputs):
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "pooler_output"):
        return outputs.pooler_output
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def _module_device(module: torch.nn.Module, fallback: torch.device) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return fallback


def _call_language_model_for_verify(language_model, inputs_embeds, position_ids):
    target_device = _module_device(language_model, inputs_embeds.device)
    inputs_embeds = inputs_embeds.to(target_device)
    position_ids = position_ids.to(target_device)
    try:
        outputs = language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
        )
    except TypeError:
        outputs = language_model(inputs_embeds, position_ids)
    return _extract_hidden_states(outputs)


def _qwen_full_cu_seqlens(image_grid_thw: torch.Tensor) -> torch.Tensor:
    grid = image_grid_thw.to(device="cpu", dtype=torch.int64)
    seq_lens = torch.repeat_interleave(grid[:, 1] * grid[:, 2], grid[:, 0])
    cu_seqlens = torch.nn.functional.pad(seq_lens.cumsum(dim=0), (1, 0), value=0)
    return cu_seqlens.to(device=image_grid_thw.device, dtype=torch.int32)


def _call_qwen_visual_block_reference(
    block,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
    position_embeddings,
) -> torch.Tensor:
    call_attempts = (
        lambda: block(
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
        ),
        lambda: block(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        ),
        lambda: block(hidden_states, cu_seqlens=cu_seqlens),
        lambda: block(hidden_states, cu_seqlens),
    )
    last_error = None
    for call in call_attempts:
        try:
            return _extract_hidden_states(call())
        except TypeError as exc:
            last_error = exc
    raise last_error


def _call_qwen_attention_reference(
    attn,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor,
    position_embeddings,
) -> torch.Tensor:
    call_attempts = (
        lambda: attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        ),
        lambda: attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
        ),
        lambda: attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        ),
        lambda: attn(hidden_states, cu_seqlens=cu_seqlens),
        lambda: attn(hidden_states, cu_seqlens),
    )
    last_error = None
    for call in call_attempts:
        try:
            return _extract_hidden_states(call())
        except TypeError as exc:
            last_error = exc
    raise last_error


def _qwen_windowed_rope_attention_inputs(
    visual,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    layer_idx: int,
    dtype: torch.dtype,
):
    pixel_values = pixel_values.to(device=_module_device(visual, pixel_values.device))
    if hasattr(visual, "dtype"):
        pixel_values = pixel_values.to(dtype=visual.dtype)

    hidden_states = visual.patch_embed(pixel_values)
    rotary_pos_emb = visual.rot_pos_emb(image_grid_thw)
    window_index, cu_window_seqlens = visual.get_window_index(image_grid_thw)

    window_index = torch.as_tensor(
        window_index, device=hidden_states.device, dtype=torch.long
    )
    cu_window_seqlens = torch.as_tensor(
        cu_window_seqlens, device=hidden_states.device, dtype=torch.int32
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
    cu_seqlens = _qwen_full_cu_seqlens(image_grid_thw).to(hidden_states.device)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(
        seq_len // visual.spatial_merge_unit,
        visual.spatial_merge_unit,
        -1,
    )
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)

    rotary_pos_emb = rotary_pos_emb.to(device=hidden_states.device)
    rotary_pos_emb = rotary_pos_emb.reshape(
        seq_len // visual.spatial_merge_unit,
        visual.spatial_merge_unit,
        -1,
    )
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    attention_mask = torch.zeros(1, seq_len, seq_len, dtype=dtype, device=hidden_states.device)
    window_attention_mask = torch.full(
        (1, seq_len, seq_len),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=hidden_states.device,
    )
    for start, end in zip(cu_window_seqlens[:-1], cu_window_seqlens[1:]):
        window_attention_mask[:, start:end, start:end] = 0

    full_attention = layer_idx in visual.fullatt_block_indexes
    mask_now = attention_mask if full_attention else window_attention_mask
    cu_seqlens_now = cu_seqlens if full_attention else cu_window_seqlens
    hidden_states = visual.blocks[layer_idx].norm1(hidden_states)
    return hidden_states, mask_now, cu_seqlens_now, rotary_pos_emb, position_embeddings


def _qwen_windowed_rope_block_inputs(
    visual,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    layer_idx: int,
    dtype: torch.dtype,
):
    pixel_values = pixel_values.to(device=_module_device(visual, pixel_values.device))
    if hasattr(visual, "dtype"):
        pixel_values = pixel_values.to(dtype=visual.dtype)

    hidden_states = visual.patch_embed(pixel_values)
    rotary_pos_emb = visual.rot_pos_emb(image_grid_thw)
    window_index, cu_window_seqlens = visual.get_window_index(image_grid_thw)

    window_index = torch.as_tensor(
        window_index, device=hidden_states.device, dtype=torch.long
    )
    cu_window_seqlens = torch.as_tensor(
        cu_window_seqlens, device=hidden_states.device, dtype=torch.int32
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
    cu_seqlens = _qwen_full_cu_seqlens(image_grid_thw).to(hidden_states.device)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(
        seq_len // visual.spatial_merge_unit,
        visual.spatial_merge_unit,
        -1,
    )
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)

    rotary_pos_emb = rotary_pos_emb.to(device=hidden_states.device)
    rotary_pos_emb = rotary_pos_emb.reshape(
        seq_len // visual.spatial_merge_unit,
        visual.spatial_merge_unit,
        -1,
    )
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    attention_mask = torch.zeros(
        1, seq_len, seq_len, dtype=dtype, device=hidden_states.device
    )
    window_attention_mask = torch.full(
        (1, seq_len, seq_len),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=hidden_states.device,
    )
    for start, end in zip(cu_window_seqlens[:-1], cu_window_seqlens[1:]):
        window_attention_mask[:, start:end, start:end] = 0

    for prior_layer_idx in range(layer_idx):
        full_attention = prior_layer_idx in visual.fullatt_block_indexes
        cu_seqlens_now = cu_seqlens if full_attention else cu_window_seqlens
        hidden_states = _call_qwen_visual_block_reference(
            visual.blocks[prior_layer_idx],
            hidden_states,
            cu_seqlens_now,
            rotary_pos_emb,
            position_embeddings,
        )

    full_attention = layer_idx in visual.fullatt_block_indexes
    mask_now = attention_mask if full_attention else window_attention_mask
    cu_seqlens_now = cu_seqlens if full_attention else cu_window_seqlens
    return hidden_states, mask_now, cu_seqlens_now, rotary_pos_emb, position_embeddings


def _call_qwen_block_with_attention_wrapper(
    block,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    position_embeddings,
    vision_config,
    layer_idx: int,
) -> torch.Tensor:
    attention = ViTPluginAttention(
        block.attn,
        vision_config,
        layer_idx,
        use_plugin_op=False,
    ).eval()

    residual = hidden_states
    candidate = block.norm1(hidden_states)
    candidate = attention(
        candidate,
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
    )
    candidate = residual + candidate

    residual = candidate
    candidate = block.norm2(candidate)
    candidate = block.mlp(candidate)
    candidate = residual + candidate
    return candidate


def _forward_qwen_windowed_rope_reference(
    visual,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> torch.Tensor:
    pixel_values = pixel_values.to(device=_module_device(visual, pixel_values.device))
    if hasattr(visual, "dtype"):
        pixel_values = pixel_values.to(dtype=visual.dtype)

    hidden_states = visual.patch_embed(pixel_values)
    rotary_pos_emb = visual.rot_pos_emb(image_grid_thw)
    window_index, cu_window_seqlens = visual.get_window_index(image_grid_thw)

    window_index = torch.as_tensor(
        window_index, device=hidden_states.device, dtype=torch.long
    )
    reverse_window_index = torch.argsort(window_index)
    cu_window_seqlens = torch.as_tensor(
        cu_window_seqlens, device=hidden_states.device, dtype=torch.int32
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
    cu_seqlens = _qwen_full_cu_seqlens(image_grid_thw).to(hidden_states.device)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(
        seq_len // visual.spatial_merge_unit,
        visual.spatial_merge_unit,
        -1,
    )
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)

    rotary_pos_emb = rotary_pos_emb.to(device=hidden_states.device)
    rotary_pos_emb = rotary_pos_emb.reshape(
        seq_len // visual.spatial_merge_unit,
        visual.spatial_merge_unit,
        -1,
    )
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    for layer_idx, block in enumerate(visual.blocks):
        cu_seqlens_now = (
            cu_seqlens
            if layer_idx in visual.fullatt_block_indexes
            else cu_window_seqlens
        )
        hidden_states = _call_qwen_visual_block_reference(
            block,
            hidden_states,
            cu_seqlens_now,
            rotary_pos_emb,
            position_embeddings,
        )

    hidden_states = visual.merger(hidden_states)
    hidden_states = hidden_states[reverse_window_index, :]
    return hidden_states


def _call_qwen_visual_direct(
    visual,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
) -> torch.Tensor:
    target_device = _module_device(visual, pixel_values.device)
    pixel_values = pixel_values.to(device=target_device)
    image_grid_thw = image_grid_thw.to(device=target_device)
    if hasattr(visual, "dtype"):
        pixel_values = pixel_values.to(dtype=visual.dtype)

    call_attempts = (
        lambda: visual(pixel_values, grid_thw=image_grid_thw),
        lambda: visual(pixel_values, image_grid_thw),
        lambda: visual(pixel_values),
    )
    last_error = None
    for call in call_attempts:
        try:
            return _extract_vision_output(call())
        except TypeError as exc:
            last_error = exc
    raise last_error


def _call_qwen_get_image_features_owner(
    owner,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
):
    get_image_features = getattr(owner, "get_image_features", None)
    if not callable(get_image_features):
        return None

    call_attempts = (
        lambda: get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        ),
        lambda: get_image_features(pixel_values, image_grid_thw),
    )
    for call in call_attempts:
        try:
            return _extract_vision_output(call())
        except TypeError:
            continue
    return None


def _print_tensor_comparison(
    name: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    atol: float,
    rtol: float,
) -> bool:
    reference = reference.detach()
    candidate = candidate.detach()
    if reference.shape != candidate.shape:
        print(f"{name}: shape mismatch, ref={tuple(reference.shape)}, trt={tuple(candidate.shape)}")
        return False

    ref_float = reference.float()
    candidate_float = candidate.float()
    diff = (ref_float - candidate_float).abs()
    is_close = torch.allclose(ref_float, candidate_float, atol=atol, rtol=rtol)
    ref_finite = torch.isfinite(ref_float)
    candidate_finite = torch.isfinite(candidate_float)
    print(
        f"{name}: allclose={is_close}, "
        f"max_abs={diff.max().item():.6f}, "
        f"mean_abs={diff.mean().item():.6f}, "
        f"ref_norm={ref_float.norm().item():.6f}, "
        f"trt_norm={candidate_float.norm().item():.6f}, "
        f"ref_finite={ref_finite.sum().item()}/{ref_finite.numel()}, "
        f"trt_finite={candidate_finite.sum().item()}/{candidate_finite.numel()}"
    )
    return is_close


@torch.inference_mode()
def verify_qwen_vision_wrapper(
    model: torch.nn.Module,
    inputs,
    atol: float,
    rtol: float,
) -> None:
    print("========= Vision Wrapper Verification =========")
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    input_ids = inputs["input_ids"]
    image_mask = input_ids == model.config.image_token_id
    num_image_tokens = image_mask.sum().item()
    visual = get_vision_model(model)

    ref_image_embeds = get_qwen_image_embeds(
        model,
        pixel_values,
        image_grid_thw,
        expected_tokens=num_image_tokens,
    )
    direct_visual_embeds = _call_qwen_visual_direct(
        visual,
        pixel_values,
        image_grid_thw,
    )
    wrapper_image_embeds = _forward_qwen_windowed_rope_reference(
        visual,
        pixel_values,
        image_grid_thw,
    )

    parent_feature_embeds = None
    parent = getattr(model, "model", None)
    if isinstance(parent, torch.nn.Module):
        parent_feature_embeds = _call_qwen_get_image_features_owner(
            parent,
            pixel_values,
            image_grid_thw,
        )

    helper_close = _print_tensor_comparison(
        "HF get_image_features vs direct visual",
        ref_image_embeds,
        direct_visual_embeds,
        atol,
        rtol,
    )
    parent_close = None
    if isinstance(parent_feature_embeds, torch.Tensor):
        parent_close = _print_tensor_comparison(
            "HF model.get_image_features vs direct visual",
            parent_feature_embeds,
            direct_visual_embeds,
            atol,
            rtol,
        )
    wrapper_close = _print_tensor_comparison(
        "direct visual vs reconstructed PyTorch visual",
        direct_visual_embeds,
        wrapper_image_embeds,
        atol,
        rtol,
    )
    print(
        "vision wrapper verification summary: "
        f"helper={helper_close}, parent_helper={parent_close}, "
        f"wrapper={wrapper_close}"
    )
    print("===============================================")


@torch.inference_mode()
def verify_qwen_attention_wrapper(
    model: torch.nn.Module,
    inputs,
    args: argparse.Namespace,
    device: torch.device,
    atol: float,
    rtol: float,
) -> None:
    """
    Compare direct Qwen visual output against the same reconstructed visual
    path with attention modules replaced by ViTPluginAttention running its
    PyTorch fallback instead of the TensorRT plugin op.
    """
    print("========= Vision Attention Wrapper Verification =========")

    torch_dtype = {
        "FP16": torch.float16,
        "BF16": torch.bfloat16,
    }.get(args.precision, torch.float32)

    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    visual_model = get_vision_model(model)

    direct_visual_embeds = _call_qwen_visual_direct(
        visual_model,
        pixel_values,
        image_grid_thw,
    )

    vision_config = _get_vision_config(model.config)
    replace_vit_attention_with_plugin(
        visual_model,
        vision_config,
        use_plugin_op=False,
    )
    replacement_count = count_vit_plugin_attention_modules(visual_model)
    print(f"PyTorch attention replacement modules inserted: {replacement_count}")

    plugin_pixel_values = pixel_values.to(dtype=torch_dtype)
    input_contract, core_inputs, max_window_seq_len = _prepare_vit_plugin_inputs(
        visual_model,
        inputs,
        plugin_pixel_values,
        device,
        torch_dtype,
    )
    wrapper = ViTPluginWrapper(
        visual_model,
        input_contract=input_contract,
        max_window_seq_len=max_window_seq_len,
    ).eval()
    replacement_visual_embeds = _extract_vision_output(
        wrapper(plugin_pixel_values, **core_inputs)
    )

    replacement_close = _print_tensor_comparison(
        "direct visual vs PyTorch attention wrapper visual",
        direct_visual_embeds,
        replacement_visual_embeds,
        atol,
        rtol,
    )
    print(
        "vision attention wrapper verification summary: "
        f"attention_wrapper={replacement_close}"
    )
    print("=========================================================")


@torch.inference_mode()
def verify_qwen_attention_module(
    model: torch.nn.Module,
    inputs,
    args: argparse.Namespace,
    atol: float,
    rtol: float,
    layer_idx: int = 0,
) -> None:
    """
    Compare one original Qwen visual attention module against ViTPluginAttention
    running its PyTorch fallback.
    """
    print("========= Vision Attention Module Verification =========")

    torch_dtype = {
        "FP16": torch.float16,
        "BF16": torch.bfloat16,
    }.get(args.precision, torch.float32)

    visual = get_vision_model(model)
    (
        hidden_states,
        attention_mask,
        cu_seqlens,
        rotary_pos_emb,
        position_embeddings,
    ) = _qwen_windowed_rope_attention_inputs(
        visual,
        inputs["pixel_values"],
        inputs["image_grid_thw"],
        layer_idx,
        torch_dtype,
    )

    original_attn = visual.blocks[layer_idx].attn
    ref_attn = _call_qwen_attention_reference(
        original_attn,
        hidden_states,
        attention_mask,
        cu_seqlens,
        rotary_pos_emb,
        position_embeddings,
    )
    wrapper_attn = ViTPluginAttention(
        original_attn,
        _get_vision_config(model.config),
        layer_idx,
        use_plugin_op=False,
    ).eval()
    candidate_attn = wrapper_attn(
        hidden_states,
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
    )

    attn_close = _print_tensor_comparison(
        f"layer {layer_idx} original attention vs PyTorch attention wrapper",
        ref_attn,
        candidate_attn,
        atol,
        rtol,
    )
    print(
        "vision attention module verification summary: "
        f"layer={layer_idx}, attention={attn_close}"
    )
    print("========================================================")


@torch.inference_mode()
def verify_qwen_attention_plugin_module(
    model: torch.nn.Module,
    inputs,
    args: argparse.Namespace,
    device: torch.device,
    atol: float,
    rtol: float,
    layer_idx: int = 0,
) -> None:
    """
    Compare one original Qwen visual attention module against the real TensorRT
    ViTAttentionPlugin lowering. This isolates plugin math from the rest of the
    vision tower.
    """
    print("========= Vision Attention Plugin Module Verification =========")

    torch_dtype = {
        "FP16": torch.float16,
        "BF16": torch.bfloat16,
    }.get(args.precision, torch.float32)

    visual = get_vision_model(model)
    (
        hidden_states,
        attention_mask,
        cu_seqlens,
        rotary_pos_emb,
        position_embeddings,
    ) = _qwen_windowed_rope_attention_inputs(
        visual,
        inputs["pixel_values"],
        inputs["image_grid_thw"],
        layer_idx,
        torch_dtype,
    )

    original_attn = visual.blocks[layer_idx].attn
    ref_attn = _call_qwen_attention_reference(
        original_attn,
        hidden_states,
        attention_mask,
        cu_seqlens,
        rotary_pos_emb,
        position_embeddings,
    )

    vision_config = _get_vision_config(model.config)
    fallback_attn = ViTPluginAttention(
        original_attn,
        vision_config,
        layer_idx,
        use_plugin_op=False,
    ).eval()
    plugin_rope_position_embeddings = tuple(
        value.to(dtype=hidden_states.dtype) for value in position_embeddings
    )
    fallback_with_plugin_rope = fallback_attn(
        hidden_states,
        attention_mask=attention_mask,
        position_embeddings=plugin_rope_position_embeddings,
    )
    fallback_close = _print_tensor_comparison(
        f"layer {layer_idx} original attention vs PyTorch wrapper with plugin RoPE dtype",
        ref_attn,
        fallback_with_plugin_rope,
        atol,
        rtol,
    )

    class _AttentionPluginModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = ViTPluginAttention(
                original_attn,
                vision_config,
                layer_idx,
                use_plugin_op=True,
            ).eval()

        def forward(self, hidden_states, attention_mask, cos, sin):
            return self.attn(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=(cos, sin),
            )

    load_vit_plugin()
    register_vit_plugin_op()
    _set_vit_plugin_config_from_vision(
        vision_config,
        inputs["pixel_values"].to(dtype=torch_dtype),
    )
    print(f"ViT plugin config: {get_vit_plugin_config()}")

    reset_vit_plugin_conversion_count()
    compiled_attn = compile_vit_plugin_model(
        _AttentionPluginModule().eval(),
        (
            hidden_states,
            attention_mask,
            position_embeddings[0],
            position_embeddings[1],
        ),
        device,
        debug=args.debug,
    )
    plugin_attn = _extract_hidden_states(
        compiled_attn(
            hidden_states,
            attention_mask,
            position_embeddings[0],
            position_embeddings[1],
        )
    )
    plugin_close = _print_tensor_comparison(
        f"layer {layer_idx} original attention vs TensorRT plugin attention",
        ref_attn,
        plugin_attn,
        atol,
        rtol,
    )
    print(
        "vision attention plugin module verification summary: "
        f"layer={layer_idx}, fallback_plugin_rope={fallback_close}, "
        f"plugin={plugin_close}, conversions={get_vit_plugin_conversion_count()}"
    )
    print("===============================================================")


@torch.inference_mode()
def verify_qwen_block_plugin_module(
    model: torch.nn.Module,
    inputs,
    args: argparse.Namespace,
    device: torch.device,
    atol: float,
    rtol: float,
    layer_idx: int = 0,
) -> None:
    """
    Compare one original Qwen visual block against the same block compiled with
    the real TensorRT ViTAttentionPlugin inside it.
    """
    print("========= Vision Block Plugin Module Verification =========")

    torch_dtype = {
        "FP16": torch.float16,
        "BF16": torch.bfloat16,
    }.get(args.precision, torch.float32)

    visual = get_vision_model(model)
    (
        hidden_states,
        attention_mask,
        cu_seqlens,
        rotary_pos_emb,
        position_embeddings,
    ) = _qwen_windowed_rope_block_inputs(
        visual,
        inputs["pixel_values"],
        inputs["image_grid_thw"],
        layer_idx,
        torch_dtype,
    )

    block = visual.blocks[layer_idx]
    ref_block = _call_qwen_visual_block_reference(
        block,
        hidden_states,
        cu_seqlens,
        rotary_pos_emb,
        position_embeddings,
    )

    vision_config = _get_vision_config(model.config)

    class _BlockPluginModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = block.norm1
            self.attn = ViTPluginAttention(
                block.attn,
                vision_config,
                layer_idx,
                use_plugin_op=True,
            ).eval()
            self.norm2 = block.norm2
            self.mlp = block.mlp

        def forward(self, hidden_states, attention_mask, cos, sin):
            residual = hidden_states
            candidate = self.norm1(hidden_states)
            candidate = self.attn(
                candidate,
                attention_mask=attention_mask,
                position_embeddings=(cos, sin),
            )
            candidate = residual + candidate

            residual = candidate
            candidate = self.norm2(candidate)
            candidate = self.mlp(candidate)
            return residual + candidate

    load_vit_plugin()
    register_vit_plugin_op()
    _set_vit_plugin_config_from_vision(
        vision_config,
        inputs["pixel_values"].to(dtype=torch_dtype),
    )
    print(f"ViT plugin config: {get_vit_plugin_config()}")

    reset_vit_plugin_conversion_count()
    compiled_block = compile_vit_plugin_model(
        _BlockPluginModule().eval(),
        (
            hidden_states,
            attention_mask,
            position_embeddings[0],
            position_embeddings[1],
        ),
        device,
        debug=args.debug,
    )
    plugin_block = _extract_hidden_states(
        compiled_block(
            hidden_states,
            attention_mask,
            position_embeddings[0],
            position_embeddings[1],
        )
    )
    block_close = _print_tensor_comparison(
        f"layer {layer_idx} original block vs TensorRT plugin block",
        ref_block,
        plugin_block,
        atol,
        rtol,
    )
    print(
        "vision block plugin module verification summary: "
        f"layer={layer_idx}, block={block_close}, "
        f"conversions={get_vit_plugin_conversion_count()}"
    )
    print("===========================================================")


@torch.inference_mode()
def verify_qwen_block_plugin_parts(
    model: torch.nn.Module,
    inputs,
    args: argparse.Namespace,
    device: torch.device,
    atol: float,
    rtol: float,
    layer_idx: int = 0,
) -> None:
    """
    Split one Qwen visual block into attention-residual and MLP-residual halves
    and compile each half. This isolates late-block drift to the surrounding
    block ops instead of the attention plugin itself.
    """
    print("========= Vision Block Plugin Parts Verification =========")

    torch_dtype = {
        "FP16": torch.float16,
        "BF16": torch.bfloat16,
    }.get(args.precision, torch.float32)

    visual = get_vision_model(model)
    (
        hidden_states,
        attention_mask,
        cu_seqlens,
        rotary_pos_emb,
        position_embeddings,
    ) = _qwen_windowed_rope_block_inputs(
        visual,
        inputs["pixel_values"],
        inputs["image_grid_thw"],
        layer_idx,
        torch_dtype,
    )

    block = visual.blocks[layer_idx]
    norm1_hidden_states = block.norm1(hidden_states)
    ref_attn = _call_qwen_attention_reference(
        block.attn,
        norm1_hidden_states,
        attention_mask,
        cu_seqlens,
        rotary_pos_emb,
        position_embeddings,
    )
    ref_after_attn = hidden_states + ref_attn
    ref_mlp = block.mlp(block.norm2(ref_after_attn))
    ref_after_mlp = ref_after_attn + ref_mlp

    vision_config = _get_vision_config(model.config)
    load_vit_plugin()
    register_vit_plugin_op()
    _set_vit_plugin_config_from_vision(
        vision_config,
        inputs["pixel_values"].to(dtype=torch_dtype),
    )
    print(f"ViT plugin config: {get_vit_plugin_config()}")

    class _AttentionResidualPluginModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = block.norm1
            self.attn = ViTPluginAttention(
                block.attn,
                vision_config,
                layer_idx,
                use_plugin_op=True,
            ).eval()

        def forward(self, hidden_states, attention_mask, cos, sin):
            residual = hidden_states
            candidate = self.norm1(hidden_states)
            candidate = self.attn(
                candidate,
                attention_mask=attention_mask,
                position_embeddings=(cos, sin),
            )
            return residual + candidate

    reset_vit_plugin_conversion_count()
    compiled_attn_residual = compile_vit_plugin_model(
        _AttentionResidualPluginModule().eval(),
        (
            hidden_states,
            attention_mask,
            position_embeddings[0],
            position_embeddings[1],
        ),
        device,
        debug=args.debug,
    )
    plugin_after_attn = _extract_hidden_states(
        compiled_attn_residual(
            hidden_states,
            attention_mask,
            position_embeddings[0],
            position_embeddings[1],
        )
    )
    attention_residual_close = _print_tensor_comparison(
        f"layer {layer_idx} attention residual half",
        ref_after_attn,
        plugin_after_attn,
        atol,
        rtol,
    )
    attention_conversions = get_vit_plugin_conversion_count()

    class _MlpResidualModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm2 = block.norm2
            self.mlp = block.mlp

        def forward(self, hidden_states):
            residual = hidden_states
            candidate = self.norm2(hidden_states)
            candidate = self.mlp(candidate)
            return residual + candidate

    reset_vit_plugin_conversion_count()
    compiled_mlp_residual = compile_vit_plugin_model(
        _MlpResidualModule().eval(),
        (ref_after_attn,),
        device,
        debug=args.debug,
    )
    plugin_after_mlp = _extract_hidden_states(compiled_mlp_residual(ref_after_attn))
    mlp_residual_close = _print_tensor_comparison(
        f"layer {layer_idx} MLP residual half",
        ref_after_mlp,
        plugin_after_mlp,
        atol,
        rtol,
    )
    mlp_conversions = get_vit_plugin_conversion_count()

    chained_after_mlp = _extract_hidden_states(
        compiled_mlp_residual(plugin_after_attn)
    )
    chained_close = _print_tensor_comparison(
        f"layer {layer_idx} chained compiled attention half into MLP half",
        ref_after_mlp,
        chained_after_mlp,
        atol,
        rtol,
    )

    print(
        "vision block plugin parts verification summary: "
        f"layer={layer_idx}, attention_residual={attention_residual_close}, "
        f"mlp_residual={mlp_residual_close}, "
        f"chained={chained_close}, "
        f"attention_conversions={attention_conversions}, "
        f"mlp_conversions={mlp_conversions}"
    )
    print("===========================================================")


@torch.inference_mode()
def verify_qwen_block_wrapper(
    model: torch.nn.Module,
    inputs,
    args: argparse.Namespace,
    atol: float,
    rtol: float,
    layer_idx: int = 0,
) -> None:
    """
    Compare one original Qwen visual block against the same block manually
    wired with ViTPluginAttention running its PyTorch fallback.
    """
    print("========= Vision Block Wrapper Verification =========")

    torch_dtype = {
        "FP16": torch.float16,
        "BF16": torch.bfloat16,
    }.get(args.precision, torch.float32)

    visual = get_vision_model(model)
    (
        hidden_states,
        attention_mask,
        cu_seqlens,
        rotary_pos_emb,
        position_embeddings,
    ) = _qwen_windowed_rope_block_inputs(
        visual,
        inputs["pixel_values"],
        inputs["image_grid_thw"],
        layer_idx,
        torch_dtype,
    )

    block = visual.blocks[layer_idx]
    ref_block = _call_qwen_visual_block_reference(
        block,
        hidden_states,
        cu_seqlens,
        rotary_pos_emb,
        position_embeddings,
    )
    candidate_block = _call_qwen_block_with_attention_wrapper(
        block,
        hidden_states,
        attention_mask,
        position_embeddings,
        _get_vision_config(model.config),
        layer_idx,
    )

    block_close = _print_tensor_comparison(
        f"layer {layer_idx} original block vs PyTorch attention-wrapper block",
        ref_block,
        candidate_block,
        atol,
        rtol,
    )
    print(
        "vision block wrapper verification summary: "
        f"layer={layer_idx}, block={block_close}"
    )
    print("===================================================")


@torch.inference_mode()
def verify_qwen_vlm_components(
    model: torch.nn.Module,
    trt_model: torch.nn.Module,
    inputs,
    emb_layer: torch.nn.Embedding,
    tokenizer,
    atol: float,
    rtol: float,
    verify_stage: str = "all",
) -> None:
    """
    Compare the compiled Qwen VLM components against the PyTorch reference.
    This checks the vision tower, then isolates the LM by feeding both models
    the same multimodal embeddings and Qwen multimodal RoPE position ids.
    """
    print("========= Component Verification =========")

    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    image_mask = input_ids == model.config.image_token_id
    num_image_tokens = image_mask.sum().item()
    device = input_ids.device

    ref_image_embeds = get_qwen_image_embeds(
        model,
        pixel_values,
        image_grid_thw,
        expected_tokens=num_image_tokens,
    ).to(device=device)
    trt_image_embeds = get_qwen_image_embeds(
        trt_model,
        pixel_values,
        image_grid_thw,
        expected_tokens=num_image_tokens,
    ).to(device=device)
    direct_visual_embeds = _call_qwen_visual_direct(
        get_vision_model(model),
        pixel_values,
        image_grid_thw,
    ).to(device=device)
    trt_direct_visual_embeds = _call_qwen_visual_direct(
        get_vision_model(trt_model),
        pixel_values,
        image_grid_thw,
    ).to(device=device)
    direct_vision_close = _print_tensor_comparison(
        "direct visual embeddings", direct_visual_embeds, trt_direct_visual_embeds, atol, rtol
    )
    helper_vision_close = _print_tensor_comparison(
        "get_image_features embeddings", ref_image_embeds, trt_image_embeds, atol, rtol
    )
    if verify_stage == "vision":
        print(
            "component verification summary: "
            f"direct_vision={direct_vision_close}, "
            f"get_image_features={helper_vision_close}"
        )
        print("==========================================")
        return

    seq_tokens = input_ids.clone()
    seq_embeds = emb_layer(seq_tokens)
    mask_expanded = image_mask.unsqueeze(-1).expand_as(seq_embeds)
    seq_embeds = seq_embeds.masked_scatter(
        mask_expanded,
        ref_image_embeds.to(device=seq_embeds.device, dtype=seq_embeds.dtype),
    )
    position_ids = get_qwen_position_ids(
        model,
        seq_tokens,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    ).to(device=seq_embeds.device)

    ref_hidden = _call_language_model_for_verify(
        get_language_model(model), seq_embeds, position_ids
    ).to(device)
    trt_hidden = _call_language_model_for_verify(
        get_language_model(trt_model), seq_embeds, position_ids
    ).to(device)
    lm_close = _print_tensor_comparison(
        "LM hidden states", ref_hidden, trt_hidden, atol, rtol
    )

    ref_logits = model.lm_head(
        ref_hidden[:, -1, :].to(_module_device(model.lm_head, device))
    ).to(device)
    trt_logits = trt_model.lm_head(
        trt_hidden[:, -1, :].to(_module_device(trt_model.lm_head, device))
    ).to(device)
    logits_close = _print_tensor_comparison("next-token logits", ref_logits, trt_logits, atol, rtol)

    ref_next = ref_logits.argmax(dim=-1)
    trt_next = trt_logits.argmax(dim=-1)
    print(
        "next-token argmax: "
        f"ref={ref_next.tolist()} ({tokenizer.batch_decode(ref_next[:, None])}), "
        f"trt={trt_next.tolist()} ({tokenizer.batch_decode(trt_next[:, None])}), "
        f"match={torch.equal(ref_next, trt_next)}"
    )
    print(
        "component verification summary: "
        f"direct_vision={direct_vision_close}, "
        f"get_image_features={helper_vision_close}, "
        f"lm={lm_close}, logits={logits_close}"
    )
    print("==========================================")


def _qwen_inputs_embeds_with_vision(
    model_for_vision: torch.nn.Module,
    reference_model: torch.nn.Module,
    inputs,
    emb_layer: torch.nn.Embedding,
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    image_mask = input_ids == reference_model.config.image_token_id
    num_image_tokens = image_mask.sum().item()

    image_embeds = get_qwen_image_embeds(
        model_for_vision,
        pixel_values,
        image_grid_thw,
        expected_tokens=num_image_tokens,
    )
    seq_tokens = input_ids.clone()
    seq_embeds = emb_layer(seq_tokens)
    mask_expanded = image_mask.unsqueeze(-1).expand_as(seq_embeds)
    seq_embeds = seq_embeds.masked_scatter(
        mask_expanded,
        image_embeds.to(device=seq_embeds.device, dtype=seq_embeds.dtype),
    )
    return seq_tokens, seq_embeds


def _print_generated_token_divergence(
    ref_gen_tokens: torch.Tensor,
    trt_gen_tokens: torch.Tensor,
    tokenizer,
) -> None:
    min_len = min(ref_gen_tokens.shape[1], trt_gen_tokens.shape[1])
    mismatch_idx = None
    for idx in range(min_len):
        if not torch.equal(ref_gen_tokens[:, idx], trt_gen_tokens[:, idx]):
            mismatch_idx = idx
            break

    if mismatch_idx is None and ref_gen_tokens.shape[1] == trt_gen_tokens.shape[1]:
        print("generated divergence: none")
        return

    if mismatch_idx is None:
        mismatch_idx = min_len
        print(
            "generated divergence: length mismatch after shared prefix "
            f"of {min_len} tokens"
        )
    else:
        ref_tok = ref_gen_tokens[:, mismatch_idx]
        trt_tok = trt_gen_tokens[:, mismatch_idx]
        print(
            "generated divergence: "
            f"step={mismatch_idx}, "
            f"ref={ref_tok.tolist()} ({tokenizer.batch_decode(ref_tok[:, None])}), "
            f"trt={trt_tok.tolist()} ({tokenizer.batch_decode(trt_tok[:, None])})"
        )

    context_start = max(0, mismatch_idx - 4)
    context_end = min(
        max(ref_gen_tokens.shape[1], trt_gen_tokens.shape[1]),
        mismatch_idx + 5,
    )
    ref_context = ref_gen_tokens[:, context_start : min(context_end, ref_gen_tokens.shape[1])]
    trt_context = trt_gen_tokens[:, context_start : min(context_end, trt_gen_tokens.shape[1])]
    print(
        "generated divergence context: "
        f"ref={tokenizer.batch_decode(ref_context, skip_special_tokens=True)}, "
        f"trt={tokenizer.batch_decode(trt_context, skip_special_tokens=True)}"
    )


@torch.inference_mode()
def verify_qwen_vision_semantics(
    model: torch.nn.Module,
    trt_model: torch.nn.Module,
    inputs,
    emb_layer: torch.nn.Embedding,
    tokenizer,
    args: argparse.Namespace,
    atol: float,
    rtol: float,
) -> None:
    """
    Check whether compiled/plugin vision changes the language-model behavior
    before compiling the LM. The first-token comparison uses the original
    PyTorch LM for both paths, so the only intentional input difference is the
    vision embedding source.
    """
    print("========= Vision Semantic Verification =========")

    attention_mask = inputs.get("attention_mask")
    ref_tokens, ref_embeds = _qwen_inputs_embeds_with_vision(
        model, model, inputs, emb_layer
    )
    trt_tokens, trt_embeds = _qwen_inputs_embeds_with_vision(
        trt_model, model, inputs, emb_layer
    )
    position_ids = get_qwen_position_ids(
        model,
        ref_tokens,
        image_grid_thw=inputs["image_grid_thw"],
        attention_mask=attention_mask,
    ).to(device=ref_embeds.device)

    language_model = get_language_model(model)
    ref_hidden = _call_language_model_for_verify(
        language_model,
        ref_embeds,
        position_ids,
    )
    trt_hidden = _call_language_model_for_verify(
        language_model,
        trt_embeds,
        position_ids,
    )
    ref_logits = model.lm_head(
        ref_hidden[:, -1, :].to(_module_device(model.lm_head, ref_hidden.device))
    ).to(ref_embeds.device)
    trt_logits = model.lm_head(
        trt_hidden[:, -1, :].to(_module_device(model.lm_head, trt_hidden.device))
    ).to(ref_embeds.device)
    logits_close = _print_tensor_comparison(
        "first-token logits with PyTorch LM",
        ref_logits,
        trt_logits,
        atol,
        rtol,
    )

    ref_next = ref_logits.argmax(dim=-1)
    trt_next = trt_logits.argmax(dim=-1)
    print(
        "first-token argmax: "
        f"ref={ref_next.tolist()} ({tokenizer.batch_decode(ref_next[:, None])}), "
        f"trt={trt_next.tolist()} ({tokenizer.batch_decode(trt_next[:, None])}), "
        f"match={torch.equal(ref_next, trt_next)}"
    )

    ref_gen_tokens = generate_mm_qwen2_5_vl(
        model,
        inputs["pixel_values"],
        inputs["input_ids"],
        inputs["image_grid_thw"],
        attention_mask,
        tokenizer.eos_token_id,
        emb_layer,
        max_new_tokens=args.num_tokens,
    )
    trt_gen_tokens = generate_mm_qwen2_5_vl(
        trt_model,
        inputs["pixel_values"],
        inputs["input_ids"],
        inputs["image_grid_thw"],
        attention_mask,
        tokenizer.eos_token_id,
        emb_layer,
        max_new_tokens=args.num_tokens,
    )
    generated_match = torch.equal(ref_gen_tokens, trt_gen_tokens)
    print(
        "generated tokens: "
        f"ref={ref_gen_tokens.tolist()} "
        f"({tokenizer.batch_decode(ref_gen_tokens, skip_special_tokens=True)}), "
        f"trt={trt_gen_tokens.tolist()} "
        f"({tokenizer.batch_decode(trt_gen_tokens, skip_special_tokens=True)}), "
        f"match={generated_match}"
    )
    _print_generated_token_divergence(ref_gen_tokens, trt_gen_tokens, tokenizer)
    print(
        "vision semantic verification summary: "
        f"logits={logits_close}, first_token={torch.equal(ref_next, trt_next)}, "
        f"generated={generated_match}"
    )
    print("==============================================")


# -----------------------------------------------------------------------------#
# Main driver
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run VLM inference (PyTorch & TensorRT back-ends)"
    )
    parser.add_argument(
        "--model",
        default="nvidia/Eagle2-2B",
        help="VLM model name",
    )
    parser.add_argument(
        "--processor",
        default="",
        help="Processor name/path. Defaults to --model.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow Hugging Face remote model/processor code.",
    )
    parser.add_argument(
        "--attn_implementation",
        default="",
        help=(
            "Attention implementation passed to from_pretrained. Defaults to "
            "SDPA for the torchtrt vision backend and model default for plugin."
        ),
    )
    parser.add_argument("--prompt", default="Describe this image.", help="Prompt text")
    parser.add_argument(
        "--precision",
        default="FP16",
        choices=["FP16", "BF16", "FP32"],
        help="Computation precision",
    )
    parser.add_argument("--iterations", type=int, default=5, help="# iterations")
    parser.add_argument("--min_block_size", type=int, default=1, help="Min block size")
    parser.add_argument("--num_tokens", type=int, default=128, help="# new tokens")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--isl", type=int, default=2048, help="Input seq length")
    parser.add_argument(
        "--enable_pytorch_run",
        action="store_true",
        help="Run the PyTorch baseline as well",
    )
    parser.add_argument(
        "--verify_accuracy",
        action="store_true",
        help="Compare PyTorch and TensorRT VLM component outputs before generation.",
    )
    parser.add_argument(
        "--verify_stage",
        default="all",
        choices=[
            "vision_wrapper",
            "attention_module",
            "attention_plugin_module",
            "block_plugin_module",
            "block_plugin_parts",
            "block_wrapper",
            "attention_wrapper",
            "vision",
            "vision_semantic",
            "all",
        ],
        help=(
            "Component verification stage. 'vision_wrapper' compares the HF "
            "visual output against the reconstructed PyTorch visual path before "
            "TRT/plugin compilation; 'attention_module' checks one attention "
            "module with the PyTorch fallback; 'attention_plugin_module' checks "
            "one attention module with the real TensorRT plugin; "
            "'block_plugin_module' checks one full visual block with the real "
            "TensorRT plugin; "
            "'block_plugin_parts' checks one block split into attention and MLP "
            "halves; "
            "'block_wrapper' checks one full visual block; "
            "'attention_wrapper' replaces attention with the plugin wrapper but "
            "runs PyTorch attention; 'vision' stops after comparing visual "
            "embeddings; 'vision_semantic' checks whether compiled vision changes "
            "PyTorch LM logits or generated tokens; 'all' also checks LM hidden "
            "states and logits."
        ),
    )
    parser.add_argument(
        "--verify_atol",
        type=float,
        default=5e-2,
        help="Absolute tolerance for --verify_accuracy tensor comparisons.",
    )
    parser.add_argument(
        "--verify_rtol",
        type=float,
        default=5e-2,
        help="Relative tolerance for --verify_accuracy tensor comparisons.",
    )
    parser.add_argument(
        "--verify_layer",
        type=int,
        default=0,
        help="Layer index used by --verify_stage attention_module.",
    )
    parser.add_argument(
        "--cache",
        default="",
        choices=["", "static_v1"],
        help="KV-cache variant to use",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable Torch-TensorRT debug logs"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Enable benchmarking mode"
    )
    parser.add_argument(
        "--vision_backend",
        default="torchtrt",
        choices=["torchtrt", "plugin"],
        help=(
            "Vision backend. 'torchtrt' keeps the existing component compiler; "
            "'plugin' uses the TensorRT-Edge-LLM ViT attention plugin where supported."
        ),
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to local image file. If not provided, uses default URL image.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on (e.g., 'cuda:0', 'cuda:1')",
    )
    parser.add_argument(
        "--disable_tf32",
        action="store_false",
        default=True,
        help="Disable TF32 precision for TensorRT compilation (default: True)",
    )
    parser.add_argument(
        "--use_python_runtime",
        action="store_false",
        default=True,
        help="Use Python runtime for TensorRT compilation (default: True)",
    )
    parser.add_argument(
        "--offload_module_to_cpu",
        action="store_false",
        default=True,
        help="Offload module to CPU for TensorRT compilation (default: True)",
    )

    args = parser.parse_args()

    if args.vision_backend == "plugin" and not VIT_PLUGIN_AVAILABLE:
        raise RuntimeError(
            "ViT plugin vision backend requested but plugin utilities are not available."
        )

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    # -------------------------------------------------------------------------#
    # 1. Model / processor / embeddings
    # -------------------------------------------------------------------------#
    dtype = {
        "FP16": torch.float16,
        "BF16": torch.bfloat16,
    }.get(args.precision, torch.float32)

    model, processor, emb_layer = get_model(args, device, dtype)

    # -------------------------------------------------------------------------#
    # 2. Input construction (image + text prompt)
    # -------------------------------------------------------------------------#
    inputs = load_inputs(args, processor, device)

    max_output_len = inputs["input_ids"].shape[1] + args.num_tokens

    if args.verify_accuracy and args.verify_stage == "vision_wrapper":
        if _is_qwen2_5_vl(args.model):
            verify_qwen_vision_wrapper(
                model,
                inputs,
                args.verify_atol,
                args.verify_rtol,
            )
            raise SystemExit(0)
        print(
            "--verify_stage vision_wrapper currently supports Qwen2.5-VL "
            "windowed-RoPE visual towers."
        )
        raise SystemExit(0)

    if args.verify_accuracy and args.verify_stage == "attention_wrapper":
        if _is_qwen2_5_vl(args.model):
            verify_qwen_attention_wrapper(
                model,
                inputs,
                args,
                device,
                args.verify_atol,
                args.verify_rtol,
            )
            raise SystemExit(0)
        print(
            "--verify_stage attention_wrapper currently supports Qwen2.5-VL "
            "windowed-RoPE visual towers."
        )
        raise SystemExit(0)

    if args.verify_accuracy and args.verify_stage == "attention_module":
        if _is_qwen2_5_vl(args.model):
            verify_qwen_attention_module(
                model,
                inputs,
                args,
                args.verify_atol,
                args.verify_rtol,
                args.verify_layer,
            )
            raise SystemExit(0)
        print(
            "--verify_stage attention_module currently supports Qwen2.5-VL "
            "windowed-RoPE visual towers."
        )
        raise SystemExit(0)

    if args.verify_accuracy and args.verify_stage == "attention_plugin_module":
        if _is_qwen2_5_vl(args.model):
            verify_qwen_attention_plugin_module(
                model,
                inputs,
                args,
                device,
                args.verify_atol,
                args.verify_rtol,
                args.verify_layer,
            )
            raise SystemExit(0)
        print(
            "--verify_stage attention_plugin_module currently supports "
            "Qwen2.5-VL windowed-RoPE visual towers."
        )
        raise SystemExit(0)

    if args.verify_accuracy and args.verify_stage == "block_plugin_module":
        if _is_qwen2_5_vl(args.model):
            verify_qwen_block_plugin_module(
                model,
                inputs,
                args,
                device,
                args.verify_atol,
                args.verify_rtol,
                args.verify_layer,
            )
            raise SystemExit(0)
        print(
            "--verify_stage block_plugin_module currently supports "
            "Qwen2.5-VL windowed-RoPE visual towers."
        )
        raise SystemExit(0)

    if args.verify_accuracy and args.verify_stage == "block_plugin_parts":
        if _is_qwen2_5_vl(args.model):
            verify_qwen_block_plugin_parts(
                model,
                inputs,
                args,
                device,
                args.verify_atol,
                args.verify_rtol,
                args.verify_layer,
            )
            raise SystemExit(0)
        print(
            "--verify_stage block_plugin_parts currently supports "
            "Qwen2.5-VL windowed-RoPE visual towers."
        )
        raise SystemExit(0)

    if args.verify_accuracy and args.verify_stage == "block_wrapper":
        if _is_qwen2_5_vl(args.model):
            verify_qwen_block_wrapper(
                model,
                inputs,
                args,
                args.verify_atol,
                args.verify_rtol,
                args.verify_layer,
            )
            raise SystemExit(0)
        print(
            "--verify_stage block_wrapper currently supports Qwen2.5-VL "
            "windowed-RoPE visual towers."
        )
        raise SystemExit(0)

    # -------------------------------------------------------------------------#
    # 3. Optional: PyTorch baseline
    # -------------------------------------------------------------------------#

    pyt_gen_tokens = pyt_timings = pyt_stats = None
    if args.enable_pytorch_run:
        # For benchmarking, we run the generation with timing enabled.
        # For regular runs, we run without timing for a single output.
        if args.benchmark:
            if _is_qwen2_5_vl(args.model):
                (
                    pyt_gen_tokens,
                    _,
                    overall_time,
                    _,
                    _,
                ) = generate_mm_qwen2_5_vl(
                    model,
                    inputs["pixel_values"],
                    inputs["input_ids"],
                    inputs["image_grid_thw"],
                    inputs.get("attention_mask"),
                    processor.tokenizer.eos_token_id,
                    emb_layer,
                    max_new_tokens=args.num_tokens,
                    with_timing=True,
                )
            else:  # eagle2
                (
                    pyt_gen_tokens,
                    _,
                    overall_time,
                    _,
                    _,
                ) = generate_mm(
                    model,
                    inputs["pixel_values"],
                    inputs["input_ids"],
                    processor.tokenizer.eos_token_id,
                    emb_layer,
                    max_new_tokens=args.num_tokens,
                    with_timing=True,
                )
            pyt_stats = record_stats(
                "PyTorch",
                [overall_time / 1000],  # time_generate returns seconds
                args.precision,
                batch_size=args.batch_size,
            )
        else:
            if _is_qwen2_5_vl(args.model):
                pyt_gen_tokens = generate_mm_qwen2_5_vl(
                    model,
                    inputs["pixel_values"],
                    inputs["input_ids"],
                    inputs["image_grid_thw"],
                    inputs.get("attention_mask"),
                    processor.tokenizer.eos_token_id,
                    emb_layer,
                    max_new_tokens=args.num_tokens,
                )
            else:  # eagle2
                pyt_gen_tokens = generate_mm(
                    model,
                    inputs["pixel_values"],
                    inputs["input_ids"],
                    processor.tokenizer.eos_token_id,
                    emb_layer,
                    max_new_tokens=args.num_tokens,
                )

    # -------------------------------------------------------------------------#
    # 4. Torch-TensorRT compile & run
    # -------------------------------------------------------------------------#

    trt_model = copy.deepcopy(model)
    # 4.1. Vision model compilation
    # --- Add vision model compilation --- #
    trt_vision = compile_vision_torchtrt(trt_model, args, inputs, device)
    set_vision_model(trt_model, trt_vision)

    if args.verify_accuracy and args.verify_stage == "vision":
        if _is_qwen2_5_vl(args.model):
            verify_qwen_vlm_components(
                model,
                trt_model,
                inputs,
                emb_layer,
                processor.tokenizer,
                args.verify_atol,
                args.verify_rtol,
                args.verify_stage,
            )
            raise SystemExit(0)
        print(
            "--verify_stage vision currently has detailed component checks for "
            "Qwen2.5-VL."
        )
        raise SystemExit(0)

    if args.verify_accuracy and args.verify_stage == "vision_semantic":
        if _is_qwen2_5_vl(args.model):
            verify_qwen_vision_semantics(
                model,
                trt_model,
                inputs,
                emb_layer,
                processor.tokenizer,
                args,
                args.verify_atol,
                args.verify_rtol,
            )
            raise SystemExit(0)
        print(
            "--verify_stage vision_semantic currently supports Qwen2.5-VL."
        )
        raise SystemExit(0)

    # -------------------------------------------------------------------------#
    # 4.2. Language model compilation
    # -------------------------------------------------------------------------#
    # Register static cache lowering passes if requested
    # Cache is not applied to vision model.
    print("--- Registering SDPA lowering pass locally for LM compilation ---")
    from torchtrt_ext import register_sdpa

    register_sdpa.enable_sdpa_converter(args.model, model.config)

    if args.cache == "static_v1":
        import static_cache_v1  # noqa: F401
    elif args.cache not in ("", None):
        raise ValueError(
            f"Cache mode '{args.cache}' is not supported. Only 'static_v1' is supported."
        )

    trt_lm = compile_lm_torchtrt(trt_model, args, device)
    set_language_model(trt_model, trt_lm)

    emb_layer = emb_layer.to(device)
    if _is_qwen2_5_vl(args.model) and hasattr(trt_model, "lm_head"):
        trt_model.lm_head = trt_model.lm_head.to(device)

    if args.verify_accuracy:
        if _is_qwen2_5_vl(args.model):
            verify_qwen_vlm_components(
                model,
                trt_model,
                inputs,
                emb_layer,
                processor.tokenizer,
                args.verify_atol,
                args.verify_rtol,
                args.verify_stage,
            )
        else:
            print(
                "--verify_accuracy currently has detailed component checks for "
                "Qwen2.5-VL. Falling back to generated-token comparison."
            )

    if args.cache == "static_v1":
        if _is_qwen2_5_vl(args.model):
            trt_generate = generate_mm_qwen2_5_vl_with_static_cache
        else:  # eagle2
            trt_generate = generate_mm_with_static_cache
    else:
        if _is_qwen2_5_vl(args.model):
            trt_generate = generate_mm_qwen2_5_vl
        else:  # eagle2
            trt_generate = generate_mm

    # Prepare args for generate function
    generate_args = {
        "model": trt_model,
        "pixel_values": inputs["pixel_values"],
        "input_ids": inputs["input_ids"],
        "eos_token_id": processor.tokenizer.eos_token_id,
        "emb_layer": emb_layer,
        "max_new_tokens": args.num_tokens,
    }
    if _is_qwen2_5_vl(args.model):
        generate_args["image_grid_thw"] = inputs["image_grid_thw"]
        generate_args["attention_mask"] = inputs.get("attention_mask")

    if args.cache == "static_v1" or args.benchmark:
        generate_args["with_timing"] = True

    if args.cache == "static_v1":
        generate_args["device"] = device

    # Run TRT generation
    trt_output = trt_generate(**generate_args)

    # Unpack results
    if args.benchmark or args.cache == "static_v1":
        trt_gen_tokens, _, overall_time, _, _ = trt_output
        trt_stats = record_stats(
            "TensorRT",
            [overall_time / 1000],  # time is in ms, convert to s
            args.precision,
            batch_size=args.batch_size,
        )
    else:
        trt_gen_tokens = trt_output

    # -------------------------------------------------------------------------#
    # 5. Reporting
    # -------------------------------------------------------------------------#
    if not args.benchmark:
        if args.enable_pytorch_run:
            print_outputs("PyTorch", pyt_gen_tokens, processor.tokenizer)
        print_outputs("TensorRT", trt_gen_tokens, processor.tokenizer)

        if args.enable_pytorch_run:
            print(
                f"PyTorch and TensorRT outputs match: "
                f"{torch.equal(pyt_gen_tokens, trt_gen_tokens)}"
            )

    if args.benchmark:
        if args.enable_pytorch_run:
            print("========= PyTorch PERFORMANCE =========\n")
            print(pyt_stats)
        print("=====================\n")
        print("========= TensorRT PERFORMANCE =========\n")
        print(trt_stats)
