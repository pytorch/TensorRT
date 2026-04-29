"""
Auto-detect model family from a HuggingFace model tag.

Uses AutoConfig (a single config-file fetch, no weights downloaded) to get
model_type, then maps it to one of the supported strategy names:
  "llm"            – decoder-only causal LM (GPT, LLaMA, Qwen, Gemma, ...)
  "encoder"        – encoder-only / sequence classification (BERT, RoBERTa, ViT, ...)
  "seq2seq"        – encoder-decoder (T5, BART, mT5, ...)
  "diffusion"      – image diffusion pipeline (stable-diffusion, FLUX, ...)
  "video_diffusion"– video diffusion pipeline (CogVideoX, AnimateDiff, SVD, ...)
  "audio"          – speech / ASR (Whisper, wav2vec2, ...)

Returns the family string.  Raises ValueError if the family cannot be
determined and no --task override was given.
"""

from __future__ import annotations

from typing import Optional

# Maps HuggingFace model_type strings to strategy family names.
# Keep sorted within each group for readability.
_LLM_TYPES = {
    "bloom",
    "codegen",
    "falcon",
    "gemma",
    "gemma2",
    "gemma3",
    "gpt2",
    "gpt_bigcode",
    "gpt_neo",
    "gpt_neox",
    "gptj",
    "llama",
    "mistral",
    "mixtral",
    "mpt",
    "opt",
    "phi",
    "phi3",
    "qwen2",
    "qwen2_moe",
    "starcoder2",
    "stablelm",
}

_ENCODER_TYPES = {
    "albert",
    "bert",
    "camembert",
    "convnext",
    "deberta",
    "deberta-v2",
    "distilbert",
    "electra",
    "mobilenet_v2",
    "resnet",
    "roberta",
    "swin",
    "vit",
    "xlm",
    "xlm-roberta",
    "efficientnet",
}

_SEQ2SEQ_TYPES = {
    "bart",
    "longt5",
    "mt5",
    "mbart",
    "pegasus",
    "t5",
}

_DIFFUSION_TYPES = {
    "flux",
    "stable-diffusion",
    "stable_diffusion",
    "stable-diffusion-xl",
    "stable_diffusion_xl",
    "unet-2d-condition",
}

# Video diffusion model_type strings (from config.json when available).
# Most video pipelines are diffusers-only and detected via model_index.json.
_VIDEO_DIFFUSION_TYPES = {
    "cogvideox",
    "unet-spatio-temporal-condition",
    "unet_spatio_temporal_condition",
}

# Pipeline class name substrings that indicate a video diffusion pipeline.
_VIDEO_PIPELINE_KEYWORDS = (
    "cogvideox",
    "animatediff",
    "stablevideo",
    "stablevideodiffusion",
    "img2vid",
    "imagetovideo",
    "texttovideo",
    "i2vgen",
    "videocrafter",
    "modelscope",
    "zeroscope",
)

_AUDIO_TYPES = {
    "hubert",
    "wav2vec2",
    "wavlm",
    "whisper",
}

_MULTIMODAL_TYPES = {
    "blip",
    "blip-2",
    "clip",
    "clipseg",
    "siglip",
}


def detect_family(model_id: str, task_override: Optional[str] = None) -> str:
    """
    Return the strategy family for model_id.

    Args:
        model_id: HuggingFace model tag (e.g. "bert-base-uncased")
        task_override: If provided, maps standard HF task names to family names
                       and skips the config fetch.

    Returns:
        One of: "llm", "encoder", "seq2seq", "diffusion", "audio"
    """
    if task_override:
        return _task_to_family(task_override)

    # Try transformers first.  Diffusers pipelines have model_index.json
    # instead of config.json with a model_type field — fall back to that.
    model_type = ""
    last_error: Optional[Exception] = None
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
        model_type = getattr(cfg, "model_type", "").lower()
    except Exception as e:
        last_error = e

    if not model_type:
        try:
            import json

            from huggingface_hub import hf_hub_download

            path = hf_hub_download(model_id, "model_index.json")
            with open(path) as f:
                idx = json.load(f)
            cls_name = idx.get("_class_name", "").lower()
            if any(k in cls_name for k in _VIDEO_PIPELINE_KEYWORDS):
                return "video_diffusion"
            if "pipeline" in cls_name or any(
                k in cls_name for k in ("diffusion", "flux")
            ):
                return "diffusion"
        except Exception as e:
            last_error = e

    if not model_type:
        msg = f"Could not determine family for '{model_id}'."
        # Detect gated / authentication errors and surface them clearly.
        err_str = str(last_error) if last_error else ""
        if any(
            k in err_str
            for k in ("Repository Not Found", "401", "gated", "authenticated", "access")
        ):
            msg += (
                "\n\nThis model may be gated or private.  Run "
                "`hf auth login` (or `huggingface-cli login`) and accept the "
                "model's license on https://huggingface.co/" + model_id
            )
        msg += "\n\nHint: pass --task to specify the model family explicitly."
        raise ValueError(msg)

    return _model_type_to_family(model_type, model_id)


def _model_type_to_family(model_type: str, model_id: str) -> str:
    if model_type in _LLM_TYPES:
        return "llm"
    if model_type in _ENCODER_TYPES:
        return "encoder"
    if model_type in _SEQ2SEQ_TYPES:
        return "seq2seq"
    if model_type in _VIDEO_DIFFUSION_TYPES:
        return "video_diffusion"
    if model_type in _DIFFUSION_TYPES:
        return "diffusion"
    if model_type in _AUDIO_TYPES:
        return "audio"
    if model_type in _MULTIMODAL_TYPES:
        return "multimodal"

    # Heuristics for types not yet in the explicit lists.
    lower_id = model_id.lower()
    if any(k in lower_id for k in _VIDEO_PIPELINE_KEYWORDS):
        return "video_diffusion"
    if any(k in lower_id for k in ("diffusion", "flux", "sd-", "sdxl")):
        return "diffusion"
    if any(k in lower_id for k in ("whisper", "wav2vec", "asr")):
        return "audio"
    if any(k in lower_id for k in ("llama", "gpt", "mistral", "qwen", "gemma")):
        return "llm"
    if any(k in lower_id for k in ("bert", "vit", "resnet")):
        return "encoder"

    raise ValueError(
        f"Cannot determine model family for model_type='{model_type}' "
        f"(model='{model_id}').\n"
        "Hint: pass --task text-generation|text-classification|"
        "image-generation|video-generation|automatic-speech-recognition "
        "to specify explicitly."
    )


# HF pipeline task names → family
_TASK_MAP = {
    "text-generation": "llm",
    "text2text-generation": "seq2seq",
    "text-classification": "encoder",
    "token-classification": "encoder",
    "fill-mask": "encoder",
    "feature-extraction": "encoder",
    "image-classification": "encoder",
    "image-segmentation": "encoder",
    "object-detection": "encoder",
    "zero-shot-image-classification": "multimodal",
    "image-generation": "diffusion",
    "image-to-image": "diffusion",
    "text-to-video": "video_diffusion",
    "image-to-video": "video_diffusion",
    "video-generation": "video_diffusion",
    "automatic-speech-recognition": "audio",
    "audio-classification": "audio",
}


def _task_to_family(task: str) -> str:
    family = _TASK_MAP.get(task.lower())
    if family is None:
        raise ValueError(
            f"Unknown task '{task}'. Valid tasks: {sorted(_TASK_MAP.keys())}"
        )
    return family
