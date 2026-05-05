"""
Auto-detect model family from a HuggingFace model tag.

Uses AutoConfig (a single config-file fetch, no weights downloaded) to get
model_type, then maps it to one of the supported strategy names:
  "llm"            – decoder-only causal LM (GPT, LLaMA, Qwen, Gemma, ...)
  "vlm"            – vision-language model (LLaVA, PaliGemma, Qwen2-VL, ...)
  "vla"            – vision-language-action model for robotics (OpenVLA, SpatialVLA, ...)
  "encoder"        – encoder-only / sequence classification (BERT, RoBERTa, ViT, ...)
  "seq2seq"        – encoder-decoder (T5, BART, MarianMT, ...)
  "detection"      – object detection / segmentation (DETR, RT-DETR, SAM, ...)
  "diffusion"      – image diffusion pipeline (stable-diffusion, FLUX, ...)
  "video_diffusion"– video diffusion pipeline (CogVideoX, AnimateDiff, SVD, ...)
  "audio"          – speech / ASR (Whisper, wav2vec2, ...)
  "multimodal"     – dual-encoder vision-text models (CLIP, SigLIP, ...)

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
    "cohere",  # Command-R
    "cohere2",  # Command-R+
    "deepseek",  # DeepSeek V1
    "deepseek_v2",  # DeepSeek V2 / V3 / R1
    "deepseek_v3",
    "dbrx",  # DBRX
    "exaone",  # EXAONE (LG AI)
    "falcon",
    "gemma",
    "gemma2",
    "gemma3",
    "gpt2",
    "gpt_bigcode",
    "gpt_neo",
    "gpt_neox",
    "gptj",
    "internlm",  # InternLM
    "internlm2",  # InternLM2
    "llama",
    "mistral",
    "mixtral",
    "mpt",
    "nemotron",  # Nemotron
    "olmo",  # OLMo
    "olmo2",  # OLMo2
    "opt",
    "persimmon",  # Persimmon
    "phi",
    "phi3",
    "qwen2",
    "qwen2_moe",
    "starcoder2",
    "stablelm",
}

_ENCODER_TYPES = {
    "albert",
    "beit",  # BEiT
    "bert",
    "bit",  # BiT (Big Transfer)
    "camembert",
    "convnext",
    "data2vec_vision",  # Data2Vec Vision
    "deberta",
    "deberta-v2",
    "deit",  # DeiT
    "depth_anything",  # Depth Anything v2
    "dinov2",  # DINOv2
    "distilbert",
    "dpt",  # Dense Prediction Transformer (Depth Anything v1)
    "efficientnet",
    "electra",
    "glpn",  # GLPN depth estimation
    "levit",  # LeViT
    "mobilenet_v2",
    "mobilevit",  # MobileViT
    "mobilevitv2",  # MobileViTv2
    "poolformer",  # PoolFormer / MetaFormer
    "regnet",  # RegNet
    "resnet",
    "roberta",
    "segformer",  # SegFormer
    "swin",
    "vit",
    "xlm",
    "xlm-roberta",
}

_SEQ2SEQ_TYPES = {
    "bart",
    "fsmt",  # Fairseq Machine Translation
    "longt5",
    "m2m_100",  # M2M-100 multilingual translation
    "marian",  # MarianMT (1000+ translation models on HF)
    "mbart",
    "mt5",
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
    "align",  # ALIGN
    "blip",
    "blip-2",
    "clip",
    "clipseg",
    "flava",  # FLAVA
    "groupvit",  # GroupViT
    "siglip",
    "x_clip",  # X-CLIP (video-text)
}

# Vision-language-action models: image + instruction → robot action tokens.
# Mostly VLM backbones with an action vocabulary; require trust_remote_code.
_VLA_TYPES = {
    "openvla",  # OpenVLA (LLaMA2 + DINOv2 + SigLIP)
    "prismatic",  # Prismatic VLMs (OpenVLA's base architecture)
    "spatialvla",  # SpatialVLA (spatial reasoning + actions)
    "tinyvla",  # TinyVLA
    "pi0",  # Physical Intelligence Pi-0
}

# Vision-language models: image(s) + text → text generation.
# These require the VLM strategy (torch.compile path; export not supported).
_VLM_TYPES = {
    "aria",  # Aria
    "chameleon",  # Chameleon (Meta)
    "florence2",  # Florence-2 (Microsoft)
    "idefics2",  # Idefics2
    "idefics3",  # Idefics3 / SmolVLM
    "llava",  # LLaVA 1.5
    "llava_next",  # LLaVA-NeXT (1.6)
    "llava_next_video",  # LLaVA-NeXT-Video
    "llava_onevision",  # LLaVA-OneVision
    "paligemma",  # PaliGemma
    "phi3_v",  # Phi-3-Vision
    "qwen2_vl",  # Qwen2-VL
}

# Object detection / segmentation models.
_DETECTION_TYPES = {
    "conditional_detr",  # Conditional DETR
    "dab_detr",  # DAB-DETR
    "deta",  # DETA
    "detr",  # DETR
    "grounding_dino",  # GroundingDINO (text-conditioned detection)
    "owlv2",  # OWLv2
    "owlvit",  # OWL-ViT
    "rt_detr",  # RT-DETR
    "rt_detr_resnet",  # RT-DETRv2
    "sam",  # Segment Anything Model (image encoder compiled)
    "table_transformer",  # Table Transformer
    "yolos",  # YOLOS
}


def detect_family(model_id: str, task_override: Optional[str] = None) -> str:
    """
    Return the strategy family for model_id.

    Args:
        model_id: HuggingFace model tag (e.g. "bert-base-uncased")
        task_override: If provided, maps standard HF task names to family names
                       and skips the config fetch.

    Returns:
        One of: "llm", "vlm", "encoder", "seq2seq", "detection",
                "diffusion", "video_diffusion", "audio", "multimodal"
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
    if model_type in _VLA_TYPES:
        return "vla"
    if model_type in _VLM_TYPES:
        return "vlm"
    if model_type in _ENCODER_TYPES:
        return "encoder"
    if model_type in _SEQ2SEQ_TYPES:
        return "seq2seq"
    if model_type in _DETECTION_TYPES:
        return "detection"
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
    if any(
        k in lower_id
        for k in ("openvla", "spatialvla", "tinyvla", "prismatic-vla", "/vla-", "/pi0")
    ):
        return "vla"
    if any(
        k in lower_id
        for k in (
            "llava",
            "paligemma",
            "qwen2-vl",
            "qwen2vl",
            "smolvlm",
            "idefics",
            "moondream",
        )
    ):
        return "vlm"
    if any(k in lower_id for k in ("detr", "rtdetr", "sam-vit", "owlvit", "owlv2")):
        return "detection"
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
        "visual-question-answering|object-detection|"
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
    "image-segmentation": "detection",
    "object-detection": "detection",
    "zero-shot-object-detection": "detection",
    "zero-shot-image-classification": "multimodal",
    "visual-question-answering": "vlm",
    "image-to-text": "vlm",
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
