from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ...rope import _plugin_rope_layout, language_head_dim

logger = logging.getLogger(__name__)

# Cosmos-Reason2-2B is gated on the Hub; it is a Qwen3-VL-2B fine-tune with the
# same tokenizer, chat template and image processor.
QWEN3_VL_PROCESSOR_FALLBACK = "Qwen/Qwen3-VL-2B-Instruct"


def make_embodiment_id(
    policy: Any,
    state: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    from lerobot.policies.groot.processor_groot import N1_7_EMBODIMENT_MAPPING

    embodiment_tag = getattr(policy.config, "embodiment_tag", "new_embodiment")
    return torch.full(
        (state.shape[0],),
        N1_7_EMBODIMENT_MAPPING.get(embodiment_tag, 0),
        dtype=dtype,
        device=device,
    )


def _groot(model: nn.Module) -> nn.Module:
    if hasattr(model, "_groot_model"):
        return model._groot_model
    backbone = getattr(model, "backbone", None)
    if getattr(backbone, "visual", None) is not None and hasattr(model, "action_head"):
        return model
    raise RuntimeError(
        "GR00T spec expected GrootPolicy or a GR00T N1.7 module with backbone.visual"
    )


def build_n1_7_processor():
    """Qwen3-VL processor for N1.7; falls back to the public base when gated."""
    from lerobot.policies.groot.processor_groot import _build_n1_7_processor

    try:
        return _build_n1_7_processor()
    except OSError as exc:
        logger.warning(
            "Cannot load the Cosmos-Reason2-2B processor (%s); using %s instead",
            exc.__class__.__name__,
            QWEN3_VL_PROCESSOR_FALLBACK,
        )
        return _build_n1_7_processor(QWEN3_VL_PROCESSOR_FALLBACK)


def make_vlm_encode_step(base_model_path: str) -> tuple[Any, bool]:
    """``GrootN17VLMEncodeStep`` configured from the checkpoint ``processor_config.json``.

    The full LeRobot preprocessor also needs per-embodiment statistics, which
    the base checkpoint lacks for ``new_embodiment``; only the VLM step matters here.
    Returns the step and the checkpoint ``formalize_language`` flag.
    """
    from lerobot.policies.groot.configuration_groot import (
        N1_7_DEFAULT_IMAGE_CROP_SIZE,
        N1_7_DEFAULT_IMAGE_TARGET_SIZE,
    )
    from lerobot.policies.groot.processor_groot import GrootN17VLMEncodeStep
    from lerobot.policies.groot.utils import (
        as_int_pair,
        as_optional_float,
        as_optional_int,
        read_json,
    )

    kwargs: dict[str, Any] = {}
    try:
        path = Path(base_model_path) / "processor_config.json"
        if not path.is_file():
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(base_model_path, "processor_config.json"))
        kwargs = read_json(path).get("processor_kwargs", {}) or {}
    except Exception as exc:  # noqa: BLE001 - fall back to LeRobot defaults
        logger.warning("No N1.7 processor_config.json (%s); using defaults", exc)

    step = GrootN17VLMEncodeStep(
        image_crop_size=as_int_pair(kwargs.get("image_crop_size"))
        or list(N1_7_DEFAULT_IMAGE_CROP_SIZE),
        image_target_size=as_int_pair(kwargs.get("image_target_size"))
        or list(N1_7_DEFAULT_IMAGE_TARGET_SIZE),
        shortest_image_edge=as_optional_int(kwargs.get("shortest_image_edge")),
        crop_fraction=as_optional_float(kwargs.get("crop_fraction")),
        use_albumentations=bool(kwargs.get("use_albumentations", False)),
        letter_box_transform=bool(kwargs.get("letter_box_transform", False)),
    )
    step._proc = build_n1_7_processor()
    return step, bool(kwargs.get("formalize_language", True))


def vision_grid_inputs(visual: nn.Module, grid_thw: torch.Tensor) -> dict[str, torch.Tensor]:
    """Precompute the ``grid_thw``-dependent tensors of the Qwen3-VL vision tower.

    ``Qwen3VLVisionModel.forward`` derives these with ``.tolist()`` and
    ``repeat_interleave``, which do not export; the vision engine takes them as inputs.
    """
    from transformers.vision_utils import (
        get_vision_cu_seqlens,
        get_vision_interpolation_indices_and_weights,
        get_vision_position_ids,
    )

    merge = int(visual.spatial_merge_size)
    interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
        grid_thw,
        num_grid_per_side=visual.num_grid_per_side,
        mode=visual.interpolation_mode,
        align_corners=visual.interpolation_align_corners,
        spatial_merge_size=merge,
    )
    cu_seqlens = get_vision_cu_seqlens(grid_thw).to(torch.int32)
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
    return {
        "vision_interp_indices": interp_indices,
        "vision_interp_weights": interp_weights,
        "vision_position_ids": get_vision_position_ids(grid_thw, merge),
        "vision_cu_seqlens": cu_seqlens,
        "vision_max_seqlen_carrier": torch.zeros(
            max_seqlen, dtype=torch.int32, device=grid_thw.device
        ),
    }


@torch.no_grad()
def mrope_rotary_cos_sin(
    language: nn.Module,
    position_ids: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """AttentionPlugin RoPE cache for Qwen3-VL interleaved M-RoPE.

    ``position_ids`` is ``[3, 1, S]`` (T/H/W rows) for the prompt. Row ``j`` of the
    cache rotates token ``j``; rows past the prompt continue text positions from
    ``max(position_ids) + 1``, matching HF decode with ``rope_deltas``.
    """
    seq_len = int(position_ids.shape[-1])
    if seq_len < max_seq_len:
        start = int(position_ids.max().item()) + 1
        tail = torch.arange(
            start,
            start + max_seq_len - seq_len,
            device=position_ids.device,
            dtype=position_ids.dtype,
        )
        position_ids = torch.cat(
            [position_ids[..., :max_seq_len], tail.view(1, 1, -1).expand(3, 1, -1)],
            dim=-1,
        )
    dummy = torch.ones(1, 1, 1, device=position_ids.device, dtype=torch.float32)
    cos, sin = language.rotary_emb(dummy, position_ids[..., :max_seq_len])
    return _plugin_rope_layout(
        cos,
        sin,
        max_seq_len=max_seq_len,
        rotary_dim=language_head_dim(language.config),
    )
