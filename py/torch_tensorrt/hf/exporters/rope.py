from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def language_head_dim(config) -> int:
    return int(
        getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
    )


def config_to_dict(config) -> dict:
    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    raise TypeError(
        f"Expected a HuggingFace-style config with to_dict(), got {type(config)!r}"
    )


def select_rope_parameters(config_dict: dict) -> dict:
    rope_parameters = config_dict.get("rope_parameters")
    if rope_parameters is None:
        rope_parameters = config_dict.get("rope_scaling")
    if not isinstance(rope_parameters, dict):
        return {}

    layer_types = config_dict.get("layer_types")
    if layer_types and set(rope_parameters.keys()).issubset(set(layer_types)):
        full_attention = rope_parameters.get("full_attention")
        if isinstance(full_attention, dict):
            return dict(full_attention)
        for layer_type in layer_types:
            layer_params = rope_parameters.get(layer_type)
            if isinstance(layer_params, dict):
                return dict(layer_params)
        return {}

    full_attention = rope_parameters.get("full_attention")
    if isinstance(full_attention, dict):
        return dict(full_attention)

    return dict(rope_parameters)


def normalize_rope_scaling(rope_params: dict) -> dict:
    rope_scaling = dict(rope_params)
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if rope_type is not None:
        rope_scaling.setdefault("rope_type", rope_type)
        rope_scaling.setdefault("type", rope_type)
    return rope_scaling


def export_rope_fields(config_dict: dict) -> dict:
    """RoPE fields for LLMEngineRunner::collectRopeConfig / initializeRopeCosSinCache."""
    rope_fields: dict = {}
    rope_params = select_rope_parameters(config_dict)

    if "rope_theta" in config_dict:
        rope_fields["rope_theta"] = config_dict["rope_theta"]
    elif "rope_theta" in rope_params:
        rope_fields["rope_theta"] = rope_params["rope_theta"]
    else:
        raise KeyError("rope_theta not found in language model config")

    if rope_params:
        rope_fields["rope_scaling"] = normalize_rope_scaling(rope_params)
    else:
        rope_fields["rope_scaling"] = None

    if rope_fields["rope_scaling"] is not None:
        rope_type = rope_fields["rope_scaling"].get("rope_type")
        if rope_type == "longrope":
            original_max = rope_fields["rope_scaling"].get(
                "original_max_position_embeddings",
                config_dict.get("original_max_position_embeddings"),
            )
            if original_max is None:
                raise KeyError(
                    "original_max_position_embeddings required for longrope scaling"
                )
            rope_fields["original_max_position_embeddings"] = original_max

    if "partial_rotary_factor" in config_dict:
        rope_fields["partial_rotary_factor"] = config_dict["partial_rotary_factor"]
    elif "partial_rotary_factor" in rope_params:
        rope_fields["partial_rotary_factor"] = rope_params["partial_rotary_factor"]
    else:
        rope_fields["partial_rotary_factor"] = 1.0

    return rope_fields


def rotary_dim_from_config(config) -> int:
    head_dim = language_head_dim(config)
    partial = float(getattr(config, "partial_rotary_factor", 1.0) or 1.0)
    rotary_dim = int(head_dim * partial)
    if rotary_dim <= 0 or rotary_dim > head_dim:
        rotary_dim = head_dim
    return rotary_dim


def _resolve_rotary_emb(language_model: nn.Module) -> nn.Module | None:
    for module in (language_model, getattr(language_model, "model", None)):
        if module is None:
            continue
        rotary_emb = getattr(module, "rotary_emb", None)
        if rotary_emb is not None:
            return rotary_emb
    return None


def _plugin_rope_layout(
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    max_seq_len: int,
    rotary_dim: int,
) -> torch.Tensor:
    """Pack cos/sin into AttentionPlugin layout: [:, :, :half]=cos, [:, :, half:]=sin."""
    half = rotary_dim // 2
    cos_half = cos[..., :half].float()
    sin_half = sin[..., :half].float()
    cache = torch.cat(
        [cos_half[:, :max_seq_len], sin_half[:, :max_seq_len]],
        dim=-1,
    )
    if cache.shape[0] != 1:
        cache = cache[:1]
    return cache.contiguous()


def make_normal_rope_rotary_cos_sin(
    max_seq_len: int,
    rotary_dim: int,
    *,
    rope_theta: float,
    rotary_scale: float = 1.0,
    device: torch.device,
) -> torch.Tensor:
    """Build RoPE cache using the same formula as Edge-LLM initializeNormalRopeCosSin."""
    half = rotary_dim // 2
    zid = torch.arange(half, device=device, dtype=torch.float32)
    inv_denominator = float(rope_theta) ** (2 * zid / float(rotary_dim))
    positions = torch.arange(max_seq_len, device=device, dtype=torch.float32).unsqueeze(
        1
    )
    angles = positions * float(rotary_scale) / inv_denominator.unsqueeze(0)
    cos = angles.cos()
    sin = angles.sin()
    return torch.cat([cos, sin], dim=-1).unsqueeze(0).to(dtype=torch.float32)


def make_rope_rotary_cos_sin_from_config(
    config,
    max_seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Build RoPE cache from HF config fields (matches LLMEngineRunner default/dynamic RoPE)."""
    config_dict = config_to_dict(config)
    rope_params = select_rope_parameters(config_dict)
    rope_type = None
    if rope_params:
        rope_type = rope_params.get("rope_type", rope_params.get("type"))

    if rope_type == "longrope":
        raise NotImplementedError(
            "longrope config cache is not implemented; pass language_model= to "
            "make_rope_rotary_cos_sin() instead"
        )
    if rope_type not in (None, "default", "dynamic", "llama3"):
        raise NotImplementedError(
            f"RoPE type {rope_type!r} requires language_model= and position_ids="
        )

    rotary_dim = rotary_dim_from_config(config)
    rope_theta = export_rope_fields(config_dict)["rope_theta"]
    return make_normal_rope_rotary_cos_sin(
        max_seq_len,
        rotary_dim,
        rope_theta=float(rope_theta),
        rotary_scale=1.0,
        device=device,
    )


@torch.no_grad()
def make_rope_rotary_cos_sin_from_model(
    language_model: nn.Module,
    config,
    max_seq_len: int,
    device: torch.device,
    *,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build plugin RoPE cache from the model's rotary_emb (best eager parity)."""
    rotary_emb = _resolve_rotary_emb(language_model)
    if rotary_emb is None:
        raise AttributeError("language_model has no rotary_emb")

    rotary_dim = rotary_dim_from_config(config)
    if position_ids is None:
        position_ids = torch.arange(
            max_seq_len, device=device, dtype=torch.long
        ).unsqueeze(0)
    else:
        position_ids = position_ids.to(device=device)

    seq_len = int(position_ids.shape[-1])
    dummy = torch.ones(
        int(position_ids.shape[0] if position_ids.ndim >= 2 else 1),
        seq_len,
        1,
        device=device,
        dtype=torch.float16,
    )
    cos, sin = rotary_emb(dummy, position_ids)
    return _plugin_rope_layout(
        cos,
        sin,
        max_seq_len=max_seq_len,
        rotary_dim=rotary_dim,
    )


def config_is_nope(config) -> bool:
    """True when attention has no positional encoding (Nemotron-H / Nano)."""
    if getattr(config, "use_rope", None) is False:
        return True
    model_type = str(getattr(config, "model_type", "") or "").lower()
    return model_type.startswith("nemotron_h")


def make_nope_rotary_cos_sin(
    max_seq_len: int,
    rotary_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Identity cos/sin cache so AttentionPlugin is a RoPE pass-through.

    Matches Edge-LLM ``initializeNopeCosSinCache``: first half 1.0 (cos),
    second half 0.0 (sin). Nemotron-H / Nano omit RoPE; position lives in SSM.
    """
    half = int(rotary_dim) // 2
    cos = torch.ones(1, int(max_seq_len), half, dtype=torch.float32, device=device)
    sin = torch.zeros(1, int(max_seq_len), half, dtype=torch.float32, device=device)
    return torch.cat([cos, sin], dim=-1)


@torch.no_grad()
def make_rope_rotary_cos_sin(
    config,
    max_seq_len: int,
    device: torch.device,
    *,
    language_model: nn.Module | None = None,
    position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build real RoPE cache for inference/parity (not export tracing).

    Prefers the model's ``rotary_emb`` when available for eager parity. Falls back
    to config-based generation that matches Edge-LLM ``initializeNormalRopeCosSin``.
    Nemotron-H has no ``rotary_emb`` / ``rope_theta``; uses the NoPE identity cache.
    """
    if config_is_nope(config):
        return make_nope_rotary_cos_sin(
            max_seq_len, rotary_dim_from_config(config), device
        )
    if language_model is not None:
        try:
            return make_rope_rotary_cos_sin_from_model(
                language_model,
                config,
                max_seq_len,
                device,
                position_ids=position_ids,
            )
        except (AttributeError, NotImplementedError, RuntimeError) as exc:
            logger.warning(
                "Building RoPE from model rotary_emb failed (%s); using config kernel",
                exc,
            )
    return make_rope_rotary_cos_sin_from_config(config, max_seq_len, device)


def make_dummy_rope_rotary_cos_sin(
    max_seq_len: int,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Placeholder RoPE cache for export/compile tracing (runtime overwrites values)."""
    return torch.randn(
        1,
        int(max_seq_len),
        int(head_dim),
        dtype=torch.float32,
        device=device,
    )
