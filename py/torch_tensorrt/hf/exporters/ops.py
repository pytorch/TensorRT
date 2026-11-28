"""Named Edge-LLM runtime ops. Engines are values; behavior is the operator."""

from __future__ import annotations

from typing import Any

import torch

_ENGINE_META: dict[str, dict[str, Any]] = {}
_COMPILED_MODULES: dict[str, torch.nn.Module] = {}


def record_engine(
    path: str,
    *,
    component: str,
    input_names: list[str],
    outputs: tuple[torch.Tensor, ...],
    module: torch.nn.Module | None = None,
) -> None:
    _ENGINE_META[path] = {
        "component": component,
        "input_names": list(input_names),
        "output_shapes": [tuple(t.shape) for t in outputs],
        "output_dtypes": [t.dtype for t in outputs],
    }
    if module is not None:
        _COMPILED_MODULES[path] = module


def _as_tuple(value: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(value, tuple):
        return tuple(value)
    if isinstance(value, list):
        return tuple(value)
    return (value,)


@torch.library.custom_op("edge_llm::execute_engine", mutates_args=())  # type: ignore[misc]
def execute_engine(
    engine_path: str, component: str, tensors: list[torch.Tensor]
) -> list[torch.Tensor]:
    compiled = _COMPILED_MODULES.get(engine_path)
    if compiled is not None:
        out = compiled(*tensors)
        return list(_as_tuple(out))
    raise RuntimeError(
        f"No in-process module for engine {engine_path!r} ({component}). "
        "Compile first, or load a serialized engine at this path."
    )


@execute_engine.register_fake  # type: ignore[misc]
def _(
    engine_path: str, component: str, tensors: list[torch.Tensor]
) -> list[torch.Tensor]:
    meta = _ENGINE_META.get(engine_path)
    device = tensors[0].device if tensors else torch.device("cpu")
    if meta is None:
        return [torch.empty_like(t) for t in tensors]
    return [
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in zip(meta["output_shapes"], meta["output_dtypes"])
    ]


def call_engine(
    engine_path: str, component: str, *tensors: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Python helper so specs can pass ``*tensors`` instead of a list."""
    out = torch.ops.edge_llm.execute_engine.default(
        engine_path, component, list(tensors)
    )
    return tuple(out)


@torch.library.custom_op("edge_llm::fuse_prefix", mutates_args=())  # type: ignore[misc]
def fuse_prefix(
    vision_tokens: torch.Tensor,
    lang_embeds: torch.Tensor,
    compact_index: torch.Tensor,
) -> torch.Tensor:
    hidden = lang_embeds.shape[-1]
    batch = lang_embeds.shape[0]
    vis = vision_tokens
    # Vision engines may emit [B, S, H] (HF image features) or flattened [N, H].
    if vis.ndim == 3:
        vis = vis.reshape(-1, vis.shape[-1])
    if vis.ndim == 2:
        vis = vis.reshape(batch, -1, hidden)
    embs = torch.cat([vis, lang_embeds], dim=1)
    index = compact_index.to(dtype=torch.long)
    return torch.gather(embs, 1, index.unsqueeze(-1).expand(-1, -1, hidden))


@fuse_prefix.register_fake  # type: ignore[misc]
def _(
    vision_tokens: torch.Tensor,
    lang_embeds: torch.Tensor,
    compact_index: torch.Tensor,
) -> torch.Tensor:
    batch, compact_len = compact_index.shape
    return torch.empty(
        batch,
        compact_len,
        lang_embeds.shape[-1],
        dtype=lang_embeds.dtype,
        device=lang_embeds.device,
    )


@torch.library.custom_op("edge_llm::scatter_image_tokens", mutates_args=())  # type: ignore[misc]
def scatter_image_tokens(
    vision_tokens: torch.Tensor,
    lang_embeds: torch.Tensor,
    image_token_mask: torch.Tensor,
) -> torch.Tensor:
    hidden = lang_embeds.shape[-1]
    vis = vision_tokens.reshape(-1, hidden).to(dtype=lang_embeds.dtype)
    out = lang_embeds.clone()
    flat = out.reshape(-1, hidden)
    mask = image_token_mask.reshape(-1).to(dtype=torch.bool)
    n = int(mask.sum().item())
    if n:
        flat[mask] = vis[:n]
    return flat.reshape_as(lang_embeds)


@scatter_image_tokens.register_fake  # type: ignore[misc]
def _(
    vision_tokens: torch.Tensor,
    lang_embeds: torch.Tensor,
    image_token_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(lang_embeds)
