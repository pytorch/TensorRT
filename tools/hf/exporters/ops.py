"""Named Edge-LLM runtime ops. Engines are values; behavior is the operator."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from pathlib import Path
from typing import Any

import torch

from .executorch.serialization import EdgeComponentMetadata

# One process-wide table so pytest dual-imports of this module still share
# execute_engine state with record_engine.
_REGISTRY: dict[str, Any] = sys.modules.setdefault(
    "_edge_llm_engine_registry",
    {"meta": {}, "modules": {}},
)
_ENGINE_META: dict[str, dict[str, Any]] = _REGISTRY["meta"]
_COMPILED_MODULES: dict[str, torch.nn.Module] = _REGISTRY["modules"]
_ENGINE_LOAD_LOCK: threading.Lock = _REGISTRY.setdefault("lock", threading.Lock())
_EMBEDDED_MODULES: dict[str, torch.nn.Module] = _REGISTRY.setdefault(
    "embedded_modules", {}
)


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


def _load_serialized_engine(engine_path: str, component: str) -> torch.nn.Module:
    from torch_tensorrt.dynamo.runtime import TorchTensorRTModule

    engine_dir = Path(engine_path)
    config_path = engine_dir / "config.json"
    try:
        config = json.loads(config_path.read_text())
        engine_file = config["engine_file"]
        input_names = config["input_names"]
        output_names = config["output_names"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot load TensorRT engine metadata for {component!r} from "
            f"{config_path}"
        ) from exc

    serialized_path = engine_dir / engine_file
    try:
        serialized_engine = serialized_path.read_bytes()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Serialized TensorRT engine for {component!r} was not found at "
            f"{serialized_path}"
        ) from exc

    return TorchTensorRTModule(
        serialized_engine=serialized_engine,
        input_binding_names=list(input_names),
        output_binding_names=list(output_names),
        name=component,
    )


def _get_engine(engine_path: str, component: str) -> torch.nn.Module:
    compiled = _COMPILED_MODULES.get(engine_path)
    if compiled is not None:
        return compiled

    # Engine deserialization is expensive and must only happen once per path.
    with _ENGINE_LOAD_LOCK:
        compiled = _COMPILED_MODULES.get(engine_path)
        if compiled is None:
            compiled = _load_serialized_engine(engine_path, component)
            _COMPILED_MODULES[engine_path] = compiled
        return compiled


@torch.library.custom_op("edge_llm::execute_engine", mutates_args=())  # type: ignore[misc]
def execute_engine(
    engine_path: str, component: str, tensors: list[torch.Tensor]
) -> list[torch.Tensor]:
    compiled = _get_engine(engine_path, component)
    out = compiled(*tensors)
    return list(_as_tuple(out))


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


def _tensor_bytes(value: torch.Tensor) -> bytes:
    data = value.detach().cpu().contiguous().view(torch.uint8)
    return bytes(memoryview(data.numpy()))


def _get_embedded_engine(
    trt_blob: torch.Tensor, metadata: EdgeComponentMetadata
) -> torch.nn.Module:
    from torch_tensorrt.dynamo.runtime import TorchTensorRTModule
    from torch_tensorrt.executorch.serialization import deserialize_engine

    blob_bytes = _tensor_bytes(trt_blob)
    cache_key = hashlib.sha256(
        blob_bytes + metadata.to_json().encode("utf-8")
    ).hexdigest()
    compiled = _EMBEDDED_MODULES.get(cache_key)
    if compiled is not None:
        return compiled

    with _ENGINE_LOAD_LOCK:
        compiled = _EMBEDDED_MODULES.get(cache_key)
        if compiled is None:
            engine_bytes, trt_metadata = deserialize_engine(blob_bytes)
            input_names = [
                binding.name for binding in trt_metadata.io_bindings if binding.is_input
            ]
            output_names = [
                binding.name
                for binding in trt_metadata.io_bindings
                if not binding.is_input
            ]
            compiled = TorchTensorRTModule(
                serialized_engine=engine_bytes,
                input_binding_names=input_names,
                output_binding_names=output_names,
                name=f"edge_llm_{metadata.component}",
            )
            _EMBEDDED_MODULES[cache_key] = compiled
        return compiled


def _vision_metadata(metadata_json: str) -> EdgeComponentMetadata:
    metadata = EdgeComponentMetadata.from_json(metadata_json)
    if metadata.component != "vision" or metadata.runner != "vit":
        raise ValueError(
            "edge_llm::vision_tower requires component='vision' and runner='vit', "
            f"got component={metadata.component!r}, runner={metadata.runner!r}"
        )
    return metadata


@torch.library.custom_op("edge_llm::vision_tower", mutates_args=())  # type: ignore[misc]
def vision_tower(
    tensors: list[torch.Tensor],
    trt_blob: torch.Tensor,
    metadata_json: str,
) -> list[torch.Tensor]:
    metadata = _vision_metadata(metadata_json)
    compiled = _get_embedded_engine(trt_blob, metadata)
    return list(_as_tuple(compiled(*tensors)))


@vision_tower.register_fake  # type: ignore[misc]
def _(
    tensors: list[torch.Tensor],
    trt_blob: torch.Tensor,
    metadata_json: str,
) -> list[torch.Tensor]:
    del trt_blob
    metadata = _vision_metadata(metadata_json)
    device = tensors[0].device if tensors else torch.device("cpu")
    outputs = []
    for output in metadata.outputs:
        dtype = getattr(torch, output.dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unsupported Edge output dtype {output.dtype!r}")
        outputs.append(torch.empty(output.shape, dtype=dtype, device=device))
    return outputs


def call_vision_tower(
    trt_blob: torch.Tensor,
    metadata_json: str,
    *tensors: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Call the embedded PI0.5-compatible Edge-LLM vision component."""
    out = torch.ops.edge_llm.vision_tower.default(
        list(tensors), trt_blob, metadata_json
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
