"""Load compressed ModelOpt FP8 safetensors without ModelOpt runtime wrappers."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from safetensors import safe_open

from .fp8_linear import FP8CheckpointLinear

_WEIGHT_SCALE_SUFFIX = ".weight_quantizer._scale"
_INPUT_AMAX_SUFFIX = ".input_quantizer._amax"
_UNSUPPORTED_ATTENTION_QUANTIZERS = (
    ".q_bmm_quantizer.",
    ".k_bmm_quantizer.",
    ".v_bmm_quantizer.",
    ".softmax_quantizer.",
    ".bmm2_output_quantizer.",
)


class ModelOptCheckpoint:
    """Indexed, lazy reader for sharded ModelOpt safetensors."""

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self.root = Path(checkpoint_dir)
        index_path = self.root / "model.safetensors.index.json"
        single_path = self.root / "model.safetensors"

        if index_path.is_file():
            index = json.loads(index_path.read_text())
            self.weight_map: dict[str, str] = dict(index["weight_map"])
        elif single_path.is_file():
            with safe_open(single_path, framework="pt", device="cpu") as shard:
                self.weight_map = {key: single_path.name for key in shard.keys()}
        else:
            raise FileNotFoundError(
                f"No model.safetensors or sharded index found in {self.root}"
            )

    def __contains__(self, key: str) -> bool:
        return key in self.weight_map

    def keys(self) -> tuple[str, ...]:
        return tuple(self.weight_map)

    def get_tensor(self, key: str) -> torch.Tensor:
        try:
            shard_name = self.weight_map[key]
        except KeyError as exc:
            raise KeyError(f"{key!r} is not present in {self.root}") from exc
        with safe_open(
            self.root / shard_name,
            framework="pt",
            device="cpu",
        ) as shard:
            return shard.get_tensor(key)

    def maybe_tensor(self, key: str) -> torch.Tensor | None:
        return self.get_tensor(key) if key in self else None

    def quantized_linear_names(self) -> tuple[str, ...]:
        names = []
        for key in self.weight_map:
            if not key.endswith(_WEIGHT_SCALE_SUFFIX):
                continue
            name = key[: -len(_WEIGHT_SCALE_SUFFIX)]
            required = (f"{name}.weight", f"{name}{_INPUT_AMAX_SUFFIX}")
            if all(item in self for item in required):
                names.append(name)
        return tuple(sorted(names))

    def fp8_linear(
        self,
        name: str,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> FP8CheckpointLinear:
        return FP8CheckpointLinear(
            self.get_tensor(f"{name}.weight"),
            self.get_tensor(f"{name}{_WEIGHT_SCALE_SUFFIX}"),
            self.get_tensor(f"{name}{_INPUT_AMAX_SUFFIX}"),
            self.maybe_tensor(f"{name}.bias"),
            device=device,
            dtype=dtype,
        )

    def unsupported_attention_quantizers(self) -> tuple[str, ...]:
        return tuple(
            key
            for key in self.weight_map
            if any(marker in key for marker in _UNSUPPORTED_ATTENTION_QUANTIZERS)
        )


def is_modelopt_fp8_checkpoint(checkpoint_dir: str | Path) -> bool:
    """Return whether a local checkpoint contains compressed FP8 metadata."""
    root = Path(checkpoint_dir)
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file() and not (root / "model.safetensors").is_file():
        return False
    try:
        checkpoint = ModelOptCheckpoint(root)
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return False
    return bool(checkpoint.quantized_linear_names())


def _set_submodule(root: nn.Module, path: str, value: nn.Module) -> None:
    parent_path, _, child_name = path.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, child_name, value)


def _owner_and_name(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parent_path, _, name = path.rpartition(".")
    return (root.get_submodule(parent_path) if parent_path else root), name


def _checkpoint_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.dtype:
    if tensor.is_floating_point() and tensor.dtype != torch.float8_e4m3fn:
        return dtype
    return tensor.dtype


def _materialize_parameters(
    model: nn.Module,
    checkpoint: ModelOptCheckpoint,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> list[str]:
    missing: list[str] = []
    for name, parameter in list(model.named_parameters(remove_duplicate=False)):
        if name not in checkpoint:
            if parameter.device.type == "meta":
                missing.append(name)
            continue
        value = checkpoint.get_tensor(name)
        value = value.to(
            device=device,
            dtype=_checkpoint_dtype(value, dtype),
        )
        owner, child_name = _owner_and_name(model, name)
        owner._parameters[child_name] = nn.Parameter(
            value,
            requires_grad=parameter.requires_grad,
        )
    return missing


def _materialize_buffers(
    model: nn.Module,
    checkpoint: ModelOptCheckpoint,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> list[str]:
    missing: list[str] = []
    for name, buffer in list(model.named_buffers(remove_duplicate=False)):
        if name not in checkpoint:
            if buffer.device.type == "meta":
                missing.append(name)
            continue
        value = checkpoint.get_tensor(name)
        value = value.to(
            device=device,
            dtype=_checkpoint_dtype(value, dtype),
        )
        owner, child_name = _owner_and_name(model, name)
        owner._buffers[child_name] = value
    return missing


def load_modelopt_fp8_model(
    model: nn.Module,
    checkpoint_dir: str | Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Materialize a meta model from a compressed ModelOpt FP8 checkpoint.

    Quantized ``nn.Linear`` modules become :class:`FP8CheckpointLinear`.
    Remaining parameters and persistent buffers are loaded at their checkpoint
    precision, with ordinary floating-point tensors cast to ``dtype``.
    """
    checkpoint = ModelOptCheckpoint(checkpoint_dir)
    modules = dict(model.named_modules())
    replacements: list[str] = []

    for name in checkpoint.quantized_linear_names():
        module = modules.get(name)
        if module is None:
            raise KeyError(f"Quantized checkpoint module {name!r} is not in the model")
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"Quantized checkpoint module {name!r} is {type(module).__name__}, "
                "expected nn.Linear"
            )
        _set_submodule(
            model,
            name,
            checkpoint.fp8_linear(name, device=device, dtype=dtype),
        )
        replacements.append(name)

    missing = _materialize_parameters(
        model,
        checkpoint,
        device=device,
        dtype=dtype,
    )
    missing.extend(
        _materialize_buffers(
            model,
            checkpoint,
            device=device,
            dtype=dtype,
        )
    )
    if missing:
        raise RuntimeError(
            "Compressed checkpoint did not materialize meta tensors: "
            + ", ".join(sorted(missing)[:20])
        )

    unsupported = checkpoint.unsupported_attention_quantizers()
    if unsupported:
        warnings.warn(
            "ModelOpt attention-core quantizers are present but the Edge "
            "AttentionPlugin currently executes attention BMM/softmax in FP16. "
            "FP8 projection and MLP linears remain enabled. Unsupported tensors: "
            f"{len(unsupported)}",
            stacklevel=2,
        )

    return {
        "fp8_linears": len(replacements),
        "unsupported_attention_quantizers": len(unsupported),
    }
