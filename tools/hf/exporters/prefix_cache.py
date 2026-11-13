# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch


def _is_graph_capture_active() -> bool:
    """True when running under Dynamo/export graph capture."""
    try:
        import torch._dynamo as _dynamo

        if _dynamo.is_compiling():
            return True
    except Exception:
        pass
    return False


def maybe_to(
    t: torch.Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Avoid no-op .to() calls in hot paths."""
    if device is not None and t.device != device:
        if dtype is not None and t.dtype != dtype:
            return t.to(device=device, dtype=dtype)
        return t.to(device=device)
    if dtype is not None and t.dtype != dtype:
        return t.to(dtype=dtype)
    return t


class _CacheLayerView:
    __slots__ = ("keys", "values")

    def __init__(
        self, keys: torch.Tensor | None = None, values: torch.Tensor | None = None
    ):
        self.keys = keys
        self.values = values


class _SmolVLAAttnPastEntry:
    """Per-layer KV view for ``smolvlm_with_expert`` dict-style cache access."""

    __slots__ = ("key_states", "value_states")

    def __init__(self, key_states: torch.Tensor, value_states: torch.Tensor):
        self.key_states = key_states
        self.value_states = value_states

    def __getitem__(self, key: str) -> torch.Tensor:
        if key == "key_states":
            return self.key_states
        if key == "value_states":
            return self.value_states
        raise KeyError(key)


class SmolVLAPrefixPastLayers:
    """Stacked prefix KV [L, B, H, S, D] -> SmolVLA ``past_key_values[layer]`` layout."""

    def __init__(self, prefix_k: torch.Tensor, prefix_v: torch.Tensor):
        self._k = prefix_k
        self._v = prefix_v

    def __getitem__(self, layer_idx: int) -> _SmolVLAAttnPastEntry:
        return _SmolVLAAttnPastEntry(
            self._k[layer_idx].transpose(1, 2).contiguous(),
            self._v[layer_idx].transpose(1, 2).contiguous(),
        )


class PrefixKVCache:
    """KV cache backed by stacked tensors with shape [L, B, H, S, D]."""

    __prefix_kv_cache__ = True

    def __init__(self, prefix_k: torch.Tensor, prefix_v: torch.Tensor):
        self._k = prefix_k
        self._v = prefix_v
        self._next_k: torch.Tensor | None = None
        self._next_v: torch.Tensor | None = None
        self._updated_k: list[torch.Tensor | None] = [None] * prefix_k.shape[0]
        self._updated_v: list[torch.Tensor | None] = [None] * prefix_v.shape[0]
        self.layers = [_CacheLayerView() for _ in range(prefix_k.shape[0])]
        self._sync_layer_views()

    @classmethod
    def empty(
        cls,
        *,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "PrefixKVCache":
        empty_k = torch.zeros(
            num_layers,
            batch_size,
            num_kv_heads,
            0,
            head_dim,
            dtype=dtype,
            device=device,
        )
        return cls(empty_k, torch.zeros_like(empty_k))

    @property
    def key_cache(self) -> torch.Tensor:
        return self._k

    @property
    def value_cache(self) -> torch.Tensor:
        return self._v

    def _sync_layer_views(self) -> None:
        if len(self.layers) != self._k.shape[0]:
            self.layers = [_CacheLayerView() for _ in range(self._k.shape[0])]
        for i, layer in enumerate(self.layers):
            layer.keys = self._k[i]
            layer.values = self._v[i]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        if _is_graph_capture_active():
            k = torch.cat([self._k[layer_idx], key_states], dim=-2)
            v = torch.cat([self._v[layer_idx], value_states], dim=-2)
            self._updated_k[layer_idx] = k
            self._updated_v[layer_idx] = v
            return k, v

        prefix_len = self._k.shape[3]
        step_len = key_states.shape[-2]
        total_len = prefix_len + step_len

        if self._next_k is None:
            self._next_k = torch.empty(
                self._k.shape[0],
                self._k.shape[1],
                self._k.shape[2],
                total_len,
                self._k.shape[4],
                dtype=self._k.dtype,
                device=self._k.device,
            )
            self._next_v = torch.empty_like(self._next_k)

        k = self._next_k[layer_idx]
        v = self._next_v[layer_idx]
        k[..., :prefix_len, :] = self._k[layer_idx]
        k[..., prefix_len:, :] = key_states
        v[..., :prefix_len, :] = self._v[layer_idx]
        v[..., prefix_len:, :] = value_states
        self._updated_k[layer_idx] = k
        self._updated_v[layer_idx] = v
        return k, v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        return self._k.shape[3]

    def get_max_cache_shape(self) -> int | None:
        return None

    def get_mask_sizes(
        self, cache_position: torch.Tensor, layer_idx: int = 0
    ) -> tuple[int, int]:
        del layer_idx
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self._k.shape[3] + query_length
        return kv_length, kv_offset

    def get_updated_stacked(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._next_k is not None and self._next_v is not None:
            return self._next_k, self._next_v
        if all(k is not None for k in self._updated_k) and all(
            v is not None for v in self._updated_v
        ):
            return torch.stack(self._updated_k, dim=0), torch.stack(
                self._updated_v, dim=0
            )
        return self._k, self._v

    def update_stacked(
        self, key_cache: torch.Tensor, value_cache: torch.Tensor
    ) -> None:
        self._k = key_cache
        self._v = value_cache
        self._next_k = None
        self._next_v = None
        self._updated_k = [None] * key_cache.shape[0]
        self._updated_v = [None] * value_cache.shape[0]
        self._sync_layer_views()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        beam_idx = beam_idx.to(self._k.device)
        self.update_stacked(
            self._k.index_select(1, beam_idx),
            self._v.index_select(1, beam_idx),
        )

    def batch_repeat_interleave(self, repeats: int) -> None:
        if repeats == 1:
            return
        self.update_stacked(
            self._k.repeat_interleave(repeats, dim=1),
            self._v.repeat_interleave(repeats, dim=1),
        )

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        indices = indices.to(self._k.device)
        self.update_stacked(
            self._k[:, indices, ...],
            self._v[:, indices, ...],
        )

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        if self.get_seq_length() <= max_length:
            return
        self.update_stacked(
            self._k[..., :max_length, :],
            self._v[..., :max_length, :],
        )

    def __len__(self) -> int:
        return int(self._k.shape[0])

    def __iter__(self):
        for i in range(len(self)):
            yield self.layers[i].keys, self.layers[i].values

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.layers[layer_idx].keys, self.layers[layer_idx].values


def stack_prefix_kv_from_cache(
    past_key_values,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a fully-initialized cache object into stacked [L, B, H, S, D] tensors.
    """
    if isinstance(past_key_values, PrefixKVCache):
        return (
            maybe_to(past_key_values.key_cache, device=device, dtype=dtype),
            maybe_to(past_key_values.value_cache, device=device, dtype=dtype),
        )

    if isinstance(past_key_values, tuple) and len(past_key_values) == 2:
        if past_key_values[0] is None or past_key_values[1] is None:
            raise ValueError("Expected concrete stacked KV tensors; got (None, None)")
        return (
            maybe_to(past_key_values[0], device=device, dtype=dtype),
            maybe_to(past_key_values[1], device=device, dtype=dtype),
        )

    if hasattr(past_key_values, "layers"):
        layers = list(past_key_values.layers)
        if len(layers) == 0:
            raise ValueError("Cache has no initialized layers")
        k_tensors = [getattr(layer, "keys", None) for layer in layers]
        v_tensors = [getattr(layer, "values", None) for layer in layers]
        if any((k is None) != (v is None) for k, v in zip(k_tensors, v_tensors)):
            raise ValueError("Inconsistent cache state: one of keys/values is None")
        if any(k is None for k in k_tensors):
            raise ValueError(
                "Cache contains uninitialized layers; cannot infer stacked tensors"
            )
        return (
            torch.stack(
                [maybe_to(k, device=device, dtype=dtype) for k in k_tensors], dim=0
            ),
            torch.stack(
                [maybe_to(v, device=device, dtype=dtype) for v in v_tensors], dim=0
            ),
        )

    raise ValueError("Unsupported cache type for stack_prefix_kv_from_cache")
