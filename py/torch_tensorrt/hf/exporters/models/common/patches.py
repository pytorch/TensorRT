from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def hwc_to_nchw(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"Expected 4D images, got shape {tuple(images.shape)}")
    return images.permute(0, 3, 1, 2).contiguous()


def is_nchw_pixel_values(pixel_values: torch.Tensor) -> bool:
    return (
        pixel_values.ndim == 4
        and pixel_values.shape[1] in (1, 3, 4)
        and pixel_values.shape[-1] not in (1, 3, 4)
    )


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class VisionPatch(nn.Module):
    """Base TRT trace target: probe static shapes once, flatten output to [N, H]."""

    cast_output_to_input_dtype: bool
    output_num_tokens: int
    output_hidden_size: int

    def _finalize_output(
        self, features: torch.Tensor, out_dtype: torch.dtype
    ) -> torch.Tensor:
        if self.cast_output_to_input_dtype and features.dtype != out_dtype:
            features = features.to(out_dtype)
        if features.ndim == 3:
            # [B, S, H] -> [B*S, H]
            return features.reshape(-1, features.shape[-1])
        if features.ndim == 2:
            # already [N, H] (token-pooling encoders)
            return features
        raise ValueError(
            f"Expected 2D or 3D features, got shape {tuple(features.shape)}"
        )


# ---------------------------------------------------------------------------
# Grid vision (PI0.5 / GR00T / SmolVLA VitRunner path)
# ---------------------------------------------------------------------------


class GridVisionPatch(VisionPatch):
    """pixels [B,H,W,C] -> fixed patch grid -> [B*seq_len, lm_hidden]"""

    def __init__(
        self,
        *,
        vision_model: nn.Module,
        projector: nn.Module,
        sample_pixel_values: torch.Tensor,
        select_layer: int = -1,
        pixel_shuffle: bool = False,
        downsample_ratio: float = 0.5,
        force_float32_input: bool = False,
        cast_output_to_input_dtype: bool = False,
        vision_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.vision_model = vision_model
        self.projector = projector
        self.select_layer = int(select_layer)
        self.pixel_shuffle = bool(pixel_shuffle)
        self.downsample_ratio = float(downsample_ratio)
        self.force_float32_input = bool(force_float32_input)
        self.cast_output_to_input_dtype = bool(cast_output_to_input_dtype)
        self.vision_kwargs = dict(vision_kwargs or {})

        with torch.no_grad():
            sample = sample_pixel_values
            if self.force_float32_input and sample.dtype != torch.float32:
                sample = sample.to(torch.float32)

            vit_embeds = self._select_vision_features(self._run_vision(sample))
            self.seq_len = int(vit_embeds.shape[1])
            self.hidden_size = int(vit_embeds.shape[2])

            if self.pixel_shuffle:
                self._init_pixel_shuffle_shape()
                vit_embeds = self._apply_pixel_shuffle(vit_embeds)

            projected = self.projector(self._projector_input(vit_embeds))
            self.batch_size = int(projected.shape[0])
            self.output_seq_len = int(projected.shape[1])
            self.output_hidden_size = int(projected.shape[2])
            self.output_num_tokens = self.batch_size * self.output_seq_len

    def _run_vision(self, images: torch.Tensor):
        pixel_values = (
            hwc_to_nchw(images) if not is_nchw_pixel_values(images) else images
        )
        kwargs = dict(self.vision_kwargs)
        kwargs["pixel_values"] = pixel_values
        kwargs["output_hidden_states"] = self.select_layer != -1
        kwargs.setdefault("return_dict", True)
        return self.vision_model(**kwargs)

    def _select_vision_features(self, out):
        if self.select_layer == -1:
            return (
                out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            )
        return out.hidden_states[self.select_layer]

    def _projector_input(self, vit_embeds: torch.Tensor) -> torch.Tensor:
        proj_dtype = next(self.projector.parameters()).dtype
        if vit_embeds.dtype != proj_dtype:
            return vit_embeds.to(proj_dtype)
        return vit_embeds

    def _init_pixel_shuffle_shape(self):
        side = int(self.seq_len**0.5)
        if side * side != self.seq_len:
            raise ValueError(
                f"Expected square vision sequence, got seq_len={self.seq_len}"
            )
        self.grid_w = side
        self.grid_h = side
        self.out_w = int(self.grid_w * self.downsample_ratio)
        self.out_h = int(self.grid_h * self.downsample_ratio)
        self.hidden_after_first_view = int(self.hidden_size / self.downsample_ratio)
        self.shuffle_hidden = int(
            self.hidden_size / (self.downsample_ratio * self.downsample_ratio)
        )

    def _apply_pixel_shuffle(self, x):
        n = x.shape[0]
        x = x.reshape(n, self.grid_w, self.out_h, self.hidden_after_first_view)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(n, self.out_h, self.out_w, self.shuffle_hidden)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.reshape(n, -1, self.shuffle_hidden)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out_dtype = pixel_values.dtype
        images = pixel_values
        if self.force_float32_input and images.dtype != torch.float32:
            images = images.to(torch.float32)

        vit_embeds = self._select_vision_features(self._run_vision(images))
        if self.pixel_shuffle:
            vit_embeds = self._apply_pixel_shuffle(vit_embeds)

        features = self.projector(self._projector_input(vit_embeds))
        return self._finalize_output(features, out_dtype)


def _as_tensor(x):
    """Unwrap tuple/list outputs from patched attention modules."""
    if isinstance(x, (tuple, list)):
        return x[0]
    return x


def language_decoder(language: nn.Module) -> nn.Module:
    """Inner module that owns ``.layers``.

    Paligemma / PI05 store layers on ``language_model`` itself. HF
    ``*ForCausalLM`` stores them on ``language_model.model``. Prefer ``.layers``
    so a stray ``.model`` attribute cannot silently pick the wrong submodule.
    """
    if hasattr(language, "layers"):
        return language
    inner = getattr(language, "model", None)
    if isinstance(inner, nn.Module) and hasattr(inner, "layers"):
        return inner
    raise AttributeError(f"{type(language).__name__} has no decoder .layers")


def gather_last_token_hidden(
    hidden_states: torch.Tensor,
    last_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Gather [B, S, H] at last_token_ids [B] or [B, 1] -> [B, H] for lm_head."""
    if last_token_ids.ndim == 1:
        indices = last_token_ids
    else:
        indices = last_token_ids.squeeze(-1)
    batch_idx = torch.arange(
        hidden_states.shape[0],
        device=hidden_states.device,
        dtype=torch.long,
    )
    return hidden_states[batch_idx, indices]


class CausalLMPatch(nn.Module):
    """Edge-LLM causal LM: manual decoder loop -> logits + lm_hidden_states + prefix KV.

    External RoPE and KV-cache controls match ``LLMEngineRunner`` prefill/decode.
    ``select_layer=-1`` (default for GR00T TRT) uses final RMSNorm hidden for
    context; positive values capture an intermediate layer output pre-norm.

    ``ds_stack`` is always an input (``[num_ds, B, S, H]``). Models without
    deepstack pass ``num_ds=0`` (empty leading dim); the add is a no-op.
    """

    def __init__(
        self,
        lm: nn.Module,
        lm_head: nn.Module,
        *,
        select_layer: int = -1,
    ):
        super().__init__()
        self.lm = lm
        self.lm_head = lm_head
        self.select_layer = int(select_layer)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        last_token_ids: torch.Tensor,
        ds_stack: torch.Tensor,
        *past_key_values: torch.Tensor,
    ):
        lm_dtype = next(self.lm.parameters()).dtype
        hidden = _as_tensor(inputs_embeds).to(dtype=lm_dtype)
        seq_len = inputs_embeds.shape[1]
        num_ds = int(ds_stack.shape[0])
        context_hidden = hidden if self.select_layer == 0 else None
        new_kvs = []

        for i, layer in enumerate(self.lm.layers):
            residual = hidden
            hidden = _as_tensor(layer.input_layernorm(hidden))
            hidden, kv = layer.self_attn(
                hidden_states=hidden,
                rope_rotary_cos_sin=rope_rotary_cos_sin,
                past_key_value=past_key_values[i],
                ctx_len=context_lengths,
                kvcache_start_index=kvcache_start_index,
            )
            hidden = _as_tensor(hidden)
            hidden = residual + hidden

            residual = hidden
            hidden = _as_tensor(layer.post_attention_layernorm(hidden))
            hidden = _as_tensor(layer.mlp(hidden))
            hidden = residual + hidden
            new_kvs.append(kv)

            if i < num_ds:
                hidden = hidden + ds_stack[i, :, :seq_len, :].to(dtype=hidden.dtype)

            if self.select_layer > 0 and (i + 1) == self.select_layer:
                context_hidden = hidden

        hidden = _as_tensor(self.lm.norm(hidden))
        if context_hidden is None:
            context_hidden = hidden

        last_hidden = gather_last_token_hidden(hidden, last_token_ids)
        logits = self.lm_head(last_hidden).float()

        prefix_k = torch.stack(
            [kv[:, 0, :, :seq_len, :] for kv in new_kvs],
            dim=0,
        )
        prefix_v = torch.stack(
            [kv[:, 1, :, :seq_len, :] for kv in new_kvs],
            dim=0,
        )
        return logits, context_hidden, prefix_k, prefix_v


class ActionStepEncoderPatch(nn.Module):
    """Base contract for model-specific action-step encoding.

    Subclasses implement forward() to turn noisy actions, timestep, and
    model-specific context tensors into the args/kwargs consumed by the action
    expert and velocity decoder. The default helpers cover common expert output
    and velocity shapes, while model-specific encoders can override them.
    """

    def get_action_hidden(self, expert_out, output_tokens: int):
        # Default path for experts that return either a standard model output
        # with last_hidden_state, a raw hidden-state tensor, or a tuple/list whose
        # first item is the hidden-state tensor.
        hidden = (
            expert_out.last_hidden_state
            if hasattr(expert_out, "last_hidden_state")
            else expert_out
        )

        if isinstance(hidden, (tuple, list)):
            hidden = hidden[0]

        return hidden[:, -output_tokens:]

    def process_velocity(self, velocity):
        # Default path for models whose decoder already returns the final action
        # velocity shape. Override for models that need reshaping or cropping.
        return velocity


class StaticActionVelocityStepPatch(nn.Module):
    """One static denoising step shared by VLA action diffusion modules.

    The model-specific step_encoder owns the messy part: converting noisy
    actions, timestep, and context tensors into the exact action_expert call.
    This wrapper only runs the expert, selects action-token hidden states, and
    decodes those hidden states into a velocity update.
    """

    def __init__(
        self,
        *,
        step_encoder: ActionStepEncoderPatch,
        action_expert: nn.Module,
        velocity_decoder: nn.Module,
        output_tokens: int,
        cast_hidden_fp32: bool = True,
    ):
        super().__init__()
        self.step_encoder = step_encoder
        self.action_expert = action_expert
        self.velocity_decoder = velocity_decoder
        self.output_tokens = int(output_tokens)
        self.cast_hidden_fp32 = cast_hidden_fp32

    def forward(self, x_t, timestep, *inputs):
        # Build the action expert inputs and any decoder-specific side inputs.
        expert_args, expert_kwargs, decoder_args, decoder_kwargs = self.step_encoder(
            x_t,
            timestep,
            *inputs,
        )

        # Run the model-specific action expert: Gemma expert, DiT, etc.
        expert_out = self.action_expert(*expert_args, **expert_kwargs)

        # Most experts return last_hidden_state, but some wrappers return tuples
        # or need custom suffix-token selection.
        action_hidden = self.step_encoder.get_action_hidden(
            expert_out,
            self.output_tokens,
        )

        if self.cast_hidden_fp32:
            action_hidden = action_hidden.to(dtype=torch.float32)

        # Project action-token hidden states back to action-space velocity.
        velocity = self.velocity_decoder(
            action_hidden,
            *decoder_args,
            **decoder_kwargs,
        )

        return self.step_encoder.process_velocity(velocity)
