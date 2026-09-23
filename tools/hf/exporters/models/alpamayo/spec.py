from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import torch
import torch.nn as nn

from ...ops import call_engine, scatter_image_tokens
from ...prefix_cache import PrefixKVCache
from ...spec import ComponentBundle, EdgeSpec, register_edge_spec
from ..common.helpers import (
    causal_lm_flat,
    kv_kwargs,
    split_flat_to_kwargs,
)
from .helpers import (
    alpamayo_language,
    alpamayo_visual,
    alpamayo_vlm,
    alpamayo_vlm_core,
    make_deepstack_tensor,
    prepare_fixed_grid_vision,
    scatter_visual_tokens,
    stack_deepstack_features,
)
from .patches import ALPAMAYO


def _export_module(module: nn.Module, device: torch.device, dtype: torch.dtype):
    return module.eval().to(device=device, dtype=dtype)


def _append_generation_positions(
    position_ids: torch.Tensor,
    rope_deltas: torch.Tensor,
    max_seq_len: int,
) -> torch.Tensor:
    """Extend Qwen multimodal positions with ordinary text decode positions."""
    prompt_len = int(position_ids.shape[-1])
    if prompt_len >= max_seq_len:
        return position_ids[..., :max_seq_len]
    batch_size = int(position_ids.shape[1])
    tail = torch.arange(
        prompt_len,
        max_seq_len,
        device=position_ids.device,
        dtype=position_ids.dtype,
    )
    tail = tail.unsqueeze(0).expand(batch_size, -1)
    tail = tail + rope_deltas.to(device=tail.device, dtype=tail.dtype)
    tail = tail.unsqueeze(0).expand(3, -1, -1)
    return torch.cat((position_ids, tail), dim=-1)


@register_edge_spec("alpamayo", "alpamayo_r1", "alpamayo1_5")
class AlpamayoSpec(EdgeSpec):  # type: ignore[misc]
    def apply_patches(self, model=None):
        del model
        from ...plugin.attn_patches import apply_patches

        return apply_patches(ALPAMAYO)

    def prepare_sample_inputs(
        self,
        model: nn.Module,
        raw: Mapping[str, Any],
        config: Any,
    ) -> MutableMapping[str, Any]:
        del config
        required = {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "image_grid_thw",
        }
        if required.issubset(raw):
            return dict(raw)

        clip_id = raw.get("clip_id")
        if not clip_id:
            raise ValueError(
                "Alpamayo export needs --clip-id, or pre-tokenized input_ids, "
                "attention_mask, pixel_values, and image_grid_thw."
            )

        from alpamayo1_5 import helper
        from alpamayo1_5.load_physical_aiavdataset import (
            load_physical_aiavdataset,
        )

        device = raw.get(
            "device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        dtype = raw.get("dtype", torch.float16)
        data = raw.get("data") or load_physical_aiavdataset(
            str(clip_id),
            t0_us=int(raw.get("t0_us", 5_100_000)),
        )
        messages = helper.create_message(
            data["image_frames"].flatten(0, 1),
            camera_indices=data["camera_indices"],
        )
        processor = helper.get_processor(model.tokenizer)
        tokenized = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = model.fuse_traj_tokens(
            tokenized["input_ids"],
            {
                "ego_history_xyz": data["ego_history_xyz"],
                "ego_history_rot": data["ego_history_rot"],
            },
        )
        return {
            "input_ids": input_ids.to(device=device, dtype=torch.long),
            "attention_mask": tokenized["attention_mask"].to(
                device=device,
                dtype=torch.long,
            ),
            "pixel_values": tokenized["pixel_values"].to(
                device=device,
                dtype=dtype,
            ),
            "image_grid_thw": tokenized["image_grid_thw"].to(
                device=device,
                dtype=torch.long,
            ),
        }

    def capture_eager_outputs(
        self,
        model: nn.Module,
        sample: MutableMapping[str, Any],
        config: Any,
        bench: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        del config
        from ...measure import cuda_ms

        visual = alpamayo_visual(model)
        language = alpamayo_language(model)
        px = sample["pixel_values"]
        grid = sample["image_grid_thw"]

        with torch.no_grad():
            visual_embeds, deepstack = visual(px, grid)
            deepstack = stack_deepstack_features(deepstack)
            language_out = language(
                inputs_embeds=sample["inputs_embeds"],
                attention_mask=sample["attention_mask"],
                position_ids=sample["position_ids"],
                visual_pos_masks=sample["image_token_mask"],
                deepstack_visual_embeds=[
                    deepstack[i] for i in range(int(deepstack.shape[0]))
                ],
                use_cache=False,
                return_dict=True,
            )
            action_embeds = model.action_in_proj(
                sample["step_actions"],
                sample["step_timestep"],
            )
            expert_kwargs: dict[str, Any] = {}
            if model.config.expert_non_causal_attention:
                expert_kwargs["is_causal"] = False
            expert = model.expert(
                inputs_embeds=action_embeds,
                position_ids=sample["suffix_position_ids"],
                past_key_values=PrefixKVCache(
                    sample["prefix_k"],
                    sample["prefix_v"],
                ),
                attention_mask=sample["suffix_attention_mask"],
                use_cache=False,
                return_dict=True,
                **expert_kwargs,
            )
            velocity = model.action_out_proj(
                expert.last_hidden_state[:, -int(sample["step_actions"].shape[1]) :]
            ).reshape_as(sample["step_actions"])

        if bench is not None:
            bench["vision"] = cuda_ms(lambda: visual(px, grid)[0])
            bench["language"] = cuda_ms(
                lambda: language(
                    inputs_embeds=sample["inputs_embeds"],
                    attention_mask=sample["attention_mask"],
                    position_ids=sample["position_ids"],
                    visual_pos_masks=sample["image_token_mask"],
                    deepstack_visual_embeds=[
                        deepstack[i] for i in range(int(deepstack.shape[0]))
                    ],
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state
            )
        return {
            "vision": visual_embeds,
            "language": language_out.last_hidden_state,
            "action": velocity,
        }

    def prepare(
        self,
        model: nn.Module,
        sample: MutableMapping[str, Any],
        config: Any,
    ) -> dict[str, ComponentBundle]:
        from ...plugin.attention import ContextAttentionMaskType

        visual = alpamayo_visual(model)
        language = alpamayo_language(model)
        vlm = alpamayo_vlm(model)
        vlm_core = alpamayo_vlm_core(model)
        px = sample["pixel_values"]
        grid = sample["image_grid_thw"]
        device = px.device
        dtype = px.dtype

        visual.config.attn_implementation = "sdpa"
        visual.config._attn_implementation = "sdpa"
        prepare_fixed_grid_vision(visual, grid)
        vision = ComponentBundle(
            module=_export_module(visual, device, dtype),
            trace_args=(px,),
            save_args=(px,),
            input_names=["pixel_values"],
            output_names=["visual_embeds", "deepstack_visual_embeds"],
            parity_output="visual_embeds",
            model_type="qwen3_vl",
            engine_file="visual.engine",
            trt_settings={
                "disable_tf32": False,
                "use_fp32_acc": False,
                "use_explicit_typing": False,
                "decompose_attention": True,
            },
        )

        with torch.no_grad():
            visual_embeds, deepstack = visual(px, grid)
            deepstack = stack_deepstack_features(deepstack)
            text_embeds = model.get_input_embeddings()(sample["input_ids"])

        image_token_id = int(vlm.config.image_token_id)
        image_token_mask = sample["input_ids"] == image_token_id
        inputs_embeds = scatter_visual_tokens(
            visual_embeds,
            text_embeds,
            image_token_mask,
        ).to(device=device, dtype=dtype)
        sample["lang_embeds"] = text_embeds.to(device=device, dtype=dtype)
        sample["image_token_mask"] = image_token_mask
        sample["inputs_embeds"] = inputs_embeds

        try:
            position_ids, rope_deltas = vlm_core.get_rope_index(
                sample["input_ids"],
                sample["image_grid_thw"],
                None,
                attention_mask=sample["attention_mask"],
            )
        except (TypeError, IndexError):
            image_token_types = (
                sample["input_ids"] == int(vlm_core.config.image_token_id)
            ).to(torch.int32)
            position_ids, rope_deltas = vlm_core.get_rope_index(
                sample["input_ids"],
                mm_token_type_ids=image_token_types,
                image_grid_thw=sample["image_grid_thw"],
                video_grid_thw=None,
                attention_mask=sample["attention_mask"],
            )
        sample["position_ids"] = position_ids
        sample["rope_deltas"] = rope_deltas

        decoder_layers = language.layers
        ds_stack = make_deepstack_tensor(
            deepstack,
            image_token_mask,
            num_layers=len(decoder_layers),
            batch_size=int(inputs_embeds.shape[0]),
            seq_len=int(inputs_embeds.shape[1]),
            hidden_size=int(inputs_embeds.shape[2]),
            device=device,
            dtype=dtype,
        )
        sample["ds_template"] = ds_stack

        prompt_len = int(inputs_embeds.shape[1])
        if int(config.generation_reserve) < 0:
            raise ValueError("generation_reserve must be non-negative")
        max_seq_len = max(
            int(config.max_seq_len),
            prompt_len + int(config.generation_reserve),
        )
        full_position_ids = _append_generation_positions(
            position_ids,
            rope_deltas,
            max_seq_len,
        )
        flat, meta = causal_lm_flat(
            language,
            inputs_embeds,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
            seq_len=prompt_len,
            position_ids=full_position_ids,
        )
        flat = (*flat[:5], ds_stack, *flat[6:])
        sample.update(split_flat_to_kwargs(flat, meta["input_names"]))

        language_bundle = ComponentBundle(
            module=_export_module(vlm, device, dtype),
            trace_args=flat,
            save_args=flat,
            input_names=meta["input_names"],
            output_names=["logits", "lm_hidden_states", "prefix_k", "prefix_v"],
            parity_output="lm_hidden_states",
            context_attention_mask_type=int(ContextAttentionMaskType.CAUSAL),
            model_type="language",
            engine_file="language.engine",
            trt_settings={
                "disable_tf32": True,
                "use_fp32_acc": True,
                "use_explicit_typing": True,
                "decompose_attention": True,
                "assume_dynamic_shape_support": True,
                "immutable_weights": False,
            },
        )

        action_dims = tuple(model.action_space.get_action_space_dims())
        batch_size = int(inputs_embeds.shape[0])
        step_actions = sample.get(
            "step_actions",
            torch.randn(
                batch_size,
                *action_dims,
                device=device,
                dtype=dtype,
            ),
        )
        step_timestep = sample.get(
            "step_timestep",
            torch.full(
                (batch_size, 1, 1),
                0.5,
                device=device,
                dtype=dtype,
            ),
        )
        sample["step_actions"] = step_actions
        sample["step_timestep"] = step_timestep

        language_cfg = language.config
        expert_cfg = model.expert.config
        for field in ("num_hidden_layers", "num_key_value_heads", "head_dim"):
            if int(getattr(language_cfg, field)) != int(getattr(expert_cfg, field)):
                raise ValueError(
                    "Alpamayo language/expert KV layouts differ at "
                    f"{field}: {getattr(language_cfg, field)} vs "
                    f"{getattr(expert_cfg, field)}"
                )
        num_kv_heads = int(expert_cfg.num_key_value_heads)
        head_dim = int(
            getattr(
                expert_cfg,
                "head_dim",
                expert_cfg.hidden_size // expert_cfg.num_attention_heads,
            )
        )
        prefix_k = torch.zeros(
            int(expert_cfg.num_hidden_layers),
            batch_size,
            num_kv_heads,
            prompt_len,
            head_dim,
            device=device,
            dtype=dtype,
        )
        prefix_v = torch.zeros_like(prefix_k)
        suffix_position_ids, suffix_attention_mask = (
            model._build_expert_pos_ids_and_attn_mask(
                offset=torch.full(
                    (batch_size,),
                    prompt_len,
                    device=device,
                    dtype=torch.long,
                ),
                rope_deltas=rope_deltas,
                kv_cache_seq_len=prompt_len,
                n_diffusion_tokens=int(action_dims[0]),
                b_star=batch_size,
                device=device,
                prefix_mask=sample["attention_mask"],
            )
        )
        sample["prefix_k"] = prefix_k
        sample["prefix_v"] = prefix_v
        sample["suffix_position_ids"] = suffix_position_ids
        sample["suffix_attention_mask"] = suffix_attention_mask

        model.expert.config._attn_implementation = "sdpa"
        action_args = (
            step_actions,
            step_timestep,
            prefix_k,
            prefix_v,
            suffix_position_ids,
            suffix_attention_mask,
        )
        action = ComponentBundle(
            module=_export_module(model, device, dtype),
            trace_args=action_args,
            save_args=action_args,
            input_names=[
                "noisy_action",
                "timestep",
                "prefix_k",
                "prefix_v",
                "position_ids",
                "attention_mask",
            ],
            output_names=["velocity"],
            model_type="action",
            engine_file="action.engine",
            trt_settings={
                "disable_tf32": True,
                "use_fp32_acc": True,
                "use_explicit_typing": True,
                "decompose_attention": True,
                "immutable_weights": False,
            },
        )
        return {"vision": vision, "language": language_bundle, "action": action}

    def run(self, engines: Mapping[str, str], sample: Mapping[str, Any]) -> Any:
        visual, deepstack = call_engine(
            engines["vision"],
            "vision",
            sample["pixel_values"],
        )
        inputs_embeds = scatter_image_tokens(
            visual,
            sample["lang_embeds"],
            sample["image_token_mask"],
        )

        ds_template = sample["ds_template"]
        ds_layers = []
        for layer_index in range(int(ds_template.shape[0])):
            if layer_index < int(deepstack.shape[0]):
                ds_layers.append(
                    scatter_image_tokens(
                        deepstack[layer_index],
                        ds_template[layer_index],
                        sample["image_token_mask"],
                    )
                )
            else:
                ds_layers.append(ds_template[layer_index])
        ds_stack = torch.stack(ds_layers, dim=0)

        language = call_engine(
            engines["language"],
            "language",
            inputs_embeds,
            sample["rope_rotary_cos_sin"],
            sample["context_lengths"],
            sample["kvcache_start_index"],
            sample["last_token_ids"],
            ds_stack,
            *kv_kwargs(sample),
        )
        return call_engine(
            engines["action"],
            "action",
            sample["step_actions"],
            sample["step_timestep"],
            language[2],
            language[3],
            sample["suffix_position_ids"],
            sample["suffix_attention_mask"],
        )
