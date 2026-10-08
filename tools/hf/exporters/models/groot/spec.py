from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import torch
import torch.nn as nn

from ...ops import call_engine, scatter_image_tokens
from ...spec import ComponentBundle, EdgeSpec, register_edge_spec
from ..common.helpers import causal_lm_flat, kv_kwargs, split_flat_to_kwargs
from .helpers import (
    _groot,
    make_embodiment_id,
    make_vlm_encode_step,
    mrope_rotary_cos_sin,
    vision_grid_inputs,
)
from .patches import GROOT, action_velocity
from .vision import GrootQwen3Vision

VISION_INPUTS = (
    "pixel_values",
    "vision_interp_indices",
    "vision_interp_weights",
    "vision_position_ids",
    "vision_cu_seqlens",
    "vision_max_seqlen_carrier",
)
ACTION_INPUTS = (
    "step_actions",
    "step_timestep",
    "context_embs",
    "state",
    "embodiment_id",
    "image_token_mask",
    "backbone_attention_mask",
)
# Eager-only tensors: the runtime graph rebuilds these from engine outputs.
_EAGER_ONLY = frozenset(
    {
        "input_ids",
        "attention_mask",
        "image_grid_thw",
        "mm_token_type_ids",
        "inputs_embeds",
        "ds_stack",
        "lm_hidden",
        "context_embs",
    }
)
_FP32_ACC = {
    "disable_tf32": True,
    "use_fp32_acc": True,
    "use_explicit_typing": True,
    "decompose_attention": True,
}


def _export_module(module: nn.Module, sample: Mapping[str, Any]) -> nn.Module:
    device = sample["pixel_values"].device
    dtype = sample["pixel_values"].dtype
    return module.eval().to(device=device, dtype=dtype)


def _uint8_hwc(img: torch.Tensor) -> torch.Tensor:
    img = img.detach().cpu()
    if img.dtype.is_floating_point:
        img = (img.clamp(0, 1) * 255).round().to(torch.uint8)
    return img.permute(1, 2, 0) if img.shape[0] in (1, 3) else img


def _backbone_inputs(sample: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    keys = (
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "mm_token_type_ids",
    )
    return {key: sample[key] for key in keys if key in sample}


@register_edge_spec("groot", "gr00t")
class GrootSpec(EdgeSpec):  # type: ignore[misc]
    """GR00T N1.7: Qwen3-VL vision + truncated Qwen3-VL text + VL projection + DiT step."""

    def apply_patches(self, model=None):
        del model
        from ...plugin.attn_patches import apply_patches

        return apply_patches(GROOT)

    def prepare_sample_inputs(
        self, model: nn.Module, raw: Mapping[str, Any], config: Any
    ) -> MutableMapping[str, Any]:
        if "pixel_values" in raw and "input_ids" in raw:
            return dict(raw)

        from lerobot.lerobot_types import TransitionKey
        from lerobot.policies.groot.utils import prepare_n1_7_language_batch

        from ...data import load_test_data, pack_state

        policy = model
        found = _groot(model)
        device = raw.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        dtype = raw.get("dtype", torch.float16)
        cfg = getattr(policy, "config", None)
        data = raw.get("data") or load_test_data(
            raw.get("dataset_id", "lerobot/libero"), episode_index=0, frame_index=0
        )
        step, formalize_language = make_vlm_encode_step(
            getattr(cfg, "base_model_path", None) or found.config.name_or_path
        )
        # (B, T, V, H, W, C) uint8, one timestep, views in sorted key order.
        views = [_uint8_hwc(img) for _, img in sorted(data["images"].items())]
        video = torch.stack(views)[None, None].numpy()
        language = prepare_n1_7_language_batch(
            data.get("task"), 1, formalize_language=formalize_language
        )
        encoded = step(
            {
                TransitionKey.OBSERVATION: {"video": video},
                TransitionKey.COMPLEMENTARY_DATA: {"language": language},
            }
        )[TransitionKey.COMPLEMENTARY_DATA]

        state = (
            pack_state(
                data["state"],
                max_state_dim=int(getattr(cfg, "max_state_dim", 132)),
                device=device,
            )
            .to(device=device, dtype=dtype)
            .contiguous()
        )
        grid_thw = encoded["image_grid_thw"].to(device=device)
        sample: dict[str, Any] = {
            "pixel_values": encoded["pixel_values"].to(device=device, dtype=dtype),
            "input_ids": encoded["input_ids"].to(device=device, dtype=torch.long),
            "attention_mask": encoded["attention_mask"].to(
                device=device, dtype=torch.long
            ),
            "image_grid_thw": grid_thw,
            "state": state,
            "embodiment_id": make_embodiment_id(policy, state, device, torch.long),
        }
        if "mm_token_type_ids" in encoded:
            sample["mm_token_type_ids"] = encoded["mm_token_type_ids"].to(device)
        sample.update(vision_grid_inputs(found.backbone.visual, grid_thw))
        return sample

    def capture_eager_outputs(
        self, model, sample, config, bench=None
    ) -> dict[str, torch.Tensor]:
        del config
        from ...measure import cuda_ms

        found = _groot(model)
        backbone = found.backbone
        px = sample["pixel_values"]
        grid_thw = sample["image_grid_thw"]

        def action():
            return action_velocity(
                found.action_head, *(sample[name] for name in ACTION_INPUTS)
            )

        with torch.no_grad():
            visual_embeds = backbone.visual(px, grid_thw=grid_thw).pooler_output
            velocity = action()

        if bench is not None:
            vl_input = _backbone_inputs(sample)
            bench["vision"] = cuda_ms(lambda: backbone.visual(px, grid_thw=grid_thw))
            bench["language"] = cuda_ms(lambda: backbone(dict(vl_input)))
            bench["action"] = cuda_ms(action)
        return {
            "vision": visual_embeds,
            "language": sample["lm_hidden"],
            "context_projection": sample["context_embs"],
            "action": velocity,
        }

    def prepare(
        self,
        model: nn.Module,
        sample: MutableMapping[str, Any],
        config: Any,
    ) -> dict[str, ComponentBundle]:
        from ...plugin.attention import ContextAttentionMaskType

        found = _groot(model)
        backbone = found.backbone
        visual = backbone.visual
        language = backbone.language_model
        head = found.action_head
        px = sample["pixel_values"]
        device = px.device
        dtype = px.dtype

        vision_args = tuple(sample[name] for name in VISION_INPUTS)
        vision = ComponentBundle(
            module=_export_module(GrootQwen3Vision(visual), sample),
            trace_args=vision_args,
            save_args=vision_args,
            input_names=[
                "pixel_values",
                "interp_indices",
                "interp_weights",
                "position_ids",
                "cu_seqlens",
                "max_seqlen_carrier",
            ],
            output_names=["visual_embeds", "deepstack_features"],
            model_type="vit",
            engine_file="visual.engine",
            trt_settings={
                "disable_tf32": False,
                "use_fp32_acc": False,
                "use_explicit_typing": False,
                "decompose_attention": True,
            },
        )

        input_ids = sample["input_ids"]
        image_token_mask = input_ids == backbone.model.config.image_token_id
        vl_input = _backbone_inputs(sample)
        with torch.no_grad():
            vis_out = visual(px, grid_thw=sample["image_grid_thw"])
            lang_embeds = language.get_input_embeddings()(input_ids).to(dtype=dtype)
            # GR00T's own backbone forward is the language reference: pre-norm
            # output of the last kept decoder layer.
            lm_hidden = backbone(dict(vl_input))["backbone_features"].to(dtype=dtype)
        backbone._ensure_mm_token_type_ids(vl_input)
        backbone._ensure_legacy_qwen3_position_ids(vl_input)
        position_ids = vl_input["position_ids"][-3:]

        inputs_embeds = scatter_image_tokens(
            vis_out.pooler_output, lang_embeds, image_token_mask
        ).contiguous()
        num_layers = len(language.layers)
        ds_stack = torch.zeros(
            num_layers, *lang_embeds.shape, device=device, dtype=dtype
        )
        for i, feature in enumerate(vis_out.deepstack_features):
            ds_stack[i] = scatter_image_tokens(feature, ds_stack[i], image_token_mask)

        max_seq_len = max(int(config.max_seq_len), int(inputs_embeds.shape[1]))
        packed, meta = causal_lm_flat(
            language,
            inputs_embeds,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
            rope=mrope_rotary_cos_sin(language, position_ids, max_seq_len),
        )
        named = split_flat_to_kwargs(packed, meta["input_names"])
        named["ds_stack"] = ds_stack
        packed = tuple(named[name] for name in meta["input_names"])
        sample.update(named)
        sample["lang_embeds"] = lang_embeds
        sample["image_token_mask"] = image_token_mask
        sample["backbone_attention_mask"] = sample["attention_mask"] == 1

        language_bundle = ComponentBundle(
            module=_export_module(language, sample),
            trace_args=packed,
            save_args=packed,
            input_names=meta["input_names"],
            output_names=["logits", "lm_hidden_states", "prefix_k", "prefix_v"],
            parity_output="lm_hidden_states",
            context_attention_mask_type=int(ContextAttentionMaskType.CAUSAL),
            model_type="language",
            engine_file="language.engine",
            trt_settings={**_FP32_ACC, "assume_dynamic_shape_support": True},
        )

        projection = _export_module(
            nn.Sequential(head.vlln, head.vl_self_attention), sample
        )
        with torch.no_grad():
            context_embs = projection(lm_hidden)
        sample["lm_hidden"] = lm_hidden
        sample["context_embs"] = context_embs
        context_projection = ComponentBundle(
            module=projection,
            trace_args=(lm_hidden,),
            save_args=(lm_hidden,),
            input_names=["lm_hidden_states"],
            output_names=["vl_embs"],
            model_type="context_projection",
            engine_file="context_projection.engine",
            trt_settings=dict(_FP32_ACC),
        )

        bsz = int(inputs_embeds.shape[0])
        sample.setdefault(
            "step_actions",
            torch.randn(
                bsz, head.action_horizon, head.action_dim, device=device, dtype=dtype
            ),
        )
        # get_action feeds discretized integer buckets; use a mid-trajectory step.
        sample.setdefault(
            "step_timestep",
            torch.full(
                (bsz,),
                int(head.num_timestep_buckets) // 2,
                device=device,
                dtype=torch.long,
            ),
        )
        action_args = tuple(sample[name] for name in ACTION_INPUTS)
        action = ComponentBundle(
            module=_export_module(head, sample),
            trace_args=action_args,
            save_args=action_args,
            input_names=[
                "actions",
                "timestep",
                "context_embs",
                "state",
                "embodiment_id",
                "image_mask",
                "backbone_attention_mask",
            ],
            output_names=["velocity"],
            model_type="action",
            engine_file="action.engine",
            trt_settings=dict(_FP32_ACC),
        )
        return {
            "vision": vision,
            "language": language_bundle,
            "context_projection": context_projection,
            "action": action,
        }

    def runtime_kwargs(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in super().runtime_kwargs(sample).items()
            if key not in _EAGER_ONLY
        }

    def run(self, engines: Mapping[str, str], sample: Mapping[str, Any]) -> Any:
        vis, deepstack = call_engine(
            engines["vision"], "vision", *(sample[name] for name in VISION_INPUTS)
        )[:2]
        mask = sample["image_token_mask"]
        lang_embeds = sample["lang_embeds"]
        embeds = scatter_image_tokens(vis, lang_embeds, mask)
        kvs = kv_kwargs(sample)
        zeros = torch.zeros_like(lang_embeds)
        ds_layers = [
            scatter_image_tokens(deepstack[i], zeros, mask)
            for i in range(deepstack.shape[0])
        ]
        ds_stack = torch.stack(ds_layers + [zeros] * (len(kvs) - len(ds_layers)))
        lm = call_engine(
            engines["language"],
            "language",
            embeds,
            sample["rope_rotary_cos_sin"],
            sample["context_lengths"],
            sample["kvcache_start_index"],
            sample["last_token_ids"],
            ds_stack,
            *kvs,
        )
        ctx = call_engine(engines["context_projection"], "context_projection", lm[1])[0]
        return call_engine(
            engines["action"],
            "action",
            sample["step_actions"],
            sample["step_timestep"],
            ctx,
            sample["state"],
            sample["embodiment_id"],
            mask,
            sample["backbone_attention_mask"],
        )
