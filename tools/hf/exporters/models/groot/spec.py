from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import torch
import torch.nn as nn

from ...ops import call_engine, scatter_image_tokens
from ...spec import (
    ComponentBundle,
    EdgeSpec,
    register_edge_spec,
)
from ..common.helpers import (
    causal_lm_flat,
    kv_kwargs,
    split_flat_to_kwargs,
)
from .helpers import (
    _groot,
    make_embodiment_id,
)
from .patches import apply_groot_patches


def _export_module(module: nn.Module, sample: Mapping[str, Any]) -> nn.Module:
    device = sample["pixel_values"].device
    dtype = sample["pixel_values"].dtype
    return module.eval().to(device=device, dtype=dtype)


def _causal_lm(language: nn.Module) -> nn.Module:
    get_base = getattr(language, "get_base_model", None)
    if callable(get_base):
        try:
            return get_base()
        except Exception:
            pass
    return language


@register_edge_spec("groot", "gr00t")
class GrootSpec(EdgeSpec):  # type: ignore[misc]
    def apply_patches(self, model=None):
        return apply_groot_patches(model)

    def prepare_sample_inputs(
        self, model: nn.Module, raw: Mapping[str, Any], config: Any
    ) -> MutableMapping[str, Any]:
        if "pixel_values" in raw and "input_ids" in raw:
            return dict(raw)

        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.groot.processor_groot import GrootEagleEncodeStep

        from ...data import (
            create_pil_messages,
            load_test_data,
            pack_state,
        )

        policy = model
        device = raw.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        dtype = raw.get("dtype", torch.float16)
        cfg = getattr(policy, "config", None)
        pre_processor, _ = make_pre_post_processors(
            cfg,
            None,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        eagle_step = next(
            s for s in pre_processor.steps if isinstance(s, GrootEagleEncodeStep)
        )
        proc = eagle_step.proc
        data = raw.get("data") or load_test_data(
            raw.get("dataset_id", "lerobot/libero"), episode_index=0, frame_index=0
        )
        messages = create_pil_messages(data)
        text = proc.apply_chat_template(
            messages, tokenize=False, **{"add_generation_prompt": True}
        )
        image_inputs, video_inputs = proc.process_vision_info(messages)
        tokenized = proc(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            **{
                "images_kwargs": {
                    "min_dynamic_tiles": 1,
                    "max_dynamic_tiles": 1,
                    "use_thumbnail": False,
                }
            },
        )
        state = (
            pack_state(
                data["state"],
                max_state_dim=int(getattr(cfg, "max_state_dim", 64)),
                device=device,
            )
            .to(device=device, dtype=dtype)
            .contiguous()
        )
        return {
            "pixel_values": tokenized["pixel_values"].to(device=device, dtype=dtype),
            "input_ids": tokenized["input_ids"].to(device=device, dtype=torch.long),
            "attention_mask": tokenized["attention_mask"].to(
                device=device, dtype=torch.long
            ),
            "state": state,
            "embodiment_id": make_embodiment_id(policy, state, device, torch.long),
        }

    def capture_eager_outputs(
        self, model, sample, config, bench=None
    ) -> dict[str, torch.Tensor]:
        del config
        from ...measure import cuda_ms

        found = _groot(model)
        eagle = found.backbone.eagle_model
        language = _causal_lm(eagle.language_model)
        px = sample["pixel_values"]
        lm_hidden = sample["lm_hidden"]
        action_head = found.action_head

        with torch.no_grad():
            visual_embeds = eagle.extract_feature(px)
            lm = language(
                inputs_embeds=sample["inputs_embeds"],
                attention_mask=sample.get("attention_mask"),
                return_dict=True,
            )
            context_embs = found.backbone.eagle_linear(lm_hidden)
            vlln = found.action_head.vlln
            weight = getattr(vlln, "weight", None)
            if weight is not None:
                context_embs = context_embs.to(dtype=weight.dtype)
            context_embs = vlln(context_embs)
            context_embs = found.action_head.vl_self_attention(context_embs)
            state_features = action_head.state_encoder(
                sample["state"], sample["embodiment_id"]
            )
            action_features = action_head.action_encoder(
                sample["step_actions"],
                sample["step_timestep"],
                sample["embodiment_id"],
            )
            if action_head.config.add_pos_embed:
                pos_ids = torch.arange(
                    action_features.shape[1],
                    dtype=torch.long,
                    device=action_features.device,
                )
                action_features = action_features + action_head.position_embedding(
                    pos_ids
                ).unsqueeze(0)
            future_tokens = action_head.future_tokens.weight.unsqueeze(0).expand(
                sample["context_embs"].shape[0],
                -1,
                -1,
            )
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
            expert_out = action_head.model(
                hidden_states=sa_embs,
                encoder_hidden_states=sample["context_embs"],
                timestep=sample["step_timestep"],
            )
            action_hidden = (
                expert_out.last_hidden_state
                if hasattr(expert_out, "last_hidden_state")
                else expert_out
            )
            if isinstance(action_hidden, (tuple, list)):
                action_hidden = action_hidden[0]
            velocity = action_head.action_decoder(
                action_hidden[:, -int(action_head.config.action_horizon) :],
                sample["embodiment_id"],
            )

        if bench is not None:
            bench["vision"] = cuda_ms(lambda: eagle.extract_feature(px))
            bench["language"] = cuda_ms(
                lambda: language(
                    inputs_embeds=sample["inputs_embeds"],
                    attention_mask=sample.get("attention_mask"),
                    return_dict=True,
                )
            )
        return {
            "vision": visual_embeds,
            "language": (
                lm.last_hidden_state if hasattr(lm, "last_hidden_state") else lm[0]
            ),
            "context_projection": context_embs,
            "action": velocity,
        }

    def prepare(
        self,
        model: nn.Module,
        sample: MutableMapping[str, Any],
        config: Any,
    ) -> dict[str, ComponentBundle]:
        from ...plugin.attention import (
            ContextAttentionMaskType,
        )

        found = _groot(model)
        eagle = found.backbone.eagle_model
        px = sample["pixel_values"]
        device = px.device
        dtype = px.dtype

        vision = ComponentBundle(
            module=_export_module(eagle, sample),
            trace_args=(px,),
            save_args=(px,),
            input_names=["pixel_values"],
            output_names=["visual_embeds"],
            model_type="vit",
            engine_file="visual.engine",
            trt_settings={
                "disable_tf32": False,
                "use_fp32_acc": False,
                "use_explicit_typing": False,
                "decompose_attention": True,
            },
        )

        language = _causal_lm(eagle.language_model)
        input_ids = sample["input_ids"]
        input_embs = language.get_input_embeddings()(input_ids)
        image_token_index = getattr(
            eagle, "image_token_index", eagle.config.image_token_index
        )
        mask = input_ids == image_token_index
        sample["image_token_mask"] = mask
        with torch.no_grad():
            vis = eagle.extract_feature(px)
        hidden = input_embs.shape[-1]
        flat_emb = input_embs.clone().reshape(-1, hidden)
        vis_flat = vis.reshape(-1, hidden).to(
            device=flat_emb.device, dtype=flat_emb.dtype
        )
        n = int(mask.reshape(-1).sum().item())
        flat_emb[mask.reshape(-1)] = vis_flat[:n]
        inputs_embeds = (
            flat_emb.reshape_as(input_embs).to(device=device, dtype=dtype).contiguous()
        )
        sample["lang_embeds"] = input_embs.to(device=device, dtype=dtype)
        max_seq_len = max(int(config.max_seq_len), int(inputs_embeds.shape[1]))
        packed, meta = causal_lm_flat(
            language,
            inputs_embeds,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        sample.update(split_flat_to_kwargs(packed, meta["input_names"]))

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
            trt_settings={
                "disable_tf32": True,
                "use_fp32_acc": True,
                "use_explicit_typing": True,
                "decompose_attention": True,
                "assume_dynamic_shape_support": True,
            },
        )

        bsz, seq_len, hidden_size = inputs_embeds.shape
        lm_hidden = torch.zeros(bsz, seq_len, hidden_size, device=device, dtype=dtype)
        sample["lm_hidden"] = lm_hidden
        context_projection = ComponentBundle(
            module=_export_module(found, sample),
            trace_args=(lm_hidden,),
            save_args=(lm_hidden,),
            input_names=["lm_hidden_states"],
            output_names=["vl_embs"],
            model_type="context_projection",
            engine_file="context_projection.engine",
            trt_settings={
                "disable_tf32": True,
                "use_fp32_acc": True,
                "use_explicit_typing": True,
                "decompose_attention": True,
            },
        )

        out_dim = int(found.backbone.eagle_linear.out_features)
        context_embs = torch.zeros(bsz, seq_len, out_dim, device=device, dtype=dtype)
        horizon = int(found.action_head.config.action_horizon)
        action_dim = int(found.action_head.config.action_dim)
        step_actions = sample.get(
            "step_actions",
            torch.randn(bsz, horizon, action_dim, device=device, dtype=dtype),
        )
        step_timestep = sample.get(
            "step_timestep",
            torch.zeros(bsz, device=device, dtype=dtype),
        )
        sample["step_actions"] = step_actions
        sample["step_timestep"] = step_timestep
        sample["context_embs"] = context_embs
        args = (
            step_actions,
            step_timestep,
            context_embs,
            sample["state"],
            sample["embodiment_id"],
        )
        action = ComponentBundle(
            module=_export_module(found.action_head, sample),
            trace_args=args,
            save_args=args,
            input_names=[
                "actions",
                "timestep",
                "context_embs",
                "state",
                "embodiment_id",
            ],
            output_names=["velocity"],
            model_type="action",
            engine_file="action.engine",
            trt_settings={
                "disable_tf32": True,
                "use_fp32_acc": True,
                "use_explicit_typing": True,
                "decompose_attention": True,
            },
        )
        return {
            "vision": vision,
            "language": language_bundle,
            "context_projection": context_projection,
            "action": action,
        }

    def run(self, engines: Mapping[str, str], sample: Mapping[str, Any]) -> Any:
        vis = call_engine(engines["vision"], "vision", sample["pixel_values"])[0]
        embeds = scatter_image_tokens(
            vis, sample["lang_embeds"], sample["image_token_mask"]
        )
        lm = call_engine(
            engines["language"],
            "language",
            embeds,
            sample["rope_rotary_cos_sin"],
            sample["context_lengths"],
            sample["kvcache_start_index"],
            sample["last_token_ids"],
            sample["ds_stack"],
            *kv_kwargs(sample),
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
        )
