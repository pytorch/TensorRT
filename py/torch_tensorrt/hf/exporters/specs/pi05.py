from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import torch
import torch.nn as nn
from torch_tensorrt.hf.exporters.ops import call_engine, fuse_prefix
from torch_tensorrt.hf.exporters.spec import (
    ComponentBundle,
    EdgeSpec,
    register_edge_spec,
)
from torch_tensorrt.hf.exporters.specs._language import (
    causal_lm_flat,
    kv_kwargs,
    split_flat_to_kwargs,
)


def _core(model: nn.Module) -> nn.Module:
    if hasattr(model, "paligemma_with_expert"):
        return model
    inner = getattr(model, "model", None)
    if isinstance(inner, nn.Module) and hasattr(inner, "paligemma_with_expert"):
        return inner
    raise RuntimeError("PI05 spec expected a policy or paligemma_with_expert module")


def _nchw_to_hwc(pixel_values: torch.Tensor) -> torch.Tensor:
    if pixel_values.ndim != 4:
        return pixel_values
    if pixel_values.shape[1] in (1, 3, 4) and pixel_values.shape[-1] not in (1, 3, 4):
        return pixel_values.permute(0, 2, 3, 1).contiguous()
    return pixel_values


@register_edge_spec("pi05")
class Pi05Spec(EdgeSpec):  # type: ignore[misc]
    components = ("vision", "language", "action")

    def prepare_sample_inputs(
        self, model: nn.Module, raw: Mapping[str, Any], config: Any
    ) -> MutableMapping[str, Any]:
        if "pixel_values" in raw and "tokens" in raw:
            return dict(raw)

        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )
        from torch_tensorrt.hf.exporters.data import (
            frame_from_test_data,
            load_test_data,
        )

        policy = model if hasattr(model, "_preprocess_images") else None
        if policy is None:
            raise ValueError(
                "PI05 prepare_sample_inputs needs a LeRobot policy or a pre-collated "
                "dict with pixel_values/tokens/masks"
            )
        device = raw.get("device", next(policy.parameters()).device)
        dtype = raw.get("dtype", torch.float16)
        data = raw.get("data") or load_test_data(
            raw.get("dataset_id", "lerobot/libero"), episode_index=0, frame_index=0
        )
        frame = frame_from_test_data(data, policy, fill_missing=True)
        pre_processor, _ = make_pre_post_processors(
            policy.config,
            None,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        batch = pre_processor(frame)
        images, img_masks = policy._preprocess_images(batch)
        pixel_values = torch.cat(
            [img.to(device=device, dtype=dtype) for img in images], dim=0
        ).contiguous()
        tokens = batch[OBS_LANGUAGE_TOKENS].to(device=device, dtype=torch.long)
        masks = batch[OBS_LANGUAGE_ATTENTION_MASK].to(device=device, dtype=torch.bool)
        core = _core(policy)
        lang_embeds = core.paligemma_with_expert.embed_language_tokens(tokens)
        return {
            "pixel_values": pixel_values,
            "images": images,
            "img_masks": img_masks,
            "tokens": tokens,
            "masks": masks,
            "lang_embeds": lang_embeds.to(device=device, dtype=dtype).contiguous(),
        }

    def wrap(
        self, name: str, model: nn.Module, sample: Mapping[str, Any], config: Any
    ) -> nn.Module:
        from torch_tensorrt.hf.exporters.patches.diffusion import (
            PI05PrefixKVStepEncoderPatch,
            StaticActionVelocityStepPatch,
        )
        from torch_tensorrt.hf.exporters.patches.language import (
            CausalLMPatch,
            language_decoder,
        )
        from torch_tensorrt.hf.exporters.patches.vision import GridVisionPatch

        core = _core(model)
        paligemma = core.paligemma_with_expert.paligemma.model
        if name == "vision":
            px = sample["pixel_values"]
            sample_px = px.float() if px.dtype != torch.float32 else px
            return GridVisionPatch(
                vision_model=paligemma.vision_tower.float(),
                projector=paligemma.multi_modal_projector,
                sample_pixel_values=sample_px,
                select_layer=-1,
                pixel_shuffle=False,
                downsample_ratio=0.5,
                force_float32_input=True,
            ).eval()
        if name == "language":
            language = paligemma.language_model
            lm_head = core.paligemma_with_expert.paligemma.lm_head
            return CausalLMPatch(language_decoder(language), lm_head).eval()
        if name == "action":
            return StaticActionVelocityStepPatch(
                step_encoder=PI05PrefixKVStepEncoderPatch(core),  # type: ignore[no-untyped-call]
                action_expert=core.paligemma_with_expert.gemma_expert.model,
                velocity_decoder=core.action_out_proj,
                output_tokens=int(core.config.chunk_size),
                cast_hidden_fp32=False,
            ).eval()
        raise KeyError(name)

    def prepare(
        self,
        name: str,
        model: nn.Module,
        sample: MutableMapping[str, Any],
        upstream: Mapping[str, Any],
        config: Any,
        module: nn.Module,
    ) -> ComponentBundle:
        from torch_tensorrt.hf.exporters.helpers.pi05 import (
            build_pi05_prefix_embs,
            make_pi05_suffix_position_and_mask,
            pi05_compact_index,
        )
        from torch_tensorrt.hf.exporters.plugin.attention import (
            ContextAttentionMaskType,
        )
        from torch_tensorrt.hf.exporters.plugin.plugin_utils import (
            patch_language_attention,
            patch_vision_attention,
        )

        core = _core(model)
        device = sample["pixel_values"].device
        dtype = sample["pixel_values"].dtype

        if name == "vision":
            px = sample["pixel_values"]
            seq_len = int(getattr(module, "seq_len", 1) or 1)
            batch = int(getattr(module, "batch_size", 1) or 1)

            def _patch(mod: nn.Module) -> Any:
                return patch_vision_attention(
                    mod.vision_model, batch_size=batch, seq_len=seq_len, name="SigLIP"
                )

            return ComponentBundle(
                trace_args=(px,),
                save_args=(_nchw_to_hwc(px),),
                input_names=["pixel_values"],
                output_names=["visual_embeds"],
                patch_fn=_patch,
                model_type="vit",
                engine_file="visual.engine",
            )

        if name == "language":
            paligemma = core.paligemma_with_expert.paligemma.model
            language = paligemma.language_model
            embs, pad, _attn, _pos = build_pi05_prefix_embs(
                core,
                sample["img_masks"],
                sample["tokens"],
                sample["masks"],
                upstream["visual_embeds"],
                sample["images"],
            )
            compact_len = int(embs.shape[1])
            vis = upstream["visual_embeds"]
            per_cam = int(sample["images"][0].shape[0])
            seq_per_image = int(
                vis.reshape(len(sample["images"]), per_cam, -1, vis.shape[-1]).shape[2]
            )
            sample["compact_index"] = pi05_compact_index(
                sample["img_masks"],
                sample["images"],
                seq_per_image,
                sample["masks"],
                device,
            )
            sample["prefix_pad_mask"] = pad
            max_seq_len = max(
                int(config.max_seq_len), compact_len + int(config.generation_reserve)
            )
            flat, meta = causal_lm_flat(
                language,
                embs.to(device=device, dtype=dtype),
                max_seq_len=max_seq_len,
                device=device,
                dtype=dtype,
                seq_len=compact_len,
            )
            sample.update(split_flat_to_kwargs(flat, meta["input_names"]))
            cfg = language.config
            head_dim = int(
                getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
            )

            def _patch(mod: nn.Module) -> Any:
                decoder = getattr(mod, "lm", mod)
                return patch_language_attention(
                    decoder,
                    hidden_size=int(cfg.hidden_size),
                    num_attention_heads=int(cfg.num_attention_heads),
                    num_key_value_heads=int(cfg.num_key_value_heads),
                    head_dim=head_dim,
                    context_attention_mask_type=ContextAttentionMaskType.PADDING,
                )

            return ComponentBundle(
                trace_args=flat,
                save_args=flat,
                input_names=meta["input_names"],
                output_names=["logits", "lm_hidden_states", "prefix_k", "prefix_v"],
                patch_fn=_patch,
                extra_config={"prefix_pad_mask_len": compact_len},
                model_type="language",
                engine_file="language.engine",
            )

        if name == "action":
            bsz = int(sample["lang_embeds"].shape[0])
            core_mod = _core(model)
            step_actions = sample.get("step_actions")
            if step_actions is None:
                step_actions = torch.randn(
                    bsz,
                    int(core_mod.config.chunk_size),
                    int(core_mod.config.max_action_dim),
                    device=device,
                    dtype=dtype,
                )
                sample["step_actions"] = step_actions
            step_timestep = sample.get(
                "step_timestep",
                torch.full((bsz,), 1.0, device=device, dtype=torch.float32),
            )
            sample["step_timestep"] = step_timestep
            prefix_k = upstream["prefix_k"].to(device=device, dtype=dtype)
            prefix_v = upstream["prefix_v"].to(device=device, dtype=dtype)
            pos, mask = make_pi05_suffix_position_and_mask(  # type: ignore[no-untyped-call]
                core_mod, sample["prefix_pad_mask"], step_actions, device
            )
            sample["suffix_position_ids"] = pos
            sample["suffix_attention_mask"] = mask
            args = (step_actions, step_timestep, prefix_k, prefix_v, pos, mask)
            return ComponentBundle(
                trace_args=args,
                save_args=args,
                input_names=[
                    "x_t",
                    "timestep",
                    "prefix_k",
                    "prefix_v",
                    "position_ids",
                    "attention_mask",
                ],
                output_names=["velocity"],
                model_type="action",
                engine_file="action.engine",
            )
        raise KeyError(name)

    def capture_upstream(
        self,
        name: str,
        outputs: Any,
        sample: Mapping[str, Any],
        bundle: ComponentBundle,
    ) -> dict[str, Any]:
        if name == "vision":
            vis = outputs[0] if isinstance(outputs, tuple) else outputs
            return {"visual_embeds": vis}
        if name == "language":
            return {"prefix_k": outputs[2], "prefix_v": outputs[3]}
        return {}

    def run(self, engines: Mapping[str, str], sample: Mapping[str, Any]) -> Any:
        vis = call_engine(engines["vision"], "vision", sample["pixel_values"])[0]
        prefix = fuse_prefix(vis, sample["lang_embeds"], sample["compact_index"])
        lm = call_engine(
            engines["language"],
            "language",
            prefix,
            sample["rope_rotary_cos_sin"],
            sample["context_lengths"],
            sample["kvcache_start_index"],
            sample["last_token_ids"],
            sample["ds_stack"],
            *kv_kwargs(sample),
        )
        return call_engine(
            engines["action"],
            "action",
            sample["step_actions"],
            sample["step_timestep"],
            lm[2],
            lm[3],
            sample["suffix_position_ids"],
            sample["suffix_attention_mask"],
        )
