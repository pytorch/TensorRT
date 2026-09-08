from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import torch
import torch.nn as nn
import torch_tensorrt

from ...ops import call_engine, fuse_prefix
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
from ..common.patches import language_decoder
from .helpers import (
    build_pi05_prefix_embs,
    make_pi05_suffix_position_and_mask,
    pi05_compact_index,
)
from .patches import PI05


@register_edge_spec("pi05")
class Pi05Spec(EdgeSpec):  # type: ignore[misc]
    def apply_patches(self, model=None):
        """Install vision, language, and action setattr replacements."""
        del model
        from ...plugin.attn_patches import apply_patches

        return apply_patches(PI05)

    def create_dynamic_shapes(
        self,
        input_names: list[str],
        trace_args: tuple[Any, ...],
        *,
        max_seq_len: int,
    ) -> tuple[Any, ...]:
        """Prefill/decode ``torch_tensorrt.Input`` specs (e2e language dual-profile)."""
        named = dict(zip(input_names, trace_args))
        embs = named["inputs_embeds"]
        ds = named["ds_stack"]
        kv = next(
            tensor
            for name, tensor in zip(input_names, trace_args)
            if name.startswith("past_key_values_")
        )
        bsz = int(embs.shape[0])
        hidden = int(embs.shape[-1])
        opt_prefill = max(int(max_seq_len) // 2, 1)
        num_ds = int(ds.shape[0])
        num_kv = int(kv.shape[2])
        head_dim = int(kv.shape[-1])
        prefill_profile = {
            "min_shape": (1, 1, hidden),
            "opt_shape": (bsz, opt_prefill, hidden),
            "max_shape": (bsz, max_seq_len, hidden),
        }
        decode_profile = {
            "min_shape": (1, 1, hidden),
            "opt_shape": (bsz, 1, hidden),
            "max_shape": (bsz, 1, hidden),
        }
        kv_profile = {
            "min_shape": (1, 2, num_kv, 1, head_dim),
            "opt_shape": (bsz, 2, num_kv, max_seq_len, head_dim),
            "max_shape": (bsz, 2, num_kv, max_seq_len, head_dim),
        }
        ds_prefill = {
            "min_shape": (num_ds, 1, 1, hidden),
            "opt_shape": (num_ds, bsz, opt_prefill, hidden),
            "max_shape": (num_ds, bsz, max_seq_len, hidden),
        }
        ds_decode = {
            "min_shape": (num_ds, 1, 1, hidden),
            "opt_shape": (num_ds, bsz, 1, hidden),
            "max_shape": (num_ds, bsz, 1, hidden),
        }
        input_specs = []
        for name, tensor in zip(input_names, trace_args):
            if name == "inputs_embeds":
                input_specs.append(
                    torch_tensorrt.Input(
                        profiles=[prefill_profile, decode_profile],
                        shared_dims={1: "seq_len"},
                        dtype=tensor.dtype,
                        format=torch.contiguous_format,
                        name=name,
                    )
                )
            elif name == "ds_stack":
                input_specs.append(
                    torch_tensorrt.Input(
                        profiles=[ds_prefill, ds_decode],
                        shared_dims={2: "seq_len"},
                        dtype=tensor.dtype,
                        format=torch.contiguous_format,
                        name=name,
                    )
                )
            elif name.startswith("past_key_values_"):
                input_specs.append(
                    torch_tensorrt.Input(
                        profiles=[kv_profile, kv_profile],
                        dtype=tensor.dtype,
                        format=torch.contiguous_format,
                        name=name,
                    )
                )
            else:
                input_specs.append(
                    torch_tensorrt.Input(
                        shape=tuple(tensor.shape),
                        dtype=tensor.dtype,
                        format=torch.contiguous_format,
                        name=name,
                    )
                )
        return tuple(input_specs)

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

        from ...data import (
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
        core = policy if hasattr(policy, "paligemma_with_expert") else policy.model
        lang_embeds = core.paligemma_with_expert.embed_language_tokens(tokens)
        return {
            "pixel_values": pixel_values,
            "images": images,
            "img_masks": img_masks,
            "tokens": tokens,
            "masks": masks,
            "lang_embeds": lang_embeds.to(device=device, dtype=dtype).contiguous(),
        }

    def capture_eager_outputs(
        self, model, sample, config, bench=None
    ) -> dict[str, torch.Tensor]:
        del config
        from lerobot.policies.pi05.modeling_pi05 import create_sinusoidal_pos_embedding

        from ...measure import cuda_ms
        from ...prefix_cache import PrefixKVCache

        core = model if hasattr(model, "paligemma_with_expert") else model.model
        paligemma = core.paligemma_with_expert.paligemma.model
        language = paligemma.language_model
        px = sample["pixel_values"]

        with torch.no_grad():
            tower_out = paligemma.vision_tower(px)
            hidden = getattr(tower_out, "last_hidden_state", tower_out)
            visual_embeds = paligemma.multi_modal_projector(hidden)
            lm_dtype = next(language.parameters()).dtype
            prefix_embs = sample["prefix_embs"].to(dtype=lm_dtype)
            lm = language(
                inputs_embeds=prefix_embs,
                attention_mask=sample["prefix_attention_mask"],
                position_ids=sample["prefix_position_ids"],
                return_dict=True,
            )
            suffix_embs = core.action_in_proj(sample["step_actions"])
            time_emb = create_sinusoidal_pos_embedding(
                sample["step_timestep"],
                core.action_in_proj.out_features,
                min_period=core.config.min_period,
                max_period=core.config.max_period,
                device=sample["step_timestep"].device,
            ).to(dtype=suffix_embs.dtype)
            adarms_cond = torch.nn.functional.silu(
                core.time_mlp_out(torch.nn.functional.silu(core.time_mlp_in(time_emb)))
            )
            expert_out = core.paligemma_with_expert.gemma_expert.model(
                inputs_embeds=suffix_embs,
                attention_mask=sample["suffix_attention_mask"],
                position_ids=sample["suffix_position_ids"],
                past_key_values=PrefixKVCache(sample["prefix_k"], sample["prefix_v"]),
                use_cache=False,
                adarms_cond=adarms_cond,
            )
            action_hidden = (
                expert_out.last_hidden_state
                if hasattr(expert_out, "last_hidden_state")
                else expert_out
            )
            if isinstance(action_hidden, (tuple, list)):
                action_hidden = action_hidden[0]
            velocity = core.action_out_proj(
                action_hidden[:, -int(core.config.chunk_size) :]
            )

        if bench is not None:
            bench["vision"] = cuda_ms(
                lambda: paligemma.multi_modal_projector(
                    paligemma.vision_tower(px).last_hidden_state
                )
            )
            bench["language"] = cuda_ms(
                lambda: language(
                    inputs_embeds=prefix_embs,
                    attention_mask=sample["prefix_attention_mask"],
                    position_ids=sample["prefix_position_ids"],
                    return_dict=True,
                )
            )
            bench["action"] = cuda_ms(
                lambda: core.action_out_proj(
                    core.paligemma_with_expert.gemma_expert.model(
                        inputs_embeds=suffix_embs,
                        attention_mask=sample["suffix_attention_mask"],
                        position_ids=sample["suffix_position_ids"],
                        past_key_values=PrefixKVCache(
                            sample["prefix_k"], sample["prefix_v"]
                        ),
                        use_cache=False,
                        adarms_cond=adarms_cond,
                    ).last_hidden_state[:, -int(core.config.chunk_size) :]
                )
            )
        return {
            "vision": visual_embeds,
            "language": lm.last_hidden_state,
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

        core = model if hasattr(model, "paligemma_with_expert") else model.model
        paligemma = core.paligemma_with_expert.paligemma.model
        language = paligemma.language_model
        px = sample["pixel_values"]
        device = px.device
        dtype = px.dtype

        vision = ComponentBundle(
            module=paligemma.eval(),
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

        with torch.no_grad():
            tower_out = paligemma.vision_tower(px)
            hidden = getattr(tower_out, "last_hidden_state", tower_out)
            visual_embeds = paligemma.multi_modal_projector(hidden)
        if visual_embeds.ndim == 3:
            visual_embeds = visual_embeds.reshape(-1, visual_embeds.shape[-1])

        embs, pad, attn, pos = build_pi05_prefix_embs(
            core,
            sample["img_masks"],
            sample["tokens"],
            sample["masks"],
            visual_embeds,
            sample["images"],
        )
        compact_len = int(embs.shape[1])
        per_cam = int(sample["images"][0].shape[0])
        seq_per_image = int(
            visual_embeds.reshape(
                len(sample["images"]), per_cam, -1, visual_embeds.shape[-1]
            ).shape[2]
        )
        sample["compact_index"] = pi05_compact_index(
            sample["img_masks"],
            sample["images"],
            seq_per_image,
            sample["masks"],
            device,
        )
        sample["prefix_embs"] = embs
        sample["prefix_pad_mask"] = pad
        sample["prefix_attention_mask"] = attn
        sample["prefix_position_ids"] = pos
        if int(config.generation_reserve) < 0:
            raise ValueError("generation_reserve must be non-negative")
        max_seq_len = max(
            int(config.max_seq_len),
            compact_len + int(config.generation_reserve),
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

        embs_t, rope, ctx, kv_start, last, ds, *kvs = flat
        opt_prefill = max(max_seq_len // 2, 1)
        trace_len = min(int(embs_t.shape[1]), opt_prefill)
        trace_args = (
            embs_t[:, :trace_len].contiguous(),
            rope,
            torch.full_like(ctx, trace_len),
            kv_start,
            torch.full_like(last, trace_len - 1),
            ds[:, :, :trace_len].contiguous(),
            *kvs,
        )

        decoder = language_decoder(language)
        language_bundle = ComponentBundle(
            module=decoder.eval(),
            trace_args=trace_args,
            save_args=flat,
            execute_args=flat,
            input_specs=self.create_dynamic_shapes(
                meta["input_names"], trace_args, max_seq_len=max_seq_len
            ),
            input_names=meta["input_names"],
            output_names=["logits", "lm_hidden_states", "prefix_k", "prefix_v"],
            parity_output="lm_hidden_states",
            context_attention_mask_type=int(ContextAttentionMaskType.PADDING),
            extra_config={"prefix_pad_mask_len": compact_len},
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

        bsz = int(sample["lang_embeds"].shape[0])
        step_actions = sample.get("step_actions")
        if step_actions is None:
            step_actions = torch.randn(
                bsz,
                int(core.config.chunk_size),
                int(core.config.max_action_dim),
                device=device,
                dtype=dtype,
            )
            sample["step_actions"] = step_actions
        step_timestep = sample.get(
            "step_timestep",
            torch.full((bsz,), 1.0, device=device, dtype=torch.float32),
        )
        sample["step_timestep"] = step_timestep
        cfg = language.config
        num_kv = int(cfg.num_key_value_heads)
        head_dim = int(
            getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        )
        prefix_k = torch.zeros(
            len(decoder.layers),
            bsz,
            num_kv,
            compact_len,
            head_dim,
            device=device,
            dtype=dtype,
        )
        prefix_v = torch.zeros_like(prefix_k)
        sample["prefix_k"] = prefix_k
        sample["prefix_v"] = prefix_v
        pos, mask = make_pi05_suffix_position_and_mask(  # type: ignore[no-untyped-call]
            core, sample["prefix_pad_mask"], step_actions, device
        )
        sample["suffix_position_ids"] = pos
        sample["suffix_attention_mask"] = mask
        args = (step_actions, step_timestep, prefix_k, prefix_v, pos, mask)
        action = ComponentBundle(
            module=core.eval(),
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
            trt_settings={
                "disable_tf32": True,
                "use_fp32_acc": True,
                "use_explicit_typing": True,
                "decompose_attention": True,
            },
        )
        return {"vision": vision, "language": language_bundle, "action": action}

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
