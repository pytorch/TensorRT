from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import torch
import torch.nn as nn
from torch_tensorrt.hf.exporters.ops import call_engine, scatter_image_tokens
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


def _groot(model: nn.Module) -> nn.Module:
    if hasattr(model, "_groot_model"):
        return model._groot_model
    backbone = getattr(model, "backbone", None)
    if backbone is not None and hasattr(backbone, "eagle_model"):
        return model
    raise RuntimeError(
        "GR00T spec expected GrootPolicy or a module with backbone.eagle_model"
    )


@register_edge_spec("groot", "gr00t")
class GrootSpec(EdgeSpec):  # type: ignore[misc]
    components = ("vision", "language", "context_projection", "action")

    def prepare_sample_inputs(
        self, model: nn.Module, raw: Mapping[str, Any], config: Any
    ) -> MutableMapping[str, Any]:
        if "pixel_values" in raw and "input_ids" in raw:
            return dict(raw)

        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.groot.processor_groot import GrootEagleEncodeStep
        from torch_tensorrt.hf.exporters.data import (
            create_pil_messages,
            load_test_data,
            pack_state,
        )
        from torch_tensorrt.hf.exporters.helpers.groot import make_embodiment_id

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

    def wrap(
        self, name: str, model: nn.Module, sample: Mapping[str, Any], config: Any
    ) -> nn.Module:
        from torch_tensorrt.hf.exporters.patches.diffusion import (
            GrootDiTStepEncoderPatch,
            StaticActionVelocityStepPatch,
            TRTDynamicCategorySpecificMLPPatch,
        )
        from torch_tensorrt.hf.exporters.patches.language import (
            CausalLMPatch,
            ContextProjectionPatch,
            language_decoder,
        )
        from torch_tensorrt.hf.exporters.patches.vision import GridVisionPatch

        found = _groot(model)
        eagle = found.backbone.eagle_model
        device = sample["pixel_values"].device
        dtype = sample["pixel_values"].dtype
        # GR00T loads DiT / category MLPs in bf16. Export tensors are fp16.
        # The e2e path casts the whole wrapper; without that, sa_embs is
        # float32 against bf16 attn.to_q weights.
        if name == "vision":
            module = GridVisionPatch(
                vision_model=eagle.vision_model,
                projector=eagle.mlp1,
                sample_pixel_values=sample["pixel_values"],
                select_layer=eagle.select_layer,
                pixel_shuffle=eagle.use_pixel_shuffle,
                downsample_ratio=getattr(eagle, "downsample_ratio", 0.5),
                vision_kwargs={},
            )
        elif name == "language":
            language = eagle.language_model
            module = CausalLMPatch(
                language_decoder(language), language.lm_head, select_layer=-1
            )
        elif name == "context_projection":
            module = ContextProjectionPatch(  # type: ignore[no-untyped-call]
                found.backbone.eagle_linear,
                found.action_head.vlln,
                found.action_head.vl_self_attention,
            )
        elif name == "action":
            module = StaticActionVelocityStepPatch(
                step_encoder=GrootDiTStepEncoderPatch(
                    found.action_head, sample.get("embodiment_id")
                ),
                action_expert=found.action_head.model,
                velocity_decoder=TRTDynamicCategorySpecificMLPPatch(
                    found.action_head.action_decoder
                ),
                output_tokens=int(found.action_head.config.action_horizon),
                cast_hidden_fp32=False,
            )
        else:
            raise KeyError(name)
        return module.eval().to(device=device, dtype=dtype)

    def prepare(
        self,
        name: str,
        model: nn.Module,
        sample: MutableMapping[str, Any],
        upstream: Mapping[str, Any],
        config: Any,
        module: nn.Module,
    ) -> ComponentBundle:
        from torch_tensorrt.hf.exporters.plugin.attention import (
            ContextAttentionMaskType,
        )
        from torch_tensorrt.hf.exporters.plugin.plugin_utils import (
            patch_language_attention,
            patch_vision_attention,
        )

        found = _groot(model)
        eagle = found.backbone.eagle_model
        device = sample["pixel_values"].device
        dtype = sample["pixel_values"].dtype

        if name == "vision":
            px = sample["pixel_values"]
            seq_len = int(getattr(module, "seq_len", 1) or 1)
            batch = int(getattr(module, "batch_size", 1) or 1)

            def _patch(mod: nn.Module) -> Any:
                vision = getattr(mod.vision_model, "vision_model", mod.vision_model)
                return patch_vision_attention(
                    vision, batch_size=batch, seq_len=seq_len, name="SigLIP"
                )

            return ComponentBundle(
                trace_args=(px,),
                save_args=(px,),
                input_names=["pixel_values"],
                output_names=["visual_embeds"],
                patch_fn=_patch,
                model_type="vit",
                engine_file="visual.engine",
            )

        if name == "language":
            language = eagle.language_model
            input_ids = sample["input_ids"]
            input_embs = language.get_input_embeddings()(input_ids)
            image_token_index = getattr(
                eagle, "image_token_index", eagle.config.image_token_index
            )
            mask = input_ids == image_token_index
            sample["image_token_mask"] = mask
            vis = upstream["visual_embeds"]
            hidden = input_embs.shape[-1]
            flat = input_embs.clone().reshape(-1, hidden)
            vis_flat = vis.reshape(-1, hidden).to(device=flat.device, dtype=flat.dtype)
            n = int(mask.reshape(-1).sum().item())
            flat[mask.reshape(-1)] = vis_flat[:n]
            inputs_embeds = (
                flat.reshape_as(input_embs).to(device=device, dtype=dtype).contiguous()
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
                    context_attention_mask_type=ContextAttentionMaskType.CAUSAL,
                )

            return ComponentBundle(
                trace_args=packed,
                save_args=packed,
                input_names=meta["input_names"],
                output_names=["logits", "lm_hidden_states", "prefix_k", "prefix_v"],
                patch_fn=_patch,
                model_type="language",
                engine_file="language.engine",
            )

        if name == "context_projection":
            hidden = upstream["lm_hidden"].to(dtype=dtype)
            return ComponentBundle(
                trace_args=(hidden,),
                save_args=(hidden,),
                input_names=["lm_hidden_states"],
                output_names=["vl_embs"],
                model_type="context_projection",
                engine_file="context_projection.engine",
            )

        if name == "action":
            bsz = int(upstream["context_embs"].shape[0])
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
            args = (
                step_actions,
                step_timestep,
                upstream["context_embs"].to(device=device, dtype=dtype),
                sample["state"],
                sample["embodiment_id"],
            )
            return ComponentBundle(
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
            return {"lm_hidden": outputs[1]}
        if name == "context_projection":
            ctx = outputs[0] if isinstance(outputs, tuple) else outputs
            return {"context_embs": ctx}
        return {}

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
