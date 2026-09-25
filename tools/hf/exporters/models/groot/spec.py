from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Mapping, MutableMapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ...ops import call_engine, scatter_image_tokens
from ...rope import export_rope_fields
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
from .patches import GROOT, _patch_eagle_image_features


def _write_groot_language_artifacts(
    sample: Mapping[str, Any],
    language: nn.Module,
    out_dir: Path,
) -> None:
    from safetensors.torch import save_file
    from tensorrt_edgellm.chat_template import (
        process_chat_template,
        write_fallback_processed_chat_template,
    )

    processor = sample["_groot_processor"]
    processor.save_pretrained(out_dir)
    decoder = getattr(language, "model", language)
    embedding = decoder.embed_tokens.weight.detach().to("cpu").contiguous()
    save_file({"embedding": embedding}, out_dir / "embedding.safetensors")

    with tempfile.TemporaryDirectory() as template_source:
        processor.save_pretrained(template_source)
        process_chat_template(template_source, str(out_dir))
    if not (out_dir / "processed_chat_template.json").is_file():
        write_fallback_processed_chat_template(str(out_dir), str(out_dir))
    template_path = out_dir / "processed_chat_template.json"
    template = json.loads(template_path.read_text())
    template["content_types"] = {
        "image": {"format": "<IMG_CONTEXT>"},
        "video": {"format": "<video>"},
    }
    template_path.write_text(json.dumps(template, indent=2) + "\n")


def _edge_language_input_specs(
    input_names: list[str],
    trace_args: tuple[Any, ...],
    *,
    max_seq_len: int,
) -> tuple[Any, ...]:
    import torch_tensorrt

    named = dict(zip(input_names, trace_args))
    embeds = named["inputs_embeds"]
    batch_size = int(embeds.shape[0])
    hidden_size = int(embeds.shape[-1])
    prompt_len = int(embeds.shape[1])
    specs = []
    for name, tensor in zip(input_names, trace_args):
        profiles = None
        if name == "inputs_embeds":
            profiles = [
                {
                    "min_shape": (1, 1, hidden_size),
                    "opt_shape": (batch_size, prompt_len, hidden_size),
                    "max_shape": (batch_size, max_seq_len, hidden_size),
                },
                {
                    "min_shape": (1, 1, hidden_size),
                    "opt_shape": (batch_size, 1, hidden_size),
                    "max_shape": (batch_size, 1, hidden_size),
                },
            ]
        elif name == "kvcache_start_index":
            profiles = [
                {
                    "min_shape": (0,),
                    "opt_shape": (batch_size,),
                    "max_shape": (batch_size,),
                },
                {
                    "min_shape": (batch_size,),
                    "opt_shape": (batch_size,),
                    "max_shape": (batch_size,),
                },
            ]

        if profiles is None:
            specs.append(
                torch_tensorrt.Input(
                    shape=tuple(tensor.shape),
                    dtype=tensor.dtype,
                    format=torch.contiguous_format,
                    name=name,
                )
            )
        else:
            specs.append(
                torch_tensorrt.Input(
                    profiles=profiles,
                    shared_dims={1: "seq_len"} if name == "inputs_embeds" else None,
                    dtype=tensor.dtype,
                    format=torch.contiguous_format,
                    name=name,
                )
            )
    return tuple(specs)


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
    @contextmanager
    def apply_patches(self, model=None):
        from ...plugin.attn_patches import apply_patches, patch_attribute
        from .helpers import _groot

        with apply_patches(GROOT):
            if model is None:
                yield
                return
            eagle_cls = type(_groot(model).backbone.eagle_model)
            with patch_attribute(eagle_cls, "forward", _patch_eagle_image_features):
                yield

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
            "_groot_processor": proc,
        }

    def capture_eager_outputs(
        self, model, sample, config, bench=None
    ) -> dict[str, torch.Tensor]:
        del config
        from ...measure import cuda_ms

        found = _groot(model)
        eagle = found.backbone.eagle_model
        language = _causal_lm(eagle.language_model)
        decoder = getattr(language, "model", language)
        px = sample["pixel_values"]
        lm_hidden = sample["lm_hidden"]
        action_head = found.action_head

        with torch.no_grad():
            visual_embeds = eagle.extract_feature(px)
            lm = decoder(
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
                lambda: decoder(
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

        import torch_tensorrt

        num_blocks = int(px.shape[0])
        vision_config = eagle.vision_model.config.to_dict()
        vision_config["patch_size"] = [
            int(eagle.vision_model.config.patch_size),
            int(eagle.vision_model.config.patch_size),
        ]
        vision_config["image_size"] = [
            int(eagle.vision_model.config.image_size),
            int(eagle.vision_model.config.image_size),
        ]
        text_config = _causal_lm(eagle.language_model).config.to_dict()
        tokenizer = sample["_groot_processor"].tokenizer
        vision = ComponentBundle(
            module=_export_module(eagle, sample),
            trace_args=(px,),
            save_args=(px,),
            input_specs=(
                torch_tensorrt.Input(
                    min_shape=(1, *tuple(px.shape[1:])),
                    opt_shape=tuple(px.shape),
                    max_shape=tuple(px.shape),
                    dtype=px.dtype,
                    format=torch.contiguous_format,
                    name="input",
                ),
            ),
            input_names=["input"],
            output_names=["output"],
            model_type="internvl",
            engine_file="visual.engine",
            output_subdir="visual",
            edge_runtime_bindings=True,
            extra_config={
                "image_token_id": int(
                    getattr(eagle, "image_token_index", eagle.config.image_token_index)
                ),
                "img_start_token_id": int(tokenizer.convert_tokens_to_ids("<img>")),
                "img_end_token_id": int(tokenizer.convert_tokens_to_ids("</img>")),
                "text_config": text_config,
                "vision_config": vision_config,
                "builder_config": {
                    "min_image_tokens": 256,
                    "max_image_tokens": num_blocks * 256,
                    "max_image_tokens_per_image": 256,
                },
            },
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
        max_seq_len = math.ceil(max_seq_len / 128) * 128
        packed_base, meta = causal_lm_flat(
            language,
            inputs_embeds,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        bsz, seq_len, hidden_size = inputs_embeds.shape
        num_layers = int(meta["num_layers"])
        max_pages_per_seq = max_seq_len // 128
        num_pages = int(bsz) * max_pages_per_seq
        kv_pool_shape = (
            2,
            num_pages,
            128,
            int(meta["num_key_value_heads"]),
            int(meta["head_dim"]),
        )
        paged_kvs = tuple(
            torch.zeros(kv_pool_shape, device=device, dtype=dtype)
            for _ in range(num_layers)
        )
        k_pages = torch.arange(
            num_pages,
            device=device,
            dtype=torch.int32,
        ).reshape(int(bsz), max_pages_per_seq)
        kv_page_table = torch.stack((k_pages, k_pages + num_pages), dim=1)
        language_input_names = [
            "inputs_embeds",
            "rope_rotary_cos_sin",
            "context_lengths",
            "kvcache_start_index",
            "last_token_ids",
            "kv_page_table",
            *[f"past_key_values_{i}" for i in range(num_layers)],
        ]
        packed = (
            packed_base[0],
            packed_base[1],
            packed_base[2],
            torch.zeros(int(bsz), device=device, dtype=torch.int32),
            packed_base[4],
            kv_page_table,
            *paged_kvs,
        )
        sample.update(split_flat_to_kwargs(packed, language_input_names))

        language_config = language.config
        rope_fields = export_rope_fields(language_config.to_dict())

        language_bundle = ComponentBundle(
            module=_export_module(language, sample),
            trace_args=packed,
            save_args=packed,
            input_specs=_edge_language_input_specs(
                language_input_names,
                packed,
                max_seq_len=max_seq_len,
            ),
            input_names=language_input_names,
            output_names=[
                "logits",
                "accept_hidden_states",
                *[f"present_key_values_{i}" for i in range(num_layers)],
            ],
            parity_output="accept_hidden_states",
            context_attention_mask_type=int(ContextAttentionMaskType.CAUSAL),
            extra_config={"context_mask_selector_enabled": True},
            model_type="language",
            engine_file="llm.engine",
            output_subdir="",
            edge_runtime_bindings=True,
            artifact_writer=lambda out_dir: _write_groot_language_artifacts(
                sample,
                language,
                out_dir,
            ),
            extra_config={
                "engine_role": "llm",
                "model": str(language_config.model_type),
                "num_hidden_layers": num_layers,
                "num_attention_heads": int(language_config.num_attention_heads),
                "num_key_value_heads": int(language_config.num_key_value_heads),
                "head_dim": int(meta["head_dim"]),
                "hidden_size": int(language_config.hidden_size),
                "intermediate_size": int(language_config.intermediate_size),
                "vocab_size": int(language_config.vocab_size),
                "max_position_embeddings": int(language_config.max_position_embeddings),
                "rope_theta": float(rope_fields["rope_theta"]),
                "rope_scaling": rope_fields["rope_scaling"],
                "partial_rotary_factor": float(
                    getattr(language_config, "partial_rotary_factor", 1.0)
                ),
                "kv_cache_dtype": "fp16",
                "num_deepstack_features": 0,
                "image_token_id": int(
                    getattr(eagle, "image_token_index", eagle.config.image_token_index)
                ),
                "spec_decode_type": "none",
                "accept_hidden_layer": int(found.backbone.select_layer),
                "builder_config": {
                    "max_batch_size": int(bsz),
                    "max_input_len": int(seq_len),
                    "max_kv_cache_capacity": max_seq_len,
                    "max_kv_pool_pages": num_pages,
                    "max_lora_rank": 0,
                    "spec_base": False,
                    "spec_draft": False,
                    "trt_native_ops": False,
                },
            },
            trt_settings={
                "disable_tf32": True,
                "use_fp32_acc": True,
                "use_explicit_typing": True,
                "decompose_attention": True,
                "assume_dynamic_shape_support": True,
                "offload_module_to_cpu": True,
            },
        )

        lm_hidden = torch.zeros(bsz, seq_len, hidden_size, device=device, dtype=dtype)
        sample["lm_hidden"] = lm_hidden

        context_input_specs = (
            torch_tensorrt.Input(
                min_shape=(1, 1, hidden_size),
                opt_shape=(int(bsz), int(seq_len), hidden_size),
                max_shape=(int(bsz), max_seq_len, hidden_size),
                dtype=dtype,
                format=torch.contiguous_format,
                name="lm_hidden_states",
            ),
        )
        context_projection = ComponentBundle(
            module=_export_module(found, sample),
            trace_args=(lm_hidden,),
            save_args=(lm_hidden,),
            input_specs=context_input_specs,
            input_names=["lm_hidden_states"],
            output_names=["vl_embs"],
            model_type="context_projection",
            engine_file="context_projection.engine",
            edge_runtime_bindings=True,
            trt_settings={
                "disable_tf32": True,
                "use_fp32_acc": True,
                "use_explicit_typing": True,
                "decompose_attention": True,
            },
        )

        out_dim = int(
            getattr(
                found.backbone.eagle_linear,
                "out_features",
                found.action_head.config.backbone_embedding_dim,
            )
        )
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
        action_input_specs = tuple(
            torch_tensorrt.Input(
                **(
                    {
                        "min_shape": (1, 1, out_dim),
                        "opt_shape": (int(bsz), int(seq_len), out_dim),
                        "max_shape": (int(bsz), max_seq_len, out_dim),
                    }
                    if name == "context_embs"
                    else {"shape": tuple(tensor.shape)}
                ),
                dtype=tensor.dtype,
                format=torch.contiguous_format,
                name=name,
            )
            for name, tensor in zip(
                [
                    "actions",
                    "timestep",
                    "context_embs",
                    "state",
                    "embodiment_id",
                ],
                args,
            )
        )
        action = ComponentBundle(
            module=_export_module(found.action_head, sample),
            trace_args=args,
            save_args=args,
            input_specs=action_input_specs,
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
            output_subdir="groot_action",
            edge_runtime_bindings=True,
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
            sample["context_mask_selector"],
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
