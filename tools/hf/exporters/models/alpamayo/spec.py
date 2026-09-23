from __future__ import annotations

import math
import shutil
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...ops import call_engine, scatter_image_tokens
from ...quantization import FP8CheckpointLinear
from ...rope import export_rope_fields
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
    prepare_edge_visual_inputs,
    prepare_fixed_grid_vision,
    scatter_visual_tokens,
    stack_deepstack_features,
    unpack_visual_output,
)
from .patches import ALPAMAYO


def _write_visual_processor(model: nn.Module, out_dir: Path) -> None:
    from alpamayo1_5 import helper

    processor = helper.get_processor(model.tokenizer)
    processor.save_pretrained(out_dir)
    processor_config = out_dir / "processor_config.json"
    preprocessor_config = out_dir / "preprocessor_config.json"
    if not preprocessor_config.is_file() and processor_config.is_file():
        shutil.copy2(processor_config, preprocessor_config)


def _write_language_artifacts(
    model: nn.Module,
    language: nn.Module,
    out_dir: Path,
) -> None:
    from safetensors.torch import save_file

    model.tokenizer.save_pretrained(out_dir)
    embedding = language.embed_tokens.weight.detach().to("cpu").contiguous()
    save_file({"embedding": embedding}, out_dir / "embedding.safetensors")
    try:
        import tempfile

        from tensorrt_edgellm.chat_template import (
            process_chat_template,
            write_fallback_processed_chat_template,
        )

        with tempfile.TemporaryDirectory() as template_source:
            template_source_path = Path(template_source)
            model.tokenizer.save_pretrained(template_source_path)
            checkpoint_config = Path(model._edge_checkpoint_dir) / "config.json"
            if checkpoint_config.is_file():
                shutil.copy2(
                    checkpoint_config,
                    template_source_path / "config.json",
                )
            process_chat_template(str(template_source_path), str(out_dir))
        if not (out_dir / "processed_chat_template.json").is_file():
            write_fallback_processed_chat_template(
                str(model._edge_checkpoint_dir),
                str(out_dir),
            )
    except ImportError as exc:
        raise ImportError(
            "TensorRT-Edge-LLM is required to generate " "processed_chat_template.json"
        ) from exc


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
        if name == "inputs_embeds" or name.startswith("deepstack_embeds_"):
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
            shared_dims = (
                {1: "seq_len"}
                if name == "inputs_embeds" or name.startswith("deepstack_embeds_")
                else None
            )
            specs.append(
                torch_tensorrt.Input(
                    profiles=profiles,
                    shared_dims=shared_dims,
                    dtype=tensor.dtype,
                    format=torch.contiguous_format,
                    name=name,
                )
            )
    return tuple(specs)


def _export_module(module: nn.Module, device: torch.device, dtype: torch.dtype):
    if any(isinstance(child, FP8CheckpointLinear) for child in module.modules()):
        return module.eval().to(device=device)
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
        vlm = alpamayo_vlm(model)
        px = sample["pixel_values"]
        grid = sample["image_grid_thw"]

        with torch.no_grad():
            visual_embeds, deepstack = unpack_visual_output(visual(px, grid))
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
            token_indices = sample["last_token_ids"].squeeze(-1)
            last_hidden = language_out.last_hidden_state[
                torch.arange(
                    language_out.last_hidden_state.shape[0],
                    device=language_out.last_hidden_state.device,
                    dtype=torch.long,
                ),
                token_indices,
            ]
            logits = vlm.lm_head(last_hidden).float()

            with self.apply_patches(model):
                denoised = model(*sample["edge_action_args"])[0]

        if bench is not None:
            bench["vision"] = cuda_ms(lambda: unpack_visual_output(visual(px, grid))[0])
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
            "language": logits,
            "action": denoised,
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
        visual_args = prepare_edge_visual_inputs(visual, px, grid)
        num_deepstack_features = len(visual.deepstack_visual_indexes)
        visual_config = visual.config.to_dict()
        text_config = language.config.to_dict()
        rope_fields = export_rope_fields(text_config)
        text_config["rope_theta"] = float(rope_fields["rope_theta"])
        text_config["rope_scaling"] = rope_fields["rope_scaling"]
        patch_area = (
            int(visual.config.patch_size) ** 2
            * int(visual.config.spatial_merge_size) ** 2
        )
        min_image_tokens = int(model.config.min_pixels) // patch_area
        max_image_tokens_per_image = int(model.config.max_pixels) // patch_area
        max_image_tokens = int(px.shape[0]) // (
            int(visual.config.spatial_merge_size) ** 2
        )
        vision = ComponentBundle(
            module=_export_module(visual, device, dtype),
            trace_args=visual_args,
            save_args=visual_args,
            input_names=[
                "input",
                "rotary_pos_emb",
                "cu_seqlens",
                "fast_pos_embed_idx",
                "fast_pos_embed_weight",
                "max_seqlen_carrier",
            ],
            output_names=[
                "output",
                *[
                    f"deepstack_features_{index}"
                    for index in range(num_deepstack_features)
                ],
            ],
            parity_output="output",
            model_type="qwen3_vl",
            engine_file="visual.engine",
            edge_runtime_bindings=True,
            extra_config={
                "vision_start_token_id": int(vlm.config.vision_start_token_id),
                "vision_end_token_id": int(vlm.config.vision_end_token_id),
                "image_token_id": int(vlm.config.image_token_id),
                "video_token_id": int(vlm.config.video_token_id),
                "vocab_size": int(language.config.vocab_size),
                "text_config": text_config,
                "vision_config": visual_config,
                "rope_theta": float(rope_fields["rope_theta"]),
                "rope_scaling": rope_fields["rope_scaling"],
                "builder_config": {
                    "min_image_tokens": min_image_tokens,
                    "max_image_tokens": max_image_tokens,
                    "max_image_tokens_per_image": max_image_tokens_per_image,
                    "use_trt_native_vit_attn": False,
                },
            },
            artifact_writer=lambda out_dir: _write_visual_processor(model, out_dir),
            trt_settings={
                "disable_tf32": False,
                "use_fp32_acc": False,
                "use_explicit_typing": True,
                "decompose_attention": True,
            },
        )

        with torch.no_grad():
            visual_embeds, deepstack = unpack_visual_output(visual(px, grid))
            deepstack = stack_deepstack_features(deepstack)
            text_embeds = language.embed_tokens(sample["input_ids"])

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
        max_seq_len = math.ceil(max_seq_len / 128) * 128
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
        num_layers = int(meta["num_layers"])
        num_deepstack = min(
            len(visual.deepstack_visual_indexes),
            int(ds_stack.shape[0]),
        )
        batch_size = int(inputs_embeds.shape[0])
        max_pages_per_seq = max_seq_len // 128
        num_pages = batch_size * max_pages_per_seq
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
        ).reshape(batch_size, max_pages_per_seq)
        kv_page_table = torch.stack((k_pages, k_pages + num_pages), dim=1)
        deepstack_inputs = tuple(ds_stack[index] for index in range(num_deepstack))
        language_input_names = [
            "inputs_embeds",
            "rope_rotary_cos_sin",
            "context_lengths",
            "kvcache_start_index",
            "last_token_ids",
            *[f"deepstack_embeds_{i}" for i in range(num_deepstack)],
            "kv_page_table",
            *[f"past_key_values_{i}" for i in range(num_layers)],
        ]
        flat = (
            flat[0],
            flat[1],
            flat[2],
            torch.zeros(batch_size, device=device, dtype=torch.int32),
            flat[4],
            *deepstack_inputs,
            kv_page_table,
            *paged_kvs,
        )
        sample.update(split_flat_to_kwargs(flat, language_input_names))

        language_bundle = ComponentBundle(
            module=_export_module(vlm, device, dtype),
            trace_args=flat,
            save_args=flat,
            input_specs=_edge_language_input_specs(
                language_input_names,
                flat,
                max_seq_len=max_seq_len,
            ),
            input_names=language_input_names,
            output_names=[
                "logits",
                *[f"present_key_values_{i}" for i in range(num_layers)],
            ],
            context_attention_mask_type=int(ContextAttentionMaskType.CAUSAL),
            model_type="language",
            engine_file="llm.engine",
            output_subdir="",
            edge_runtime_bindings=True,
            artifact_writer=lambda out_dir: _write_language_artifacts(
                model,
                language,
                out_dir,
            ),
            extra_config={
                "engine_role": "llm",
                "model": "qwen3_vl_text",
                "num_hidden_layers": num_layers,
                "num_attention_heads": int(language.config.num_attention_heads),
                "num_key_value_heads": int(language.config.num_key_value_heads),
                "head_dim": int(meta["head_dim"]),
                "hidden_size": int(language.config.hidden_size),
                "intermediate_size": int(language.config.intermediate_size),
                "vocab_size": int(language.config.vocab_size),
                "max_position_embeddings": int(language.config.max_position_embeddings),
                "rope_theta": float(rope_fields["rope_theta"]),
                "rope_scaling": rope_fields["rope_scaling"],
                "partial_rotary_factor": float(
                    getattr(language.config, "partial_rotary_factor", 1.0)
                ),
                "kv_cache_dtype": "fp16",
                "num_deepstack_features": num_deepstack,
                "image_token_id": int(vlm.config.image_token_id),
                "spec_decode_type": "none",
                "builder_config": {
                    "max_batch_size": batch_size,
                    "max_input_len": prompt_len,
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
                "immutable_weights": False,
                "offload_module_to_cpu": True,
            },
        )

        action_dims = tuple(model.action_space.get_action_space_dims())
        batch_size = int(inputs_embeds.shape[0])
        noise_trajectory = sample.get(
            "step_actions",
            torch.randn(
                batch_size,
                *action_dims,
                device=device,
                dtype=torch.float32,
            ),
        )

        language_cfg = language.config
        expert_cfg = model.expert.config
        expert_rope_fields = export_rope_fields(expert_cfg.to_dict())
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
        num_layers = int(expert_cfg.num_hidden_layers)
        max_kv_capacity = int(max_seq_len)
        model.expert.config._attn_implementation = "sdpa"
        num_action_tokens = int(action_dims[0])
        time_steps_t0 = torch.tensor([0.0], device=device, dtype=torch.float32)
        time_steps_t1 = torch.tensor([0.1], device=device, dtype=torch.float32)
        action_kv_start = torch.full(
            (batch_size,),
            prompt_len,
            device=device,
            dtype=torch.int32,
        )
        action_rope = torch.randn(
            batch_size,
            num_action_tokens,
            head_dim,
            device=device,
            dtype=torch.float32,
        )
        action_position_ids = (
            torch.arange(
                num_action_tokens,
                device=device,
                dtype=torch.int32,
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        cache_shape = (
            batch_size,
            num_kv_heads,
            max_kv_capacity,
            head_dim,
        )
        k_caches = [
            torch.zeros(cache_shape, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        v_caches = [
            torch.zeros(cache_shape, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        action_args = (
            noise_trajectory,
            time_steps_t0,
            time_steps_t1,
            action_kv_start,
            action_rope,
            action_position_ids,
            *k_caches,
            *v_caches,
        )
        sample["edge_action_args"] = action_args
        traj_cfg = model.config.traj_tokenizer_cfg
        traj_token_start = int(model.config.traj_token_start_idx) + int(
            traj_cfg["num_bins"]
        )
        action = ComponentBundle(
            module=_export_module(model, device, dtype),
            trace_args=action_args,
            save_args=action_args,
            input_names=[
                "noise_trajectory",
                "time_steps_t0",
                "time_steps_t1",
                "kvcache_start_index",
                "rope_rotary_cos_sin",
                "attention_pos_id",
                *[f"k_cache_{i}" for i in range(num_layers)],
                *[f"v_cache_{i}" for i in range(num_layers)],
            ],
            output_names=[
                "denoised_trajectory",
                *[f"present_k_cache_{i}" for i in range(num_layers)],
                *[f"present_v_cache_{i}" for i in range(num_layers)],
            ],
            model_type="action",
            engine_file="action.engine",
            edge_runtime_bindings=True,
            extra_config={
                "rope_theta": float(expert_rope_fields["rope_theta"]),
                "rope_scaling": expert_rope_fields["rope_scaling"],
                "num_hidden_layers": num_layers,
                "num_attention_heads": int(expert_cfg.num_attention_heads),
                "num_key_value_heads": num_kv_heads,
                "head_dim": head_dim,
                "hidden_size": int(expert_cfg.hidden_size),
                "intermediate_size": int(expert_cfg.intermediate_size),
                "rms_norm_eps": float(expert_cfg.rms_norm_eps),
                "num_traj_tokens": 1000,
                "traj_token_start": traj_token_start,
                "n_diffusion_tokens": num_action_tokens,
                "builder_config": {
                    "max_batch_size": batch_size,
                    "max_kv_cache_capacity": max_kv_capacity,
                },
            },
            trt_settings={
                "disable_tf32": True,
                "use_fp32_acc": True,
                "use_explicit_typing": True,
                "decompose_attention": True,
                "immutable_weights": False,
            },
        )
        sample["step_actions"] = noise_trajectory
        sample["action_kv_start"] = action_kv_start
        sample["action_rope"] = action_rope
        sample["action_position_ids"] = action_position_ids
        sample["action_time_steps"] = torch.linspace(
            0.0,
            1.0,
            11,
            device=device,
            dtype=torch.float32,
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
        prefix_k, prefix_v = language[2], language[3]
        cache_capacity = int(sample["rope_rotary_cos_sin"].shape[1])
        cache_padding = cache_capacity - int(prefix_k.shape[-2])
        prefix_k = F.pad(prefix_k, (0, 0, 0, cache_padding))
        prefix_v = F.pad(prefix_v, (0, 0, 0, cache_padding))
        k_caches = list(prefix_k.unbind(0))
        v_caches = list(prefix_v.unbind(0))

        noise = sample["step_actions"]
        time_steps = sample["action_time_steps"]
        for step_index in range(10):
            action = call_engine(
                engines["action"],
                "action",
                noise,
                time_steps[step_index : step_index + 1],
                time_steps[step_index + 1 : step_index + 2],
                sample["action_kv_start"],
                sample["action_rope"],
                sample["action_position_ids"],
                *k_caches,
                *v_caches,
            )
            noise = action[0]
            k_caches = list(action[1 : 1 + len(k_caches)])
            v_caches = list(action[1 + len(k_caches) :])
        return noise
