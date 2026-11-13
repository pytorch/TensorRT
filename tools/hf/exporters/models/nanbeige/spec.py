from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import torch
import torch.nn as nn

from ...ops import call_engine
from ...spec import (
    ComponentBundle,
    EdgeSpec,
    register_edge_spec,
)
from ..common.helpers import kv_kwargs, split_flat_to_kwargs
from .helpers import ensure_valid_rope_inv_freq
from .patches import apply_nanbeige_patches


@register_edge_spec("nanbeige")
class NanbeigeSpec(EdgeSpec):  # type: ignore[misc]
    def apply_patches(self, model=None):
        return apply_nanbeige_patches(model)

    def prepare_sample_inputs(
        self, model: nn.Module, raw: Mapping[str, Any], config: Any
    ) -> MutableMapping[str, Any]:
        del config
        ensure_valid_rope_inv_freq(model)
        if "inputs_embeds" in raw:
            sample = dict(raw)
            mask = sample.get("attention_mask")
            if mask is not None:
                sample["attention_mask"] = mask.to(
                    device=sample["inputs_embeds"].device
                )
            return sample
        if "input_ids" not in raw:
            raise KeyError("Nanbeige inputs require input_ids or inputs_embeds")

        embedding = model.get_input_embeddings()
        input_ids = raw["input_ids"]
        with torch.no_grad():
            inputs_embeds = embedding(input_ids.to(device=embedding.weight.device))

        attention_mask = raw.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=inputs_embeds.device)
        return {
            "inputs_embeds": inputs_embeds.contiguous(),
            "attention_mask": attention_mask,
        }

    def capture_eager_outputs(
        self, model, sample, config, bench=None
    ) -> dict[str, torch.Tensor]:
        del config
        from ...measure import cuda_ms

        kwargs = {
            "inputs_embeds": sample["inputs_embeds"],
            "use_cache": False,
            "return_dict": True,
        }
        if sample.get("attention_mask") is not None:
            kwargs["attention_mask"] = sample["attention_mask"]

        with torch.no_grad():
            output = model(**kwargs)
            logits = output.logits

        # The plugin forward returns logits for one selected token, not [B, S, V].
        mask = sample.get("attention_mask")
        if mask is None:
            last_token_ids = torch.full(
                (logits.shape[0],),
                logits.shape[1] - 1,
                device=logits.device,
                dtype=torch.long,
            )
        else:
            positions = torch.arange(logits.shape[1], device=logits.device)
            last_token_ids = (
                positions.expand_as(mask)
                .masked_fill(~mask.bool(), -1)
                .max(dim=1)
                .values
            )
            if bool((last_token_ids < 0).any()):
                raise ValueError("attention_mask contains an empty sequence")

        batch = torch.arange(logits.shape[0], device=logits.device)
        last_logits = logits[batch, last_token_ids]

        if bench is not None:
            bench["language"] = cuda_ms(lambda: model(**kwargs).logits)

        return {"language": last_logits}

    def prepare(
        self,
        model: nn.Module,
        sample: MutableMapping[str, Any],
        config: Any,
    ) -> dict[str, ComponentBundle]:
        from ...plugin.attention import ContextAttentionMaskType
        from ...rope import make_rope_rotary_cos_sin

        decoder = model if hasattr(model, "layers") else model.model
        if getattr(decoder.config, "enable_double_loop_split", False):
            raise NotImplementedError("Nanbeige LoopSplit export is not supported")
        if getattr(decoder.config, "enable_hyper_connection", False):
            raise NotImplementedError(
                "Nanbeige hyper-connection export is not supported"
            )
        if getattr(decoder.config, "enable_depth_attention", False):
            raise NotImplementedError(
                "Nanbeige depth-attention export is not supported"
            )
        if getattr(decoder, "ngram_embeddings", None) is not None:
            raise NotImplementedError("Nanbeige n-gram export is not supported")

        embeds = sample["inputs_embeds"]
        device, dtype = embeds.device, embeds.dtype
        batch_size, seq_len, hidden_size = embeds.shape

        max_seq_len = max(int(config.max_seq_len), seq_len)
        physical_layers = len(decoder.layers)
        loop_weights = getattr(decoder.config, "loop_loss_weights", None)
        num_loops = (
            len(loop_weights) + 1
            if loop_weights
            else max(int(getattr(decoder.config, "num_loops", 1)), 1)
        )
        logical_layers = physical_layers * num_loops

        num_kv_heads = int(decoder.config.num_key_value_heads)
        head_dim = int(
            getattr(decoder.config, "head_dim", None)
            or hidden_size // int(decoder.config.num_attention_heads)
        )
        rope = make_rope_rotary_cos_sin(
            decoder.config,
            max_seq_len,
            device,
            language_model=decoder,
        )

        attention_mask = sample.get("attention_mask")
        if attention_mask is None:
            context_lengths = torch.full(
                (batch_size,),
                seq_len,
                device=device,
                dtype=torch.int32,
            )
            last_token_ids = torch.full(
                (batch_size, 1),
                seq_len - 1,
                device=device,
                dtype=torch.int64,
            )
        else:
            context_lengths = attention_mask.sum(dim=-1, dtype=torch.int32)
            positions = torch.arange(seq_len, device=device)
            last_token_ids = (
                positions.expand_as(attention_mask)
                .masked_fill(~attention_mask.bool(), -1)
                .max(dim=1)
                .values.unsqueeze(-1)
            )

        kvcache_start_index = torch.empty(0, device=device, dtype=torch.int32)
        past_key_values = [
            torch.zeros(
                batch_size,
                2,
                num_kv_heads,
                max_seq_len,
                head_dim,
                device=device,
                dtype=dtype,
            )
            for _ in range(logical_layers)
        ]
        flat = (
            embeds,
            rope,
            context_lengths,
            kvcache_start_index,
            last_token_ids,
            *past_key_values,
        )
        input_names = [
            "inputs_embeds",
            "rope_rotary_cos_sin",
            "context_lengths",
            "kvcache_start_index",
            "last_token_ids",
            *[f"past_key_values_{index}" for index in range(logical_layers)],
        ]
        sample.update(split_flat_to_kwargs(flat, input_names))
        return {
            "language": ComponentBundle(
                module=model.eval(),
                trace_args=flat,
                save_args=flat,
                input_names=input_names,
                output_names=[
                    "logits",
                    "hidden_states",
                    "prefix_k",
                    "prefix_v",
                ],
                parity_output="logits",
                context_attention_mask_type=int(ContextAttentionMaskType.CAUSAL),
                model_type="nanbeige",
                engine_file="language.engine",
                trt_settings={
                    "disable_tf32": True,
                    "use_fp32_acc": True,
                    "use_explicit_typing": True,
                    "decompose_attention": True,
                    "assume_dynamic_shape_support": True,
                },
            )
        }

    def run(self, engines: Mapping[str, str], sample: Mapping[str, Any]) -> Any:
        return call_engine(
            engines["language"],
            "language",
            sample["inputs_embeds"],
            sample["rope_rotary_cos_sin"],
            sample["context_lengths"],
            sample["kvcache_start_index"],
            sample["last_token_ids"],
            *kv_kwargs(sample),
        )
