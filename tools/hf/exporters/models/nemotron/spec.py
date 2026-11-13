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
from ..common.helpers import (
    kv_kwargs,
    split_flat_to_kwargs,
)
from .helpers import (
    _decoder,
    _kind,
    allocate_plugin_states,
)
from .patches import (
    apply_nemotron_patches,
)


@register_edge_spec("nemotron_h", "nemotron")
class NemotronSpec(EdgeSpec):  # type: ignore[misc]
    components = ("language",)

    def apply_patches(self, model=None):
        return apply_nemotron_patches(model)

    def prepare_sample_inputs(
        self, model: nn.Module, raw: Mapping[str, Any], config: Any
    ) -> MutableMapping[str, Any]:
        if "inputs_embeds" in raw:
            return dict(raw)
        device = raw["input_ids"].device
        embeddings = model.get_input_embeddings()(raw["input_ids"])
        mask = raw.get("attention_mask")
        max_seq_len = int(config.max_seq_len)
        bsz, prompt_len, hidden = embeddings.shape
        if prompt_len < max_seq_len:
            pad = max_seq_len - prompt_len
            embeddings = torch.cat(
                [
                    embeddings,
                    torch.zeros(
                        bsz, pad, hidden, device=device, dtype=embeddings.dtype
                    ),
                ],
                dim=1,
            )
            if mask is not None:
                mask = torch.cat(
                    [mask, torch.ones(bsz, pad, device=device, dtype=mask.dtype)],
                    dim=1,
                )
        return {
            "inputs_embeds": embeddings,
            "attention_mask": mask,
            "bsz": embeddings.shape[0],
        }

    def prepare(
        self,
        name: str,
        model: nn.Module,
        sample: MutableMapping[str, Any],
        upstream: Mapping[str, Any],
        config: Any,
    ) -> ComponentBundle:
        from ...rope import make_rope_rotary_cos_sin

        del name, upstream
        embeds = sample["inputs_embeds"]
        device, dtype = embeds.device, embeds.dtype
        bsz, seq_len, _ = embeds.shape
        rope = make_rope_rotary_cos_sin(
            model.config,
            int(config.max_seq_len),
            device,
            language_model=_decoder(model),
        )
        ctx_len = torch.full((bsz,), seq_len, device=device, dtype=torch.int32)
        last_token_ids = torch.full(
            (bsz, 1), seq_len - 1, device=device, dtype=torch.int64
        )
        kv_start = torch.empty(0, dtype=torch.int32, device=device)
        kvs, convs, ssms = allocate_plugin_states(
            model, model.config, bsz, int(config.max_seq_len), device, dtype
        )
        flat = (embeds, rope, ctx_len, kv_start, last_token_ids, *kvs, *convs, *ssms)
        kinds = [_kind(block.mixer) for block in _decoder(model).layers]
        na, nm = kinds.count("attention"), kinds.count("mamba")
        names = [
            "inputs_embeds",
            "rope_rotary_cos_sin",
            "context_lengths",
            "kvcache_start_index",
            "last_token_ids",
            *[f"past_key_values_{i}" for i in range(na)],
            *[f"conv_state_{i}" for i in range(nm)],
            *[f"ssm_state_{i}" for i in range(nm)],
        ]
        sample.update(split_flat_to_kwargs(flat, names))
        return ComponentBundle(
            module=model.eval(),
            trace_args=flat,
            save_args=flat,
            input_names=names,
            output_names=["logits"]
            + [f"present_kv_{i}" for i in range(na)]
            + [f"present_conv_{i}" for i in range(nm)]
            + [f"present_ssm_{i}" for i in range(nm)],
            model_type="nemotron",
            engine_file="language.engine",
        )

    def run(self, engines: Mapping[str, str], sample: Mapping[str, Any]) -> Any:
        leading = [
            sample["inputs_embeds"],
            sample["rope_rotary_cos_sin"],
            sample["context_lengths"],
            sample["kvcache_start_index"],
            sample["last_token_ids"],
        ]
        states = (
            kv_kwargs(sample)
            + kv_kwargs(sample, "conv_state_")
            + kv_kwargs(sample, "ssm_state_")
        )
        return call_engine(engines["language"], "language", *leading, *states)
