from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import torch
import torch.nn as nn
from torch_tensorrt.hf.exporters.ops import call_engine
from torch_tensorrt.hf.exporters.spec import (
    ComponentBundle,
    EdgeSpec,
    register_edge_spec,
)
from torch_tensorrt.hf.exporters.specs._language import kv_kwargs, split_flat_to_kwargs


def _decoder(model: nn.Module) -> nn.Module:
    return getattr(model, "backbone", None) or model.model


def _kind(mixer: nn.Module) -> str:
    name = type(mixer).__name__
    if "Mamba" in name:
        return "mamba"
    if "Attention" in name:
        return "attention"
    if "MoE" in name or "Moe" in name:
        return "moe"
    return "mlp"


class NemotronExportModule(nn.Module):  # type: ignore[misc]
    """Hybrid decoder: plugin attention / mamba / moe, native MLP."""

    def __init__(self, model: nn.Module):
        super().__init__()
        decoder = _decoder(model)
        self.layers = decoder.layers
        self.norm = decoder.norm_f
        self.lm_head = model.lm_head
        self.kinds = [_kind(block.mixer) for block in self.layers]
        self.num_attn = self.kinds.count("attention")
        self.num_mamba = self.kinds.count("mamba")

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        context_lengths: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        last_token_ids: torch.Tensor,
        *states: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        from trt.modules.export.language import gather_last_token_hidden

        na, nm = self.num_attn, self.num_mamba
        kvs = list(states[:na])
        convs = list(states[na : na + nm])
        ssms = list(states[na + nm : na + 2 * nm])
        kv_i = conv_i = 0
        hidden = inputs_embeds
        present_kv, present_conv, present_ssm = [], [], []
        for block, kind in zip(self.layers, self.kinds):
            residual = hidden
            hidden = block.norm(hidden)
            mixer = block.mixer
            if kind == "attention":
                hidden, kv = mixer(
                    hidden_states=hidden,
                    rope_rotary_cos_sin=rope_rotary_cos_sin,
                    past_key_value=kvs[kv_i],
                    ctx_len=context_lengths,
                    kvcache_start_index=kvcache_start_index,
                )
                present_kv.append(kv)
                kv_i += 1
            elif kind == "mamba":
                hidden, conv_out, ssm_out = mixer(
                    hidden, convs[conv_i], ssms[conv_i], context_lengths
                )
                present_conv.append(conv_out)
                present_ssm.append(ssm_out)
                conv_i += 1
            else:
                hidden = mixer(hidden)
            hidden = residual + hidden
        hidden = self.norm(hidden)
        last = gather_last_token_hidden(hidden, last_token_ids)
        logits = self.lm_head(last).float()
        return (logits, *present_kv, *present_conv, *present_ssm)


def allocate_plugin_states(
    model: nn.Module,
    config: Any,
    batch: int,
    max_seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    kinds = [_kind(block.mixer) for block in _decoder(model).layers]
    head_dim = int(
        getattr(config, "head_dim", 0)
        or config.hidden_size // config.num_attention_heads
    )
    conv_dim = int(config.mamba_num_heads) * int(config.mamba_head_dim) + 2 * int(
        config.n_groups
    ) * int(config.ssm_state_size)
    conv_kernel = int(getattr(config, "conv_kernel", 4))
    kvs, convs, ssms = [], [], []
    for kind in kinds:
        if kind == "attention":
            kvs.append(
                torch.zeros(
                    batch,
                    2,
                    int(config.num_key_value_heads),
                    max_seq_len,
                    head_dim,
                    device=device,
                    dtype=dtype,
                )
            )
        elif kind == "mamba":
            convs.append(
                torch.zeros(batch, conv_dim, conv_kernel, device=device, dtype=dtype)
            )
            ssms.append(
                torch.zeros(
                    batch,
                    int(config.mamba_num_heads),
                    int(config.mamba_head_dim),
                    int(config.ssm_state_size),
                    device=device,
                    dtype=dtype,
                )
            )
    return kvs, convs, ssms


@register_edge_spec("nemotron_h", "nemotron")
class NemotronSpec(EdgeSpec):
    components = ("language",)

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

    def wrap(
        self, name: str, model: nn.Module, sample: Mapping[str, Any], config: Any
    ) -> nn.Module:
        from trt.plugin.moe import PluginNemotronMoE
        from trt.plugin.plugin_utils import patch_nemotron_mixers

        patch_nemotron_mixers(model, model.config)
        for block in _decoder(model).layers:
            if isinstance(block.mixer, PluginNemotronMoE):
                block.mixer.prepare_for_export()
        return NemotronExportModule(model).eval()

    def prepare(
        self,
        name: str,
        model: nn.Module,
        sample: MutableMapping[str, Any],
        upstream: Mapping[str, Any],
        config: Any,
        module: nn.Module,
    ) -> ComponentBundle:
        from trt.rope import make_rope_rotary_cos_sin

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
        na, nm = module.num_attn, module.num_mamba
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
