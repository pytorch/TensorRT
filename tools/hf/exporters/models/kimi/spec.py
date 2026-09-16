from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import torch
import torch.nn as nn

from ...ops import call_engine
from ...spec import ComponentBundle, EdgeSpec, register_edge_spec
from .helpers import allocate_kda_states, language_model
from .patches import apply_kimik3_patches


@register_edge_spec("kimi", "kimi-k3", "kimi_k3")
class KimiK3Spec(EdgeSpec):  # type: ignore[misc]
    def apply_patches(self, model=None):
        return apply_kimik3_patches(model)

    def prepare_sample_inputs(
        self,
        model: nn.Module,
        raw: Mapping[str, Any],
        config: Any,
    ) -> MutableMapping[str, Any]:
        del config
        if "inputs_embeds" in raw:
            inputs_embeds = raw["inputs_embeds"]
        elif "input_ids" in raw:
            embedding = model.get_input_embeddings()
            input_ids = raw["input_ids"].to(embedding.weight.device)
            with torch.no_grad():
                inputs_embeds = embedding(input_ids)
        else:
            raise KeyError("Kimi K3 requires input_ids or inputs_embeds")

        attention_mask = raw.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2],
                device=inputs_embeds.device,
                dtype=torch.long,
            )
        else:
            attention_mask = attention_mask.to(inputs_embeds.device)
        return {
            "inputs_embeds": inputs_embeds.contiguous(),
            "attention_mask": attention_mask.contiguous(),
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

        kwargs = {
            "inputs_embeds": sample["inputs_embeds"],
            "attention_mask": sample["attention_mask"],
            "use_cache": False,
            "return_dict": True,
        }
        with torch.no_grad():
            logits = language_model(model)(**kwargs).logits

        context_lengths = sample["attention_mask"].sum(dim=-1, dtype=torch.int64)
        last_token_ids = context_lengths - 1
        batch = torch.arange(logits.shape[0], device=logits.device)
        selected_logits = logits[batch, last_token_ids]

        if bench is not None:
            bench["language"] = cuda_ms(lambda: language_model(model)(**kwargs).logits)
        return {"language": selected_logits}

    def prepare(
        self,
        model: nn.Module,
        sample: MutableMapping[str, Any],
        config: Any,
    ) -> dict[str, ComponentBundle]:
        del config
        embeddings = sample["inputs_embeds"]
        batch_size = int(embeddings.shape[0])
        context_lengths = sample["attention_mask"].sum(dim=-1, dtype=torch.int32)
        if bool((context_lengths <= 0).any()):
            raise ValueError("Kimi attention_mask contains an empty sequence")
        last_token_ids = (context_lengths - 1).to(torch.int64).unsqueeze(-1)
        states, state_names = allocate_kda_states(
            model,
            batch_size=batch_size,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        args = (embeddings, context_lengths, last_token_ids, *states)
        input_names = [
            "inputs_embeds",
            "context_lengths",
            "last_token_ids",
            *state_names,
        ]
        sample.update(dict(zip(input_names, args)))

        return {
            "language": ComponentBundle(
                module=language_model(model).eval(),
                trace_args=args,
                save_args=args,
                input_names=input_names,
                output_names=["logits"],
                parity_output="logits",
                model_type="kimi_k3",
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
        state_names = sorted(
            (
                name
                for name in sample
                if name.startswith("kda_conv_") or name.startswith("kda_recurrent_")
            ),
            key=_state_sort_key,
        )
        return call_engine(
            engines["language"],
            "language",
            sample["inputs_embeds"],
            sample["context_lengths"],
            sample["last_token_ids"],
            *(sample[name] for name in state_names),
        )


def _state_sort_key(name: str) -> tuple[int, int]:
    layer_index = int(name.rsplit("_", 1)[1])
    kind = name[: name.rfind("_")]
    order = {
        "kda_conv_q": 0,
        "kda_conv_k": 1,
        "kda_conv_v": 2,
        "kda_recurrent": 3,
    }
    return layer_index, order[kind]
