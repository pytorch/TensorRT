from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

import torch

from ..config import EdgeConfig
from ..ops import call_engine
from ..spec import ComponentBundle, EdgeSpec
from .action import save_action_pte
from .artifact import (
    build_action_artifact,
    build_language_artifact,
    build_vision_artifact,
)
from .decode import save_language_decode_pte
from .language import save_language_prefill_pte
from .serialization import EdgeOutputSpec
from .vision import save_vision_pte


class ComponentPackager(Protocol):
    """Component-specific preparation and PTE packaging."""

    def prepare_bundle(
        self,
        bundle: ComponentBundle,
        spec: EdgeSpec,
        config: EdgeConfig,
    ) -> ComponentBundle:
        """Return the bundle that should be compiled."""

    def package(
        self,
        engine_path: Path,
        bundle: ComponentBundle,
        output_dir: Path,
        *,
        device_id: int,
    ) -> dict[str, Path]:
        """Package one compiled engine into one or more PTE programs."""


class VisionPackager:
    def prepare_bundle(
        self,
        bundle: ComponentBundle,
        spec: EdgeSpec,
        config: EdgeConfig,
    ) -> ComponentBundle:
        del spec, config
        return bundle

    def package(
        self,
        engine_path: Path,
        bundle: ComponentBundle,
        output_dir: Path,
        *,
        device_id: int,
    ) -> dict[str, Path]:
        if len(bundle.save_args) != 1:
            raise ValueError("Vision ExecuTorch export requires exactly one input")

        path = output_dir / "vision.pte"
        artifact = build_vision_artifact(engine_path, device_id=device_id)
        save_vision_pte(artifact, bundle.save_args[0], path)
        return {"vision": path}


class LanguagePackager:
    @staticmethod
    def _with_kv_start(
        values: tuple[Any, ...],
        input_names: list[str],
    ) -> tuple[Any, ...]:
        try:
            index = input_names.index("kvcache_start_index")
            embeds = values[input_names.index("inputs_embeds")]
        except ValueError as exc:
            raise ValueError(
                "Language bundle requires inputs_embeds and kvcache_start_index"
            ) from exc

        normalized = list(values)
        normalized[index] = torch.zeros(
            int(embeds.shape[0]),
            dtype=torch.int32,
            device=embeds.device,
        )
        return tuple(normalized)

    def prepare_bundle(
        self,
        bundle: ComponentBundle,
        spec: EdgeSpec,
        config: EdgeConfig,
    ) -> ComponentBundle:
        # A batch-sized KV start tensor is valid for both profile 0 prefill and
        # profile 1 decode. The ordinary Edge path uses an empty prefill tensor.
        trace_args = self._with_kv_start(
            tuple(bundle.trace_args),
            bundle.input_names,
        )
        save_args = self._with_kv_start(
            tuple(bundle.save_args),
            bundle.input_names,
        )
        execute_args = self._with_kv_start(
            tuple(bundle.execute_args or bundle.save_args),
            bundle.input_names,
        )
        max_seq_len = next(
            int(value.shape[-2])
            for name, value in zip(bundle.input_names, trace_args)
            if name.startswith("past_key_values_")
        )
        input_specs = spec.create_dynamic_shapes(
            bundle.input_names,
            trace_args,
            max_seq_len=max_seq_len,
        )
        return replace(
            bundle,
            trace_args=trace_args,
            save_args=save_args,
            execute_args=execute_args,
            input_specs=input_specs,
        )

    @staticmethod
    def _decode_inputs(bundle: ComponentBundle) -> tuple[torch.Tensor, ...]:
        named = dict(zip(bundle.input_names, bundle.save_args))
        required = {
            "inputs_embeds",
            "rope_rotary_cos_sin",
            "context_lengths",
            "kvcache_start_index",
            "last_token_ids",
            "ds_stack",
        }
        missing = required.difference(named)
        if missing:
            raise ValueError(
                f"Language decode packaging is missing inputs: {sorted(missing)}"
            )

        embeds = named["inputs_embeds"]
        ds_stack = named["ds_stack"]
        batch_size = int(embeds.shape[0])
        prefix_length = int(embeds.shape[1])
        device = embeds.device

        replacements = {
            "inputs_embeds": embeds[:, -1:].contiguous(),
            "context_lengths": torch.full(
                (batch_size,),
                prefix_length + 1,
                dtype=torch.int32,
                device=device,
            ),
            "kvcache_start_index": torch.full(
                (batch_size,),
                prefix_length,
                dtype=torch.int32,
                device=device,
            ),
            "last_token_ids": torch.zeros(
                (batch_size, 1),
                dtype=torch.int64,
                device=device,
            ),
            "ds_stack": ds_stack[:, :, :1].contiguous(),
        }
        return tuple(replacements.get(name, value) for name, value in named.items())

    def package(
        self,
        engine_path: Path,
        bundle: ComponentBundle,
        output_dir: Path,
        *,
        device_id: int,
    ) -> dict[str, Path]:
        prefill_path = output_dir / "language_prefill.pte"
        prefill_artifact = build_language_artifact(
            engine_path,
            device_id=device_id,
            runner="llm_prefill",
        )
        save_language_prefill_pte(
            prefill_artifact,
            tuple(bundle.save_args),
            prefill_path,
        )

        decode_inputs = self._decode_inputs(bundle)
        with torch.no_grad():
            decode_outputs = call_engine(
                str(engine_path),
                "language",
                *decode_inputs,
            )
        decode_specs = tuple(
            EdgeOutputSpec(
                shape=tuple(output.shape),
                dtype=str(output.dtype).removeprefix("torch."),
            )
            for output in decode_outputs
        )
        decode_path = output_dir / "language_decode.pte"
        decode_artifact = build_language_artifact(
            engine_path,
            device_id=device_id,
            runner="llm_decode",
            output_specs=decode_specs,
        )
        save_language_decode_pte(
            decode_artifact,
            decode_inputs,
            decode_path,
        )
        return {
            "language_prefill": prefill_path,
            "language_decode": decode_path,
        }


class ActionPackager:
    def prepare_bundle(
        self,
        bundle: ComponentBundle,
        spec: EdgeSpec,
        config: EdgeConfig,
    ) -> ComponentBundle:
        del spec, config
        return bundle

    def package(
        self,
        engine_path: Path,
        bundle: ComponentBundle,
        output_dir: Path,
        *,
        device_id: int,
    ) -> dict[str, Path]:
        path = output_dir / "action.pte"
        artifact = build_action_artifact(engine_path, device_id=device_id)
        save_action_pte(artifact, tuple(bundle.save_args), path)
        return {"action": path}


DEFAULT_PACKAGERS: Mapping[str, ComponentPackager] = {
    "vit": VisionPackager(),
    "language": LanguagePackager(),
    "action": ActionPackager(),
}
