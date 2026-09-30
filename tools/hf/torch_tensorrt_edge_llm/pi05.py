"""PI0.5 adapters with explicit prefix caches and reusable action execution.

Inputs are already preprocessed: camera-first normalized HWC pixels, scaled
language embeddings, additive masks, positions, and caller-provided noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from . import ops
from .artifact import (
    PI05_COMPONENT_INPUTS,
    PI05_COMPONENT_OUTPUTS,
    EdgeExecuTorchArtifact,
)
from .prefix_cache import PrefixKVCache


def prepare_pi05_sample(
    core: nn.Module,
    images: list[torch.Tensor],
    image_masks: list[torch.Tensor],
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    noise: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Use preprocessed LeRobot observations without changing camera/token masks.

    Retaining masked slots gives fixed shapes and supports different valid token
    counts within one batch. Image resize/normalization and tokenization remain
    the caller's responsibility. Noise determines the output reproducibly.
    """
    from lerobot.policies.common.vla_utils import (
        make_att_2d_masks,
        prepare_attention_masks_4d,
    )

    if (
        not images
        or len(images) != len(image_masks)
        or any(image.ndim != 4 or image.shape != images[0].shape for image in images)
    ):
        raise ValueError("Expected equally shaped NCHW images and one mask per camera")
    with torch.no_grad():
        prefix, pads, attention = core.embed_prefix(
            images, image_masks, tokens, token_mask
        )
        language = prefix[:, -tokens.shape[1] :].detach().contiguous()
        batch, length = pads.shape
        index = (
            torch.arange(length, device=tokens.device)[None]
            .expand(batch, -1)
            .contiguous()
        )
        prefix_positions = pads.cumsum(1) - 1
        prefix_mask = prepare_attention_masks_4d(make_att_2d_masks(pads, attention))
        suffix_len = noise.shape[1]
        action_positions = (
            pads.sum(1)[:, None] + torch.arange(suffix_len, device=noise.device)[None]
        )
        action_valid = torch.cat(
            [
                pads[:, None].expand(-1, suffix_len, -1),
                torch.ones(
                    batch, suffix_len, suffix_len, dtype=torch.bool, device=noise.device
                ),
            ],
            -1,
        )
        action_mask = prepare_attention_masks_4d(action_valid)
        pixels = torch.cat(images).permute(0, 2, 3, 1).contiguous()
        return (
            pixels,
            language,
            index,
            prefix_mask,
            prefix_positions,
            noise,
            action_positions,
            action_mask,
        )


def write_native_inputs(
    sample: tuple[torch.Tensor, ...], directory: str | Path
) -> None:
    """Write portable raw inputs for the C++ PI0.5 reference application.

    The first application uses float32 pixels, embeddings, masks and noise, plus
    int64 positions/indices. Engine caches can use their own native dtype.
    """
    names = (
        "pixels",
        "language_embeds",
        "compact_index",
        "prefix_mask",
        "prefix_positions",
        "noise",
        "action_positions",
        "action_mask",
    )
    integer_names = {"compact_index", "prefix_positions", "action_positions"}
    if len(sample) != len(names):
        raise ValueError("Expected eight PI0.5 input tensors")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in zip(names, sample):
        expected = torch.int64 if name in integer_names else torch.float32
        if value.dtype != expected:
            raise ValueError(
                f"Native reference input {name} requires {expected}, got {value.dtype}"
            )
        (directory / f"{name}.bin").write_bytes(
            value.detach().cpu().contiguous().numpy().tobytes()
        )


class PI05Vision(nn.Module):
    def __init__(self, core: nn.Module, *, batch_size: int, cameras: int):
        super().__init__()
        paligemma = core.paligemma_with_expert.paligemma.model
        self.vision = paligemma.vision_tower
        self.projector = paligemma.multi_modal_projector
        self.batch_size, self.cameras = batch_size, cameras

    def forward(self, pixel_values):
        features = self.projector(
            self.vision(pixel_values.permute(0, 3, 1, 2).contiguous()).last_hidden_state
        )
        # [C*B,S,H] -> [B,C*S,H]. This also preserves camera order for B>1.
        return (
            features.reshape(self.cameras, self.batch_size, -1, features.shape[-1])
            .permute(1, 0, 2, 3)
            .reshape(self.batch_size, -1, features.shape[-1])
        )


class PI05Prefill(nn.Module):
    def __init__(self, core: nn.Module):
        super().__init__()
        self.language = core.paligemma_with_expert.paligemma.model.language_model
        self.language.config._attn_implementation = "eager"

    def forward(self, inputs_embeds, attention_mask, position_ids):
        config = self.language.config
        cache = PrefixKVCache.empty(
            num_layers=config.num_hidden_layers,
            batch_size=inputs_embeds.shape[0],
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        result = self.language(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        k, v = cache.get_updated_stacked()
        return result.last_hidden_state, k, v


class PI05Action(nn.Module):
    def __init__(self, core: nn.Module):
        super().__init__()
        from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding

        self.time_embedding = create_sinusoidal_pos_embedding
        self.action_in_proj = core.action_in_proj
        self.action_out_proj = core.action_out_proj
        self.time_mlp_in = core.time_mlp_in
        self.time_mlp_out = core.time_mlp_out
        self.min_period, self.max_period = (
            core.config.min_period,
            core.config.max_period,
        )
        self.chunk_size = core.config.chunk_size
        self.expert = core.paligemma_with_expert.gemma_expert.model
        self.expert.config._attn_implementation = "eager"

    def forward(self, x_t, timestep, prefix_k, prefix_v, position_ids, attention_mask):
        time = self.time_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.min_period,
            max_period=self.max_period,
            device=timestep.device,
        ).to(timestep.dtype)
        condition = torch.nn.functional.silu(
            self.time_mlp_out(torch.nn.functional.silu(self.time_mlp_in(time)))
        )
        suffix = self.action_in_proj(x_t)
        result = self.expert(
            inputs_embeds=suffix,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=PrefixKVCache(prefix_k, prefix_v),
            use_cache=False,
            adarms_cond=condition,
        )
        hidden = result.last_hidden_state[:, -self.chunk_size :].to(
            self.action_out_proj.weight.dtype
        )
        return self.action_out_proj(hidden).float()


class _Component(nn.Module):
    def __init__(self, artifact: EdgeExecuTorchArtifact):
        super().__init__()
        self.register_buffer(
            "engine", torch.frombuffer(bytearray(artifact.trt_blob), dtype=torch.uint8)
        )
        self.metadata = artifact.edge_metadata_json


class VisionMethod(_Component):
    def forward(self, pixels):
        return ops.vision_tower([pixels], self.engine, self.metadata)[0]


class PrefillMethod(_Component):
    def forward(self, inputs_embeds, attention_mask, position_ids):
        return ops.llm_prefill(
            inputs_embeds, attention_mask, position_ids, self.engine, self.metadata
        )


class ActionMethod(_Component):
    def forward(self, x_t, timestep, prefix_k, prefix_v, position_ids, attention_mask):
        return ops.action_expert(
            x_t,
            timestep,
            prefix_k,
            prefix_v,
            position_ids,
            attention_mask,
            self.engine,
            self.metadata,
        )


class PI05Runtime(nn.Module):
    """Runnable full-policy graph with explicit Euler integration."""

    def __init__(self, artifacts: dict[str, EdgeExecuTorchArtifact], *, num_steps: int):
        super().__init__()
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        self.vision = VisionMethod(artifacts["vision"])
        self.prefill = PrefillMethod(artifacts["language"])
        self.action_step = ActionMethod(artifacts["action"])
        self.num_steps = num_steps

    def forward(
        self,
        pixels,
        language_embeds,
        compact_index,
        prefix_mask,
        prefix_positions,
        noise,
        action_positions,
        action_mask,
    ):
        prefix = ops.fuse_prefix(self.vision(pixels), language_embeds, compact_index)
        _, k, v = self.prefill(prefix, prefix_mask, prefix_positions)
        x_t = noise
        for step in range(self.num_steps):
            time = torch.full(
                (noise.shape[0],),
                1.0 - step / self.num_steps,
                dtype=noise.dtype,
                device=noise.device,
            )
            velocity = self.action_step(x_t, time, k, v, action_positions, action_mask)
            x_t = x_t - velocity / self.num_steps
        return x_t


@dataclass
class PI05Export:
    program: torch.export.ExportedProgram
    methods: dict[str, torch.export.ExportedProgram]

    def save(
        self, directory: str | Path, *, output_format: str = "exported_program"
    ) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if output_format == "exported_program":
            path = directory / "pi05.pt2"
            torch.export.save(self.program, path)
            return path
        if output_format != "executorch":
            raise ValueError(f"Unsupported PI0.5 output format {output_format!r}")
        from executorch.exir import ExecutorchBackendConfig
        from executorch.exir.backend.compile_spec_schema import CompileSpec
        from torch_tensorrt.executorch import export

        from .executorch import EdgeLLMPartitioner

        # Application packing and the Euler loop stay outside native methods.
        # Each component has one engine/handle reused across all action steps.
        # Host IO uses TensorRT's native staging. CUDA device planning would
        # require ExecuTorch's device-copy kernels and allocator as well.
        lowered = export(
            self.methods,
            partitioners={
                name: [EdgeLLMPartitioner([CompileSpec("target_device", b"cpu")])]
                for name in self.methods
            },
        )
        path = directory / "pi05.pte"
        with path.open("wb") as stream:
            lowered.to_executorch(
                config=ExecutorchBackendConfig(enable_non_cpu_memory_planning=False)
            ).write_to_file(stream)
        return path


def export_pi05(
    core: nn.Module,
    sample: tuple[torch.Tensor, ...],
    *,
    engine_dir: str | Path,
    cameras: int,
    num_steps: int = 10,
    settings: dict[str, Any] | None = None,
) -> PI05Export:
    """Compile a LeRobot PI05Pytorch core into standalone opset programs.

    ``sample`` follows ``PI05Runtime.forward``. Shapes are fixed for this first
    path; MEM and RTC need separate contracts and are rejected here.
    """
    from .compiler import compile_component

    if len(sample) != 8 or cameras <= 0 or num_steps <= 0:
        raise ValueError("Expected eight policy tensors and positive cameras/steps")
    if (
        getattr(core.config, "use_proprioceptive_memory", False)
        or getattr(core.config, "rtc_config", None) is not None
    ):
        raise ValueError(
            "PI0.5 export currently supports the base policy without MEM/RTC"
        )
    (
        pixels,
        lang,
        index,
        prefix_mask,
        prefix_positions,
        noise,
        action_positions,
        action_mask,
    ) = sample
    batch = lang.shape[0]
    if pixels.shape[0] != cameras * batch:
        raise ValueError("Pixels must contain camera-first cameras * batch images")
    directory = Path(engine_dir)
    artifacts = {}
    with torch.no_grad():
        vision = PI05Vision(core, batch_size=batch, cameras=cameras).eval()
        artifacts["vision"] = compile_component(
            vision,
            (pixels,),
            directory / "vision",
            component="vision",
            input_names=("pixel_values",),
            output_names=("image_embs",),
            settings=settings,
        )
        prefix = ops.fuse_prefix(vision(pixels), lang, index)
        prefill = PI05Prefill(core).eval()
        prefill_args = (prefix, prefix_mask, prefix_positions)
        artifacts["language"] = compile_component(
            prefill,
            prefill_args,
            directory / "language",
            component="language",
            input_names=PI05_COMPONENT_INPUTS["language"],
            output_names=PI05_COMPONENT_OUTPUTS["language"],
            settings=settings,
        )
        _, k, v = prefill(*prefill_args)
        action_args = (
            noise,
            torch.ones((batch,), device=noise.device, dtype=noise.dtype),
            k,
            v,
            action_positions,
            action_mask,
        )
        artifacts["action"] = compile_component(
            PI05Action(core).eval(),
            action_args,
            directory / "action",
            component="action",
            input_names=PI05_COMPONENT_INPUTS["action"],
            output_names=PI05_COMPONENT_OUTPUTS["action"],
            settings=settings,
        )
    runtime = PI05Runtime(artifacts, num_steps=num_steps).eval()
    methods = {
        "vision": torch.export.export(runtime.vision, (pixels,)),
        "prefill": torch.export.export(runtime.prefill, prefill_args),
        "action_step": torch.export.export(runtime.action_step, action_args),
    }
    return PI05Export(torch.export.export(runtime, sample), methods)
