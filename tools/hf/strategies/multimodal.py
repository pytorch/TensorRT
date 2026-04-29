"""
Multimodal strategy: CLIP and similar dual-encoder vision-text models.

CLIP forward takes both `input_ids` (text) and `pixel_values` (image) and
returns image/text embeddings + cross-modal similarity logits.  Whole-model
export works because there's no autoregressive decode loop.

Supports:
  CLIPModel        : text encoder + vision encoder + projection heads
  CLIPVisionModel  : vision encoder only (also handled here for convenience)
  CLIPTextModel    : text encoder only

Benchmarking metric: latency + throughput of the full forward.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch_tensorrt
from common.bench import compute_stats, warmup_and_time
from common.compile import (
    compile_kwargs,
    compile_with_trt,
    maybe_save_exported_program,
    maybe_save_trt_engine,
    maybe_save_trt_module,
    model_dtype,
    safe_export,
)
from common.metrics import print_table, report_latency
from strategies.base import ModelStrategy, RunConfig


def _build_clip_kwargs(
    model_cfg, batch_size: int, dtype: torch.dtype, device: str
) -> dict:
    """
    Build {input_ids, attention_mask, pixel_values} for a CLIP-style model.
    Returned as kwargs to avoid positional-argument-order issues across model variants.
    """
    text_cfg = getattr(model_cfg, "text_config", model_cfg)
    vision_cfg = getattr(model_cfg, "vision_config", model_cfg)

    seq_len = getattr(text_cfg, "max_position_embeddings", 77)
    vocab = getattr(text_cfg, "vocab_size", 49408)
    image_size = getattr(vision_cfg, "image_size", 224)
    if isinstance(image_size, (list, tuple)):
        h, w = image_size[0], image_size[1]
    else:
        h = w = int(image_size)
    num_channels = getattr(vision_cfg, "num_channels", 3)

    return {
        "input_ids": torch.randint(
            0, vocab, (batch_size, seq_len), dtype=torch.int64
        ).to(device),
        "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.int64).to(
            device
        ),
        "pixel_values": torch.randn(batch_size, num_channels, h, w, dtype=dtype).to(
            device
        ),
    }


class MultimodalStrategy(ModelStrategy):
    def __init__(self, cfg: RunConfig):
        super().__init__(cfg)
        self._model: Optional[torch.nn.Module] = None
        self._trt_model: Optional[torch.nn.Module] = None
        self._dummy_kwargs: Optional[dict] = None
        self._model_cfg = None
        self._device = "cuda:0"
        self._model_dtype = model_dtype(cfg.precision, cfg.autocast)

    # ---------------------------------------------------------------------- #

    def load(self) -> None:
        from transformers import AutoConfig, AutoModel

        print(f"[multimodal] Loading {self.cfg.model} ...")
        self._model_cfg = AutoConfig.from_pretrained(self.cfg.model)
        self._model = (
            AutoModel.from_pretrained(
                self.cfg.model,
                ignore_mismatched_sizes=True,
            )
            .eval()
            .to(self._model_dtype)
            .to(self._device)
        )
        self._dummy_kwargs = _build_clip_kwargs(
            self._model_cfg,
            self.cfg.batch_size,
            self._model_dtype,
            self._device,
        )
        shapes = {k: v.shape for k, v in self._dummy_kwargs.items()}
        print(f"[multimodal] Model loaded.  Input shapes: {shapes}")

    # ---------------------------------------------------------------------- #

    def compile(self) -> None:
        assert self._model is not None, "Call load() before compile()"

        if self.cfg.mode == "compile":
            self._compile_torch_compile()
        else:
            self._compile_export()

    def _compile_torch_compile(self) -> None:
        print("[multimodal] torch.compile(backend='torch_tensorrt') ...")
        options = compile_kwargs(self.cfg.precision, self.cfg.autocast)
        options.update(
            {"min_block_size": self.cfg.min_block_size, "debug": self.cfg.debug}
        )

        self._trt_model = torch.compile(
            self._model, backend="torch_tensorrt", options=options
        )
        with torch.no_grad():
            _ = self._trt_model(**self._dummy_kwargs)
        torch.cuda.synchronize()

    def _compile_export(self) -> None:
        print("[multimodal] torch.export.export ...")
        # Dynamic batch on all kwargs.
        dyn = {k: {0: torch.export.Dim.AUTO} for k in self._dummy_kwargs}
        ep = safe_export(self._model, kwargs=self._dummy_kwargs, dynamic_shapes=dyn)
        maybe_save_exported_program(self.cfg, ep, log_prefix="[multimodal]")

        print("[multimodal] torch_tensorrt.dynamo.compile ...")
        # The EP was exported with all inputs as kwargs (input_ids,
        # attention_mask, pixel_values).  The compile/serialize APIs need
        # them as kwarg_inputs to preserve the EP's input spec.
        trt_kwarg_inputs = {
            k: torch_tensorrt.Input(shape=v.shape, dtype=v.dtype)
            for k, v in self._dummy_kwargs.items()
        }
        trt_inputs = list(trt_kwarg_inputs.values())
        self._trt_model = compile_with_trt(
            ep,
            inputs=trt_inputs,
            precision=self.cfg.precision,
            autocast=self.cfg.autocast,
            min_block_size=self.cfg.min_block_size,
            debug=self.cfg.debug,
            offload_module_to_cpu=self.cfg.offload_module_to_cpu,
            engine_cache_dir=self.cfg.engine_cache_dir,
        )
        torch.cuda.synchronize()

        # Serializer is strict about arg/kwarg split; CLIP's EP has only kwargs.
        maybe_save_trt_module(
            self.cfg,
            self._trt_model,
            arg_inputs=[],
            kwarg_inputs=trt_kwarg_inputs,
            log_prefix="[multimodal]",
        )
        maybe_save_trt_engine(
            self.cfg,
            ep,
            [],
            kwarg_inputs=trt_kwarg_inputs,
            log_prefix="[multimodal]",
        )

    # ---------------------------------------------------------------------- #

    def benchmark(self) -> list[dict]:
        assert self._dummy_kwargs is not None, "Call load() first"
        rows: list[dict] = []

        def _run_pt():
            with torch.no_grad():
                return self._model(**self._dummy_kwargs)

        pt_t = warmup_and_time(_run_pt, (), iterations=self.cfg.iterations)
        rows.append(
            report_latency(
                compute_stats(pt_t, self.cfg.batch_size),
                backend="pytorch",
                batch_size=self.cfg.batch_size,
                precision=self.cfg.precision,
            )
        )

        if self._trt_model is not None:
            # The exported TRT module takes positional inputs in the order
            # they were passed at export time.
            trt_args = tuple(self._dummy_kwargs.values())

            def _run_trt():
                with torch.no_grad():
                    return self._trt_model(*trt_args)

            trt_t = warmup_and_time(_run_trt, (), iterations=self.cfg.iterations)
            rows.append(
                report_latency(
                    compute_stats(trt_t, self.cfg.batch_size),
                    backend=f"torch_tensorrt[{self.cfg.mode}]",
                    batch_size=self.cfg.batch_size,
                    precision=self.cfg.precision,
                )
            )

        print_table(rows, title=f"Multimodal benchmark – {self.cfg.model}")
        return rows

    # ---------------------------------------------------------------------- #

    def _run_pt(self):
        with torch.no_grad():
            return self._model(**self._dummy_kwargs)

    def _run_trt(self):
        with torch.no_grad():
            return self._trt_model(*self._dummy_kwargs.values())

    def generate(self) -> None:
        with torch.no_grad():
            if self._trt_model is not None:
                out = self._trt_model(*self._dummy_kwargs.values())
            else:
                out = self._model(**self._dummy_kwargs)
        # CLIP outputs typically include image_embeds, text_embeds, logits_per_image.
        if hasattr(out, "logits_per_image"):
            print(f"[multimodal] logits_per_image: {out.logits_per_image.shape}")
        elif hasattr(out, "image_embeds"):
            print(f"[multimodal] image_embeds: {out.image_embeds.shape}")
        elif isinstance(out, torch.Tensor):
            print(f"[multimodal] output: {out.shape}")
        else:
            print(f"[multimodal] output type: {type(out)}")
