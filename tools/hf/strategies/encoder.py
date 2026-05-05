"""
Encoder strategy: BERT, RoBERTa, ViT, ResNet, EfficientNet, etc.

Whole-model export.  These are the easiest models to compile because they
have no KV cache, no generation loop, and minimal control flow.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch_tensorrt
from common.bench import compute_stats, warmup_and_time
from common.compile import (
    PRECISION_MAP,
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

# HF model_type values that use vision inputs (pixel_values) rather than input_ids.
_VISION_TYPES = {
    "convnext",
    "efficientnet",
    "mobilenet_v2",
    "resnet",
    "swin",
    "vit",
}


def _is_vision_model(cfg) -> bool:
    return getattr(cfg, "model_type", "").lower() in _VISION_TYPES


def _build_dummy_inputs(
    model_cfg, batch_size: int, dtype: torch.dtype, device: str
) -> tuple:
    if _is_vision_model(model_cfg):
        image_size = getattr(model_cfg, "image_size", 224)
        if isinstance(image_size, (list, tuple)):
            h, w = image_size[0], image_size[1]
        else:
            h = w = int(image_size)
        nc = getattr(model_cfg, "num_channels", 3)
        return (torch.randn(batch_size, nc, h, w, dtype=dtype).to(device),)

    seq_len = 128
    vocab_size = getattr(model_cfg, "vocab_size", 30522)
    input_ids = torch.randint(
        0, vocab_size, (batch_size, seq_len), dtype=torch.int64
    ).to(device)
    attention_mask = torch.ones_like(input_ids)
    return (input_ids, attention_mask)


class EncoderStrategy(ModelStrategy):
    def __init__(self, cfg: RunConfig):
        super().__init__(cfg)
        self._model: Optional[torch.nn.Module] = None
        self._trt_model: Optional[torch.nn.Module] = None
        self._dummy_inputs: Optional[tuple] = None
        self._model_cfg = None
        self._device = "cuda:0"
        # Path A casts in PyTorch; Path B keeps model FP32.
        self._model_dtype = model_dtype(cfg.precision, cfg.autocast)

    # ---------------------------------------------------------------------- #

    def load(self) -> None:
        from transformers import AutoConfig, AutoModel

        print(f"[encoder] Loading {self.cfg.model} ...")
        self._model_cfg = AutoConfig.from_pretrained(self.cfg.model)

        self._model = (
            AutoModel.from_pretrained(
                self.cfg.model,
                attn_implementation="eager",
                ignore_mismatched_sizes=True,
            )
            .eval()
            .to(self._model_dtype)
            .to(self._device)
        )

        self._dummy_inputs = _build_dummy_inputs(
            self._model_cfg,
            self.cfg.batch_size,
            self._model_dtype,
            self._device,
        )
        print(
            f"[encoder] Model loaded.  Input shapes: {[t.shape for t in self._dummy_inputs]}"
        )

    # ---------------------------------------------------------------------- #

    def compile(self) -> None:
        assert self._model is not None, "Call load() before compile()"

        if self.cfg.mode == "compile":
            self._compile_torch_compile()
        else:
            self._compile_export()

    def _compile_torch_compile(self) -> None:
        print("[encoder] torch.compile(backend='torch_tensorrt') ...")
        options = compile_kwargs(self.cfg.precision, self.cfg.autocast)
        options.update(
            {"min_block_size": self.cfg.min_block_size, "debug": self.cfg.debug}
        )

        self._trt_model = torch.compile(
            self._model,
            backend="torch_tensorrt",
            options=options,
        )
        with torch.no_grad():
            _ = self._trt_model(*self._dummy_inputs)
        torch.cuda.synchronize()
        print("[encoder] torch.compile done.")

    def _compile_export(self) -> None:
        dummy = self._dummy_inputs

        # Use Dim.AUTO so torch.export can specialize/dynamicize as needed
        # (some HF models specialize batch=1 to a constant in their forward).
        batch = torch.export.Dim.AUTO
        if _is_vision_model(self._model_cfg):
            dyn = ({0: batch},)
        else:
            dyn = ({0: batch}, {0: batch})

        print("[encoder] torch.export.export ...")
        ep = safe_export(self._model, args=dummy, dynamic_shapes=dyn)
        maybe_save_exported_program(self.cfg, ep, log_prefix="[encoder]")

        print("[encoder] torch_tensorrt.dynamo.compile ...")
        trt_inputs = [torch_tensorrt.Input(shape=t.shape, dtype=t.dtype) for t in dummy]
        self._trt_model = compile_with_trt(
            ep,
            inputs=trt_inputs,
            precision=self.cfg.precision,
            autocast=self.cfg.autocast,
            min_block_size=self.cfg.min_block_size,
            debug=self.cfg.debug,
            offload_module_to_cpu=self.cfg.offload_module_to_cpu,
            engine_cache_dir=self.cfg.engine_cache_dir,
            optimization_level=self.cfg.optimization_level,
        )
        torch.cuda.synchronize()
        print("[encoder] TRT compile done.")

        maybe_save_trt_module(
            self.cfg,
            self._trt_model,
            arg_inputs=trt_inputs,
            example_arg_inputs=list(dummy),
            log_prefix="[encoder]",
        )
        maybe_save_trt_engine(self.cfg, ep, trt_inputs, log_prefix="[encoder]")

    # ---------------------------------------------------------------------- #

    def benchmark(self) -> list[dict]:
        assert self._dummy_inputs is not None, "Call load() first"
        rows: list[dict] = []

        with torch.no_grad():
            pt_t = warmup_and_time(
                lambda *a: self._model(*a),
                self._dummy_inputs,
                iterations=self.cfg.iterations,
            )
        rows.append(
            report_latency(
                compute_stats(pt_t, self.cfg.batch_size),
                backend="pytorch",
                batch_size=self.cfg.batch_size,
                precision=self.cfg.precision,
            )
        )

        if self._trt_model is not None:
            with torch.no_grad():
                trt_t = warmup_and_time(
                    lambda *a: self._trt_model(*a),
                    self._dummy_inputs,
                    iterations=self.cfg.iterations,
                )
            rows.append(
                report_latency(
                    compute_stats(trt_t, self.cfg.batch_size),
                    backend=f"torch_tensorrt[{self.cfg.mode}]",
                    batch_size=self.cfg.batch_size,
                    precision=self.cfg.precision,
                )
            )

        if self.cfg.inductor:
            print("[encoder] torch.compile(backend='inductor') ...")
            ind_model = torch.compile(self._model, backend="inductor")
            with torch.no_grad():
                ind_t = warmup_and_time(
                    lambda *a: ind_model(*a),
                    self._dummy_inputs,
                    iterations=self.cfg.iterations,
                )
            rows.append(
                report_latency(
                    compute_stats(ind_t, self.cfg.batch_size),
                    backend="inductor",
                    batch_size=self.cfg.batch_size,
                    precision=self.cfg.precision,
                )
            )

        print_table(rows, title=f"Encoder benchmark – {self.cfg.model}")
        return rows

    # ---------------------------------------------------------------------- #

    def _run_pt(self):
        with torch.no_grad():
            return self._model(*self._dummy_inputs)

    def _run_trt(self):
        with torch.no_grad():
            return self._trt_model(*self._dummy_inputs)

    def generate(self) -> None:
        model = self._trt_model if self._trt_model is not None else self._model
        with torch.no_grad():
            out = model(*self._dummy_inputs)
        if hasattr(out, "last_hidden_state"):
            print(f"[encoder] last_hidden_state: {out.last_hidden_state.shape}")
        elif hasattr(out, "logits"):
            print(f"[encoder] logits: {out.logits.shape}")
        elif isinstance(out, torch.Tensor):
            print(f"[encoder] output: {out.shape}")
        else:
            print(f"[encoder] output type: {type(out)}")
