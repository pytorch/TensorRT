"""
Vision-Language Model strategy: LLaVA, PaliGemma, Qwen2-VL, SmolVLM, Idefics, etc.

Uses torch.compile(backend="torch_tensorrt", dynamic=True).  Export mode is not
supported for VLMs because the image-token embedding merge step contains Python
control flow that torch.export cannot handle without specialisation.

Metric: throughput (samples/sec) and latency for the prefill step
        (image + prompt text → full logit output tensor).
"""

from __future__ import annotations

from typing import Optional

import torch
from common.bench import compute_stats, warmup_and_time
from common.compile import compile_kwargs, model_dtype
from common.metrics import print_table, report_latency
from strategies.base import ModelStrategy, RunConfig

_LOG = "[vlm]"


class VLMStrategy(ModelStrategy):
    def __init__(self, cfg: RunConfig):
        super().__init__(cfg)
        self._model: Optional[torch.nn.Module] = None
        self._trt_model: Optional[torch.nn.Module] = None
        self._processor = None
        self._dummy_inputs: Optional[dict] = None
        self._device = "cuda:0"
        self._model_dtype = model_dtype(cfg.precision, cfg.autocast)

    # ------------------------------------------------------------------ #

    def load(self) -> None:
        from transformers import AutoModelForVision2Seq, AutoProcessor

        print(f"{_LOG} Loading {self.cfg.model} ...")
        self._processor = AutoProcessor.from_pretrained(
            self.cfg.model, trust_remote_code=False
        )
        self._model = (
            AutoModelForVision2Seq.from_pretrained(
                self.cfg.model,
                torch_dtype=self._model_dtype,
                trust_remote_code=False,
            )
            .eval()
            .to(self._device)
        )
        self._dummy_inputs = self._build_dummy_inputs()
        shapes = {
            k: tuple(v.shape)
            for k, v in self._dummy_inputs.items()
            if isinstance(v, torch.Tensor)
        }
        print(f"{_LOG} Model loaded. Input shapes: {shapes}")

    def _build_dummy_inputs(self) -> dict:
        """Build minimal dummy inputs via the processor.

        Tries the chat-template path first (works for LLaVA-NeXT, Qwen2-VL,
        SmolVLM, …), then falls back to a plain-text prompt for older models
        like LLaVA 1.5 and PaliGemma.
        """
        import numpy as np
        from PIL import Image

        dummy_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        # Prefer chat-template formatting where available.
        if hasattr(self._processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ]
            try:
                text = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self._processor(
                    text=[text],
                    images=[dummy_image],
                    return_tensors="pt",
                    padding=True,
                )
                return {
                    k: v.to(self._device)
                    for k, v in inputs.items()
                    if isinstance(v, torch.Tensor)
                }
            except Exception:
                pass

        # Fallback: plain text with and without image token.
        for prompt in ["<image>\nDescribe this image.", "Describe this image."]:
            try:
                inputs = self._processor(
                    text=[prompt],
                    images=[dummy_image],
                    return_tensors="pt",
                    padding=True,
                )
                return {
                    k: v.to(self._device)
                    for k, v in inputs.items()
                    if isinstance(v, torch.Tensor)
                }
            except Exception:
                continue

        raise RuntimeError(
            f"{_LOG} Could not build dummy inputs for {self.cfg.model}. "
            "The processor may require a custom format — pass --task to override family."
        )

    # ------------------------------------------------------------------ #

    def compile(self) -> None:
        assert self._model is not None, "Call load() before compile()"
        if self.cfg.mode == "export":
            print(
                f"{_LOG} export mode is not supported for VLMs "
                "(dynamic embedding merge requires Python control flow). "
                "Falling back to torch.compile."
            )
        self._compile_torch_compile()

    def _compile_torch_compile(self) -> None:
        print(f"{_LOG} torch.compile(backend='torch_tensorrt', dynamic=True) ...")
        opts = compile_kwargs(self.cfg.precision, self.cfg.autocast)
        opts.update(
            {"min_block_size": self.cfg.min_block_size, "debug": self.cfg.debug}
        )
        if self.cfg.optimization_level is not None:
            opts["optimization_level"] = self.cfg.optimization_level
        self._trt_model = torch.compile(
            self._model,
            backend="torch_tensorrt",
            dynamic=True,
            options=opts,
        )
        # Trigger compilation before timing so warmup is not skewed.
        print(f"{_LOG} Triggering compilation (first forward pass) ...")
        with torch.no_grad():
            _ = self._trt_model(**self._dummy_inputs)
        torch.cuda.synchronize()
        print(f"{_LOG} torch.compile done.")

    # ------------------------------------------------------------------ #

    def benchmark(self) -> list[dict]:
        assert self._dummy_inputs is not None, "Call load() first"
        rows: list[dict] = []

        def _run_pt():
            with torch.no_grad():
                return self._model(**self._dummy_inputs)

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

            def _run_trt():
                with torch.no_grad():
                    return self._trt_model(**self._dummy_inputs)

            trt_t = warmup_and_time(_run_trt, (), iterations=self.cfg.iterations)
            rows.append(
                report_latency(
                    compute_stats(trt_t, self.cfg.batch_size),
                    backend=f"torch_tensorrt[{self.cfg.mode}]",
                    batch_size=self.cfg.batch_size,
                    precision=self.cfg.precision,
                )
            )

        if self.cfg.inductor:
            print(f"{_LOG} torch.compile(backend='inductor', dynamic=True) ...")
            ind_model = torch.compile(self._model, backend="inductor", dynamic=True)

            def _run_ind():
                with torch.no_grad():
                    return ind_model(**self._dummy_inputs)

            ind_t = warmup_and_time(_run_ind, (), iterations=self.cfg.iterations)
            rows.append(
                report_latency(
                    compute_stats(ind_t, self.cfg.batch_size),
                    backend="inductor",
                    batch_size=self.cfg.batch_size,
                    precision=self.cfg.precision,
                )
            )

        print_table(rows, title=f"VLM benchmark – {self.cfg.model}")
        return rows

    # ------------------------------------------------------------------ #

    def generate(self) -> None:
        import numpy as np
        from PIL import Image

        model = self._trt_model if self._trt_model is not None else self._model
        dummy_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        if hasattr(self._processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ]
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = "<image>\nDescribe this image."
        inputs = self._processor(text=[text], images=[dummy_image], return_tensors="pt")
        inputs = {
            k: v.to(self._device)
            for k, v in inputs.items()
            if isinstance(v, torch.Tensor)
        }
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64)
        decoded = self._processor.batch_decode(out, skip_special_tokens=True)
        print(f"{_LOG} generate: {decoded[0][:300]}")

    def _run_pt(self):
        with torch.no_grad():
            return self._model(**self._dummy_inputs)

    def _run_trt(self):
        with torch.no_grad():
            return self._trt_model(**self._dummy_inputs)
