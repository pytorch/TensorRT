"""
Vision-Language-Action model strategy: OpenVLA, SpatialVLA, TinyVLA, Prismatic.

VLAs are VLMs with an action-token vocabulary on top — input is an image plus
a natural-language task instruction; output logits decode into discretised
robot actions (typically 7-DoF end-effector deltas).

Architecturally identical to a VLM at the level torch_tensorrt cares about:
  vision encoder → projector → LLM decoder → logits

This strategy:
  - Loads via AutoModelForVision2Seq with trust_remote_code=True (most VLAs
    ship custom model code; OpenVLA, SpatialVLA, Prismatic all require it).
  - Uses a robotics-flavoured prompt for dummy inputs.
  - Compiles with torch.compile(backend="torch_tensorrt", dynamic=True) — the
    dynamic embedding merge step rules out export, same as VLMs.
  - Calls the model's `predict_action` method during generate() when present.

Metric: throughput (samples/sec) and latency for the action-prediction prefill.
"""

from __future__ import annotations

from typing import Optional

import torch
from common.bench import compute_stats, warmup_and_time
from common.compile import compile_kwargs, model_dtype
from common.metrics import print_table, report_latency
from strategies.base import ModelStrategy, RunConfig

_LOG = "[vla]"

_DEFAULT_INSTRUCTION = "pick up the red block and place it on the blue plate"
_DEFAULT_PROMPT = (
    f"In: What action should the robot take to {_DEFAULT_INSTRUCTION}?\nOut:"
)


class VLAStrategy(ModelStrategy):
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

        # Most VLAs (OpenVLA, SpatialVLA, Prismatic) ship custom model code on
        # the HF hub and need trust_remote_code=True.  This is intentional —
        # if the user benchmarks a VLA, they have already chosen to trust it.
        print(f"{_LOG} Loading {self.cfg.model} (trust_remote_code=True) ...")
        self._processor = AutoProcessor.from_pretrained(
            self.cfg.model, trust_remote_code=True
        )
        self._model = (
            AutoModelForVision2Seq.from_pretrained(
                self.cfg.model,
                torch_dtype=self._model_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
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
        """Build robot-flavoured dummy inputs (zero image + task prompt)."""
        import numpy as np
        from PIL import Image

        # OpenVLA / Prismatic resize to 224×224 internally; SpatialVLA uses
        # 384×384.  Hand the processor a 224×224 zero image and let it resize.
        dummy_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        # Try chat-template path first (newer VLAs) ...
        if hasattr(self._processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": f"What action should the robot take to {_DEFAULT_INSTRUCTION}?",
                        },
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

        # OpenVLA-style "In: ... Out:" prompt.
        try:
            inputs = self._processor(_DEFAULT_PROMPT, dummy_image, return_tensors="pt")
            return {
                k: v.to(self._device)
                for k, v in inputs.items()
                if isinstance(v, torch.Tensor)
            }
        except Exception:
            pass

        # Last-resort: plain processor call.
        inputs = self._processor(
            text=[_DEFAULT_PROMPT],
            images=[dummy_image],
            return_tensors="pt",
            padding=True,
        )
        return {
            k: v.to(self._device)
            for k, v in inputs.items()
            if isinstance(v, torch.Tensor)
        }

    # ------------------------------------------------------------------ #

    def compile(self) -> None:
        assert self._model is not None, "Call load() before compile()"
        if self.cfg.mode == "export":
            print(
                f"{_LOG} export mode is not supported for VLAs "
                "(dynamic image-token embedding merge). Falling back to torch.compile."
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

        print_table(rows, title=f"VLA benchmark – {self.cfg.model}")
        return rows

    # ------------------------------------------------------------------ #

    def generate(self) -> None:
        """Predict an action.  Uses model.predict_action() when available
        (OpenVLA / Prismatic), otherwise falls back to model.generate()."""
        import numpy as np
        from PIL import Image

        model = self._trt_model if self._trt_model is not None else self._model
        dummy_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        # OpenVLA / Prismatic API: model.predict_action(...)
        if hasattr(model, "predict_action"):
            inputs = self._processor(_DEFAULT_PROMPT, dummy_image, return_tensors="pt")
            inputs = {
                k: v.to(self._device)
                for k, v in inputs.items()
                if isinstance(v, torch.Tensor)
            }
            with torch.no_grad():
                # unnorm_key is dataset-specific; "bridge_orig" is OpenVLA's default.
                try:
                    action = model.predict_action(
                        **inputs, unnorm_key="bridge_orig", do_sample=False
                    )
                except TypeError:
                    action = model.predict_action(**inputs, do_sample=False)
            print(f"{_LOG} predicted action: {action}")
            return

        # Fallback: regular text generation.
        with torch.no_grad():
            out = model.generate(
                **self._dummy_inputs, max_new_tokens=16, do_sample=False
            )
        decoded = self._processor.batch_decode(out, skip_special_tokens=True)
        print(f"{_LOG} generate: {decoded[0][:200]}")

    def _run_pt(self):
        with torch.no_grad():
            return self._model(**self._dummy_inputs)

    def _run_trt(self):
        with torch.no_grad():
            return self._trt_model(**self._dummy_inputs)
