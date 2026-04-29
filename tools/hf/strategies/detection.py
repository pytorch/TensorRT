"""
Object detection / segmentation strategy.

Supports two model classes:

  DETR-family  (detr, conditional_detr, rt_detr, yolos, table_transformer,
                grounding_dino, owlvit, owlv2, …)
    Loaded via AutoModelForObjectDetection.
    Benchmarks the full model forward (pixel_values → logits + pred_boxes).

  SAM — Segment Anything Model
    Loaded via SamModel.  Only the image encoder is compiled (ViT backbone);
    the prompt encoder and mask decoder stay in PyTorch because they are tiny
    and require dynamic prompt inputs.

Default compilation mode is torch.compile (export mode is attempted but some
models — especially those using pixel_mask or complex cross-attention — may
not export cleanly; the strategy falls back to torch.compile automatically).

Metric: throughput (samples/sec) and latency.
"""

from __future__ import annotations

from typing import Optional

import torch
from common.bench import compute_stats, warmup_and_time
from common.compile import (
    compile_kwargs,
    compile_with_trt,
    model_dtype,
    safe_export,
)
from common.metrics import print_table, report_latency
from strategies.base import ModelStrategy, RunConfig

_LOG = "[detection]"

_SAM_TYPES = {"sam", "sam2"}


class DetectionStrategy(ModelStrategy):
    def __init__(self, cfg: RunConfig):
        super().__init__(cfg)
        self._model: Optional[torch.nn.Module] = None
        self._trt_model: Optional[torch.nn.Module] = None
        self._processor = None
        self._dummy_pixel_values: Optional[torch.Tensor] = None
        self._dummy_pixel_mask: Optional[torch.Tensor] = None
        self._device = "cuda:0"
        self._model_dtype = model_dtype(cfg.precision, cfg.autocast)
        self._is_sam = False
        self._label = "detector"

    # ------------------------------------------------------------------ #

    def load(self) -> None:
        from transformers import AutoConfig

        cfg_hf = AutoConfig.from_pretrained(self.cfg.model, trust_remote_code=False)
        model_type = getattr(cfg_hf, "model_type", "").lower()
        self._is_sam = model_type in _SAM_TYPES or "sam" in self.cfg.model.lower()

        if self._is_sam:
            self._load_sam()
        else:
            self._load_detector()

    def _load_detector(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        print(f"{_LOG} Loading {self.cfg.model} ...")
        self._processor = AutoImageProcessor.from_pretrained(self.cfg.model)
        self._model = (
            AutoModelForObjectDetection.from_pretrained(
                self.cfg.model,
                ignore_mismatched_sizes=True,
            )
            .eval()
            .to(self._model_dtype)
            .to(self._device)
        )
        self._dummy_pixel_values, self._dummy_pixel_mask = self._build_detector_inputs()
        self._label = "detector"
        print(f"{_LOG} Loaded. pixel_values: {self._dummy_pixel_values.shape}")

    def _load_sam(self) -> None:
        from transformers import SamModel, SamProcessor

        print(f"{_LOG} Loading SAM {self.cfg.model} (image encoder only) ...")
        self._processor = SamProcessor.from_pretrained(self.cfg.model)
        full = (
            SamModel.from_pretrained(self.cfg.model)
            .eval()
            .to(self._model_dtype)
            .to(self._device)
        )
        # Keep the full model around for generate(); only compile image_encoder.
        self._sam_full = full
        self._model = full.image_encoder
        # SAM image encoder always expects [B, 3, 1024, 1024].
        self._dummy_pixel_values = torch.zeros(
            self.cfg.batch_size,
            3,
            1024,
            1024,
            dtype=self._model_dtype,
            device=self._device,
        )
        self._label = "SAM image encoder"
        print(f"{_LOG} Loaded. image encoder input: {self._dummy_pixel_values.shape}")

    def _build_detector_inputs(self):
        """Run the processor on a dummy image to get correctly-sized tensors."""
        try:
            import numpy as np
            from PIL import Image

            dummy_pil = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
            out = self._processor(
                images=[dummy_pil] * self.cfg.batch_size,
                return_tensors="pt",
            )
            pv = out["pixel_values"].to(self._model_dtype).to(self._device)
            pm = out.get("pixel_mask")
            if pm is not None:
                pm = pm.to(self._device)
            return pv, pm
        except Exception:
            # Fallback: DETR's typical preprocessed size is 800×800.
            pv = torch.zeros(
                self.cfg.batch_size,
                3,
                800,
                800,
                dtype=self._model_dtype,
                device=self._device,
            )
            return pv, None

    # ------------------------------------------------------------------ #

    def compile(self) -> None:
        assert self._model is not None, "Call load() first"
        if self.cfg.mode == "compile":
            self._compile_torch_compile()
        else:
            try:
                self._compile_export()
            except Exception as e:
                print(
                    f"{_LOG} export failed ({type(e).__name__}: {str(e)[:120]}); "
                    "falling back to torch.compile."
                )
                self._compile_torch_compile()

    def _compile_torch_compile(self) -> None:
        print(f"{_LOG} torch.compile(backend='torch_tensorrt') ...")
        opts = compile_kwargs(self.cfg.precision, self.cfg.autocast)
        opts.update(
            {"min_block_size": self.cfg.min_block_size, "debug": self.cfg.debug}
        )
        if self.cfg.optimization_level is not None:
            opts["optimization_level"] = self.cfg.optimization_level
        self._trt_model = torch.compile(
            self._model, backend="torch_tensorrt", options=opts
        )
        with torch.no_grad():
            _ = self._forward(self._trt_model)
        torch.cuda.synchronize()
        print(f"{_LOG} torch.compile done.")

    def _compile_export(self) -> None:
        import torch_tensorrt

        print(f"{_LOG} torch.export.export ...")
        batch = torch.export.Dim.AUTO

        if self._is_sam:
            dyn = ({0: batch},)
            ep = safe_export(
                self._model,
                args=(self._dummy_pixel_values,),
                dynamic_shapes=dyn,
            )
            trt_inputs = [
                torch_tensorrt.Input(
                    shape=self._dummy_pixel_values.shape,
                    dtype=self._dummy_pixel_values.dtype,
                )
            ]
        else:
            args = self._forward_args()
            dyn = tuple({0: batch} for _ in args)
            ep = safe_export(self._model, args=args, dynamic_shapes=dyn)
            trt_inputs = [
                torch_tensorrt.Input(shape=t.shape, dtype=t.dtype) for t in args
            ]

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
        print(f"{_LOG} TRT compile done.")

    # ------------------------------------------------------------------ #

    def _forward_args(self) -> tuple:
        """Positional args for export (pixel_values [, pixel_mask])."""
        if self._dummy_pixel_mask is not None:
            return (self._dummy_pixel_values, self._dummy_pixel_mask)
        return (self._dummy_pixel_values,)

    def _forward(self, model) -> object:
        """Unified forward call for both SAM encoder and detector models."""
        if self._is_sam:
            return model(self._dummy_pixel_values)
        if self._dummy_pixel_mask is not None:
            return model(
                pixel_values=self._dummy_pixel_values,
                pixel_mask=self._dummy_pixel_mask,
            )
        return model(pixel_values=self._dummy_pixel_values)

    # ------------------------------------------------------------------ #

    def benchmark(self) -> list[dict]:
        assert self._dummy_pixel_values is not None, "Call load() first"
        rows: list[dict] = []

        def _run_pt():
            with torch.no_grad():
                return self._forward(self._model)

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
                    return self._forward(self._trt_model)

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
            print(f"{_LOG} torch.compile(backend='inductor') ...")
            ind_model = torch.compile(self._model, backend="inductor")

            def _run_ind():
                with torch.no_grad():
                    return self._forward(ind_model)

            ind_t = warmup_and_time(_run_ind, (), iterations=self.cfg.iterations)
            rows.append(
                report_latency(
                    compute_stats(ind_t, self.cfg.batch_size),
                    backend="inductor",
                    batch_size=self.cfg.batch_size,
                    precision=self.cfg.precision,
                )
            )

        print_table(
            rows, title=f"Detection benchmark – {self.cfg.model} ({self._label})"
        )
        return rows

    # ------------------------------------------------------------------ #

    def _run_pt(self):
        with torch.no_grad():
            return self._forward(self._model)

    def _run_trt(self):
        with torch.no_grad():
            return self._forward(self._trt_model)

    def generate(self) -> None:
        """For SAM: run a full encode+decode with a dummy point prompt."""
        if not self._is_sam:
            print(f"{_LOG} generate() not implemented for detector models.")
            return
        import numpy as np
        from PIL import Image

        dummy_image = Image.fromarray(np.zeros((1024, 1024, 3), dtype=np.uint8))
        inputs = self._processor(
            images=dummy_image,
            input_points=[[[512, 512]]],
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._sam_full(**inputs)
        print(f"{_LOG} SAM masks shape: {out.pred_masks.shape}")
