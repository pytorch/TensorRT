"""
Video diffusion strategy: CogVideoX, AnimateDiff, Stable Video Diffusion, etc.

Compiles only the denoising backbone (3D UNet or 3D DiT transformer).  The
rest of the pipeline (VAE, text encoder, scheduler) stays in PyTorch.

Video backbones operate on 5D tensors (B, C, F, H, W) where F is frame count.
The same forward-pre-hook capture pattern from DiffusionStrategy is used to
capture the actual call signature, since architectures differ significantly:

  CogVideoX          : 3D DiT transformer, causal VAE with temporal compression
  AnimateDiff        : UNet2D + MotionAdapter (spatial + temporal attention)
  Stable Video Diff  : UNetSpatioTemporalConditionModel (SVD)
  I2VGen-XL          : dual-text-and-image conditioned 3D UNet

Rather than hard-coding dummy inputs per architecture, we run one short
inference pass and capture the actual backbone-call args via a forward
pre-hook.  Those captured tensors are reused for export.
"""

from __future__ import annotations

import timeit
from typing import Optional

import torch
import torch_tensorrt
from common.bench import warmup_and_time
from common.compile import (
    compile_kwargs,
    compile_with_trt,
    maybe_save_exported_program,
    model_dtype,
    safe_export,
    serialize_trt_engine,
)
from common.metrics import print_table, report_videos_per_sec
from strategies.base import ModelStrategy, RunConfig

# --------------------------------------------------------------------------- #
# Pipeline introspection helpers
# --------------------------------------------------------------------------- #


def _get_backbone(pipe) -> tuple[torch.nn.Module, str]:
    """Return (backbone_module, attribute_name) for the denoising backbone."""
    if hasattr(pipe, "transformer") and pipe.transformer is not None:
        return pipe.transformer, "transformer"
    if hasattr(pipe, "unet") and pipe.unet is not None:
        return pipe.unet, "unet"
    raise AttributeError(
        "No denoising backbone found on the pipeline "
        "(expected pipe.transformer or pipe.unet)."
    )


def _is_image_to_video(pipe) -> bool:
    cls = type(pipe).__name__.lower()
    return any(k in cls for k in ("stablevideo", "img2vid", "imagetovideo", "svd"))


def _build_pipeline_call_kwargs(
    pipe,
    num_frames: int,
    image_size: int,
    device: str,
    dtype: torch.dtype,
) -> dict:
    """Return the kwargs needed to drive one pipeline step for backbone capture."""
    if _is_image_to_video(pipe):
        # SVD and similar pipelines need a conditioning image tensor.
        # Using a random uint8 PIL image is fine for shape capture only.
        try:
            import numpy as np
            from PIL import Image as PILImage

            arr = torch.randint(
                0, 256, (image_size, image_size, 3), dtype=torch.uint8
            ).numpy()
            img = PILImage.fromarray(arr.astype("uint8"))
        except ImportError:
            img = None
        kwargs: dict = {
            "num_frames": num_frames,
            "decode_chunk_size": num_frames,
            "output_type": "latent",
        }
        if img is not None:
            kwargs["image"] = img
        return kwargs
    # Default: text-to-video
    return {
        "prompt": "a dog running in a park",
        "num_frames": num_frames,
        "output_type": "latent",
    }


def _capture_backbone_inputs(
    pipe,
    num_frames: int,
    image_size: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[tuple, dict]:
    """
    Run one denoising step and capture the backbone's positional and keyword
    args via a forward pre-hook.  Returns cloned (args, kwargs).
    """
    backbone, _ = _get_backbone(pipe)
    captured: dict = {"args": None, "kwargs": None}

    def _hook(module, args, kwargs):
        captured["args"] = tuple(
            a.detach().clone() if isinstance(a, torch.Tensor) else a for a in args
        )
        captured["kwargs"] = {
            k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
            for k, v in kwargs.items()
        }

    handle = backbone.register_forward_pre_hook(_hook, with_kwargs=True)
    call_kwargs = _build_pipeline_call_kwargs(
        pipe, num_frames, image_size, device, dtype
    )
    try:
        pipe(num_inference_steps=1, **call_kwargs)
    finally:
        handle.remove()

    if captured["args"] is None:
        raise RuntimeError("Failed to capture video backbone forward inputs.")
    return captured["args"], captured["kwargs"]


# --------------------------------------------------------------------------- #
# Strategy
# --------------------------------------------------------------------------- #


class VideoDiffusionStrategy(ModelStrategy):
    def __init__(self, cfg: RunConfig):
        super().__init__(cfg)
        self._pipe = None
        self._device = "cuda:0"
        self._model_dtype = model_dtype(cfg.precision, cfg.autocast)
        self._pt_backbone: Optional[torch.nn.Module] = None
        self._captured_args: tuple = ()
        self._captured_kwargs: dict = {}
        self._num_frames: int = cfg.num_frames

    # ---------------------------------------------------------------------- #

    def load(self) -> None:
        from diffusers import DiffusionPipeline

        print(f"[video_diffusion] Loading {self.cfg.model} ...")
        self._pipe = DiffusionPipeline.from_pretrained(
            self.cfg.model,
            torch_dtype=self._model_dtype,
        ).to(self._device)
        self._pipe.set_progress_bar_config(disable=True)
        _, attr = _get_backbone(self._pipe)
        print(f"[video_diffusion] Pipeline loaded (backbone: pipe.{attr}).")

    # ---------------------------------------------------------------------- #

    def compile(self) -> None:
        assert self._pipe is not None, "Call load() before compile()"
        backbone, attr = _get_backbone(self._pipe)
        print(f"[video_diffusion] Compiling pipe.{attr} ...")

        if self.cfg.mode == "compile":
            compiled = self._compile_torch_compile(backbone)
        else:
            compiled = self._compile_export(backbone)

        # Copy pipeline-required attributes that diffusers reads from the backbone.
        for attr_name in ("config", "dtype", "add_embedding", "device"):
            if hasattr(backbone, attr_name) and not hasattr(compiled, attr_name):
                try:
                    setattr(compiled, attr_name, getattr(backbone, attr_name))
                except Exception:
                    pass

        setattr(self._pipe, attr, compiled)
        print(f"[video_diffusion] Compiled backbone swapped into pipe.{attr}.")

    def _compile_torch_compile(self, backbone: torch.nn.Module) -> torch.nn.Module:
        options = compile_kwargs(self.cfg.precision, self.cfg.autocast)
        options.update(
            {"min_block_size": self.cfg.min_block_size, "debug": self.cfg.debug}
        )

        compiled = torch.compile(backbone, backend="torch_tensorrt", options=options)

        call_kwargs = _build_pipeline_call_kwargs(
            self._pipe,
            self._num_frames,
            self.cfg.image_size,
            self._device,
            self._model_dtype,
        )
        with torch.no_grad():
            self._pipe(num_inference_steps=1, **call_kwargs)
        torch.cuda.synchronize()
        return compiled

    def _compile_export(self, backbone: torch.nn.Module) -> torch.nn.Module:
        print(
            f"[video_diffusion] Capturing backbone inputs from a 1-step inference "
            f"({self._num_frames} frames) ..."
        )
        args, kwargs = _capture_backbone_inputs(
            self._pipe,
            num_frames=self._num_frames,
            image_size=self.cfg.image_size,
            device=self._device,
            dtype=self._model_dtype,
        )
        _summarize_captured(args, kwargs)

        self._pt_backbone = backbone
        self._captured_args = args
        self._captured_kwargs = kwargs

        print("[video_diffusion] torch.export.export ...")
        ep = safe_export(backbone, args=args, kwargs=kwargs)
        maybe_save_exported_program(
            self.cfg, ep, log_prefix="[video_diffusion:backbone]"
        )

        # Build TRT inputs: tensor args → positional Input; tensor kwargs → kwarg Input.
        trt_arg_inputs = [
            torch_tensorrt.Input(shape=t.shape, dtype=t.dtype)
            for t in args
            if isinstance(t, torch.Tensor)
        ]
        trt_kwarg_inputs = {
            k: (
                torch_tensorrt.Input(shape=v.shape, dtype=v.dtype)
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in kwargs.items()
        }
        flat_trt_inputs = trt_arg_inputs + [
            v for v in trt_kwarg_inputs.values() if isinstance(v, torch_tensorrt.Input)
        ]

        compiled = compile_with_trt(
            ep,
            inputs=flat_trt_inputs,
            precision=self.cfg.precision,
            autocast=self.cfg.autocast,
            min_block_size=self.cfg.min_block_size,
            debug=self.cfg.debug,
            offload_module_to_cpu=self.cfg.offload_module_to_cpu,
            engine_cache_dir=self.cfg.engine_cache_dir,
        )
        torch.cuda.synchronize()

        if self.cfg.save_engine:
            from common.compile import maybe_save_trt_module

            maybe_save_trt_module(
                self.cfg,
                compiled,
                arg_inputs=flat_trt_inputs,
                log_prefix="[video_diffusion]",
            )

        if self.cfg.save_trt_engine:
            _, backbone_attr = _get_backbone(self._pipe)
            try:
                engine_bytes = serialize_trt_engine(
                    ep,
                    arg_inputs=trt_arg_inputs,
                    kwarg_inputs=trt_kwarg_inputs or None,
                    precision=self.cfg.precision,
                    autocast=self.cfg.autocast,
                    min_block_size=self.cfg.min_block_size,
                    debug=self.cfg.debug,
                    offload_module_to_cpu=self.cfg.offload_module_to_cpu,
                )
                path = self.cfg.save_trt_engine
                with open(path, "wb") as f:
                    f.write(engine_bytes)
                print(
                    f"[video_diffusion:{backbone_attr}] Wrote {len(engine_bytes)} bytes → {path}"
                )
            except Exception as e:
                print(f"[video_diffusion:{backbone_attr}] FAILED to serialize: {e}")

        return compiled

    # ---------------------------------------------------------------------- #

    def benchmark(self) -> list[dict]:
        assert self._pipe is not None, "Call load() first"
        rows: list[dict] = []
        n_frames = self._num_frames
        n_steps = self.cfg.num_inference_steps
        n_videos = self.cfg.batch_size

        call_kwargs = _build_pipeline_call_kwargs(
            self._pipe, n_frames, self.cfg.image_size, self._device, self._model_dtype
        )
        call_kwargs["num_inference_steps"] = n_steps

        def _run():
            return self._pipe(**call_kwargs)

        for _ in range(3):
            _run()
        torch.cuda.synchronize()

        timings = []
        for _ in range(self.cfg.iterations):
            t0 = timeit.default_timer()
            _run()
            torch.cuda.synchronize()
            timings.append(timeit.default_timer() - t0)

        rows.append(
            report_videos_per_sec(
                n_videos=n_videos * self.cfg.iterations,
                elapsed_s=sum(timings),
                num_frames=n_frames,
                n_steps=n_steps * self.cfg.iterations,
                backend=f"torch_tensorrt[{self.cfg.mode}]",
                precision=self.cfg.precision,
            )
        )
        print_table(rows, title=f"Video diffusion benchmark – {self.cfg.model}")
        return rows

    # ---------------------------------------------------------------------- #

    def _run_pt(self):
        if self._pt_backbone is None:
            raise RuntimeError(
                "Backbone not captured.  Accuracy is only supported with "
                "--mode export (the default)."
            )
        with torch.no_grad():
            return self._pt_backbone(*self._captured_args, **self._captured_kwargs)

    def _run_trt(self):
        backbone, _ = _get_backbone(self._pipe)
        with torch.no_grad():
            return backbone(*self._captured_args, **self._captured_kwargs)

    def generate(self) -> None:
        call_kwargs = _build_pipeline_call_kwargs(
            self._pipe,
            self._num_frames,
            self.cfg.image_size,
            self._device,
            self._model_dtype,
        )
        call_kwargs["num_inference_steps"] = self.cfg.num_inference_steps

        with torch.no_grad():
            out = self._pipe(**call_kwargs)

        # Video pipelines return .frames (list of PIL images) or .frames as tensors.
        frames = getattr(out, "frames", None)
        if frames is not None:
            n = len(frames) if hasattr(frames, "__len__") else "?"
            print(
                f"[video_diffusion] Generated {n} frame batch(es) ({self._num_frames} frames each)."
            )
        else:
            print(f"[video_diffusion] Output type: {type(out)}")


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _summarize_captured(args: tuple, kwargs: dict) -> None:
    n_tensor_args = sum(1 for a in args if isinstance(a, torch.Tensor))
    n_tensor_kwargs = sum(1 for v in kwargs.values() if isinstance(v, torch.Tensor))
    tensor_shapes = [tuple(a.shape) for a in args if isinstance(a, torch.Tensor)] + [
        tuple(v.shape) for v in kwargs.values() if isinstance(v, torch.Tensor)
    ]
    print(
        f"[video_diffusion] Captured {n_tensor_args} positional + "
        f"{n_tensor_kwargs} keyword tensor args.  Shapes: {tensor_shapes}"
    )
