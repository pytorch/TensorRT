"""
Video diffusion strategy: CogVideoX, AnimateDiff, Stable Video Diffusion, etc.

Compiles the full compute-heavy pipeline with TRT:
  - denoising backbone  (3D UNet or 3D DiT transformer)
  - text encoder        (pipe.text_encoder if present)
  - VAE decoder         (pipe.vae.decoder if present)

Video backbones operate on 5D tensors (B, C, F, H, W) where F is frame count.
The same forward-pre-hook capture pattern captures all component inputs in one
pipeline pass (output_type="pt" to trigger VAE decode).

  CogVideoX          : 3D DiT transformer, causal VAE with temporal compression
  AnimateDiff        : UNet2D + MotionAdapter (spatial + temporal attention)
  Stable Video Diff  : UNetSpatioTemporalConditionModel (SVD)
  I2VGen-XL          : dual-text-and-image conditioned 3D UNet

Components that fail to compile fall back to PyTorch silently.
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
    pipe, num_frames: int, image_size: int, device: str, dtype: torch.dtype
) -> dict:
    """Return the kwargs needed to drive one pipeline step for capture."""
    if _is_image_to_video(pipe):
        try:
            import numpy as np
            from PIL import Image as PILImage

            arr = torch.randint(
                0, 256, (image_size, image_size, 3), dtype=torch.uint8
            ).numpy()
            img = PILImage.fromarray(arr.astype("uint8"))
        except ImportError:
            img = None
        kwargs: dict = {"num_frames": num_frames, "decode_chunk_size": num_frames}
        if img is not None:
            kwargs["image"] = img
        return kwargs
    return {"prompt": "a dog running in a park", "num_frames": num_frames}


def _clone(x):
    return x.detach().clone() if isinstance(x, torch.Tensor) else x


def _capture_all_inputs(
    pipe, num_frames: int, image_size: int, device: str, dtype: torch.dtype
) -> dict[str, tuple]:
    """
    Run one full pipeline pass (output_type="pt" triggers VAE decode) and capture
    the actual call args/kwargs for every compile target via forward pre-hooks.

    Returns a dict keyed by component name:
      "backbone", "text_encoder", "vae_decoder"
    Each value is (args_tuple, kwargs_dict) of cloned tensors.
    """
    backbone, _ = _get_backbone(pipe)
    captured: dict[str, tuple] = {}

    def make_hook(key: str):
        def _hook(module, args, kwargs):
            if key not in captured:
                captured[key] = (
                    tuple(_clone(a) for a in args),
                    {k: _clone(v) for k, v in kwargs.items()},
                )

        return _hook

    hooks = [
        backbone.register_forward_pre_hook(make_hook("backbone"), with_kwargs=True)
    ]
    te = getattr(pipe, "text_encoder", None)
    if te is not None:
        hooks.append(
            te.register_forward_pre_hook(make_hook("text_encoder"), with_kwargs=True)
        )
    vae = getattr(pipe, "vae", None)
    if vae is not None:
        dec = getattr(vae, "decoder", None)
        if dec is not None:
            hooks.append(
                dec.register_forward_pre_hook(
                    make_hook("vae_decoder"), with_kwargs=True
                )
            )

    call_kwargs = _build_pipeline_call_kwargs(
        pipe, num_frames, image_size, device, dtype
    )
    call_kwargs["output_type"] = "pt"  # trigger VAE decode

    try:
        pipe(num_inference_steps=1, **call_kwargs)
    finally:
        for h in hooks:
            h.remove()

    if "backbone" not in captured:
        raise RuntimeError("Failed to capture video backbone forward inputs.")
    return captured


def _trt_inputs_from(args, kwargs) -> tuple[list, dict]:
    arg_inputs = [
        torch_tensorrt.Input(shape=t.shape, dtype=t.dtype)
        for t in args
        if isinstance(t, torch.Tensor)
    ]
    kwarg_inputs = {
        k: (
            torch_tensorrt.Input(shape=v.shape, dtype=v.dtype)
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in kwargs.items()
    }
    flat = arg_inputs + [
        v for v in kwarg_inputs.values() if isinstance(v, torch_tensorrt.Input)
    ]
    return flat, kwarg_inputs


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
        # Original module refs for raw engine serialization.
        self._orig_text_encoder = None
        self._orig_vae_decoder = None

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

        self._orig_text_encoder = getattr(self._pipe, "text_encoder", None)
        vae = getattr(self._pipe, "vae", None)
        self._orig_vae_decoder = (
            getattr(vae, "decoder", None) if vae is not None else None
        )

        if self.cfg.mode == "compile":
            self._compile_all_torch_compile()
        else:
            self._compile_all_export()

    # ---------------------------------------------------------------------- #
    # torch.compile path
    # ---------------------------------------------------------------------- #

    def _trt_opts(self) -> dict:
        opts = compile_kwargs(self.cfg.precision, self.cfg.autocast)
        opts.update(
            {"min_block_size": self.cfg.min_block_size, "debug": self.cfg.debug}
        )
        if self.cfg.optimization_level is not None:
            opts["optimization_level"] = self.cfg.optimization_level
        return opts

    def _compile_all_torch_compile(self) -> None:
        opts = self._trt_opts()
        backbone, attr = _get_backbone(self._pipe)

        print(f"[video_diffusion] torch.compile pipe.{attr} ...")
        compiled_bb = torch.compile(backbone, backend="torch_tensorrt", options=opts)
        self._copy_backbone_attrs(backbone, compiled_bb)
        setattr(self._pipe, attr, compiled_bb)

        te = getattr(self._pipe, "text_encoder", None)
        if te is not None:
            print("[video_diffusion] torch.compile pipe.text_encoder ...")
            self._pipe.text_encoder = torch.compile(
                te, backend="torch_tensorrt", options=opts
            )

        vae = getattr(self._pipe, "vae", None)
        if vae is not None and getattr(vae, "decoder", None) is not None:
            print("[video_diffusion] torch.compile pipe.vae.decoder ...")
            self._pipe.vae.decoder = torch.compile(
                vae.decoder, backend="torch_tensorrt", options=opts
            )

        print(
            "[video_diffusion] Warming up (triggers TRT compilation for all components) ..."
        )
        call_kwargs = _build_pipeline_call_kwargs(
            self._pipe,
            self._num_frames,
            self.cfg.image_size,
            self._device,
            self._model_dtype,
        )
        call_kwargs["output_type"] = "pt"
        with torch.no_grad():
            self._pipe(num_inference_steps=1, **call_kwargs)
        torch.cuda.synchronize()
        print("[video_diffusion] All components compiled.")

    # ---------------------------------------------------------------------- #
    # export path
    # ---------------------------------------------------------------------- #

    def _compile_all_export(self) -> None:
        backbone, attr = _get_backbone(self._pipe)

        print(
            f"[video_diffusion] Capturing inputs from 1-step inference "
            f"({self._num_frames} frames, output_type='pt') ..."
        )
        captured = _capture_all_inputs(
            self._pipe,
            num_frames=self._num_frames,
            image_size=self.cfg.image_size,
            device=self._device,
            dtype=self._model_dtype,
        )
        print(f"[video_diffusion] Captured: {sorted(captured.keys())}")
        _summarize_captured(captured.get("backbone", ([], {})))

        # ---- backbone ----
        bb_args, bb_kwargs = captured["backbone"]
        self._pt_backbone = backbone
        self._captured_args = bb_args
        self._captured_kwargs = bb_kwargs

        print(f"[video_diffusion] Exporting + compiling pipe.{attr} ...")
        ep = safe_export(backbone, args=bb_args, kwargs=bb_kwargs)
        maybe_save_exported_program(
            self.cfg, ep, log_prefix=f"[video_diffusion:{attr}]"
        )
        flat, kw_in = _trt_inputs_from(bb_args, bb_kwargs)
        compiled_bb = self._do_compile_with_trt(
            ep, flat, log_prefix=f"[video_diffusion:{attr}]"
        )
        self._copy_backbone_attrs(backbone, compiled_bb)
        setattr(self._pipe, attr, compiled_bb)

        if self.cfg.save_trt_engine:
            try:
                engine_bytes = serialize_trt_engine(
                    ep,
                    arg_inputs=[
                        torch_tensorrt.Input(shape=t.shape, dtype=t.dtype)
                        for t in bb_args
                        if isinstance(t, torch.Tensor)
                    ],
                    kwarg_inputs=kw_in or None,
                    precision=self.cfg.precision,
                    autocast=self.cfg.autocast,
                    min_block_size=self.cfg.min_block_size,
                    debug=self.cfg.debug,
                    offload_module_to_cpu=self.cfg.offload_module_to_cpu,
                    optimization_level=self.cfg.optimization_level,
                )
                with open(self.cfg.save_trt_engine, "wb") as f:
                    f.write(engine_bytes)
                print(
                    f"[video_diffusion:{attr}] Wrote {len(engine_bytes)} bytes → {self.cfg.save_trt_engine}"
                )
            except Exception as e:
                print(f"[video_diffusion:{attr}] Engine serialization failed: {e}")

        # ---- text encoder ----
        te = getattr(self._pipe, "text_encoder", None)
        if te is not None:
            if "text_encoder" not in captured:
                print(
                    "[video_diffusion:text_encoder] Not called during capture — keeping PyTorch."
                )
            else:
                te_args, te_kwargs = captured["text_encoder"]
                print("[video_diffusion] Exporting + compiling pipe.text_encoder ...")
                try:
                    ep_te = safe_export(te, args=te_args, kwargs=te_kwargs)
                    flat_te, _ = _trt_inputs_from(te_args, te_kwargs)
                    compiled_te = self._do_compile_with_trt(
                        ep_te, flat_te, log_prefix="[video_diffusion:text_encoder]"
                    )
                    self._pipe.text_encoder = compiled_te
                except Exception as e:
                    print(
                        f"[video_diffusion:text_encoder] Compile failed ({e}) — keeping PyTorch."
                    )

        # ---- VAE decoder ----
        vae = getattr(self._pipe, "vae", None)
        if vae is not None and getattr(vae, "decoder", None) is not None:
            if "vae_decoder" not in captured:
                print(
                    "[video_diffusion:vae_decoder] Not called during capture — keeping PyTorch."
                )
            else:
                vae_args, vae_kwargs = captured["vae_decoder"]
                print("[video_diffusion] Exporting + compiling pipe.vae.decoder ...")
                try:
                    ep_vae = safe_export(vae.decoder, args=vae_args, kwargs=vae_kwargs)
                    flat_vae, _ = _trt_inputs_from(vae_args, vae_kwargs)
                    compiled_vae = self._do_compile_with_trt(
                        ep_vae, flat_vae, log_prefix="[video_diffusion:vae_decoder]"
                    )
                    self._pipe.vae.decoder = compiled_vae
                except Exception as e:
                    print(
                        f"[video_diffusion:vae_decoder] Compile failed ({e}) — keeping PyTorch."
                    )

    def _do_compile_with_trt(self, ep, flat_inputs, *, log_prefix: str):
        print(f"{log_prefix} torch_tensorrt.dynamo.compile ...")
        compiled = compile_with_trt(
            ep,
            inputs=flat_inputs,
            precision=self.cfg.precision,
            autocast=self.cfg.autocast,
            min_block_size=self.cfg.min_block_size,
            debug=self.cfg.debug,
            offload_module_to_cpu=self.cfg.offload_module_to_cpu,
            engine_cache_dir=self.cfg.engine_cache_dir,
            optimization_level=self.cfg.optimization_level,
        )
        torch.cuda.synchronize()
        return compiled

    @staticmethod
    def _copy_backbone_attrs(src, dst) -> None:
        for name in ("config", "dtype", "add_embedding", "device"):
            if hasattr(src, name) and not hasattr(dst, name):
                try:
                    setattr(dst, name, getattr(src, name))
                except Exception:
                    pass

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


def _summarize_captured(backbone_capture: tuple) -> None:
    args, kwargs = backbone_capture
    n_tensor_args = sum(1 for a in args if isinstance(a, torch.Tensor))
    n_tensor_kwargs = sum(1 for v in kwargs.values() if isinstance(v, torch.Tensor))
    tensor_shapes = [tuple(a.shape) for a in args if isinstance(a, torch.Tensor)] + [
        tuple(v.shape) for v in kwargs.values() if isinstance(v, torch.Tensor)
    ]
    print(
        f"[video_diffusion] Backbone: {n_tensor_args} positional + "
        f"{n_tensor_kwargs} keyword tensor args.  Shapes: {tensor_shapes}"
    )
