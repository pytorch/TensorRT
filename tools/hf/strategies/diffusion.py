"""
Diffusion strategy: Stable Diffusion, SDXL, FLUX, etc.

Compiles the full compute-heavy pipeline with TRT:
  - denoising backbone  (UNet or DiT transformer)
  - text encoder(s)     (pipe.text_encoder, pipe.text_encoder_2 if present)
  - VAE decoder         (pipe.vae.decoder)

UNet/DiT signatures vary widely across architectures:
  SD 1.x/2.x : (sample, timestep, encoder_hidden_states)
  SDXL       : ... + (added_cond_kwargs={"text_embeds", "time_ids"})
  FLUX       : (hidden_states, timestep, guidance, pooled_projections,
                encoder_hidden_states, txt_ids, img_ids)

Rather than hard-code dummy inputs per architecture, we run one short
inference pass (output_type="pil" to trigger VAE decode) and capture
every component's actual call args via forward pre-hooks.  Those captured
tensors are used as the export args.

Components that fail to compile fall back to PyTorch silently so a partial
TRT pipeline is still faster than a full PyTorch one.
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
from common.metrics import print_table, report_images_per_sec
from strategies.base import ModelStrategy, RunConfig


def _suffix_engine_path(base_path: str, suffix: str) -> str:
    import os

    root, ext = os.path.splitext(base_path)
    return f"{root}.{suffix}{ext}" if ext else f"{root}.{suffix}"


def _get_backbone(pipe) -> tuple[torch.nn.Module, str]:
    """Return (backbone_module, attribute_name) for the denoising backbone."""
    if hasattr(pipe, "transformer") and pipe.transformer is not None:
        return pipe.transformer, "transformer"
    if hasattr(pipe, "unet") and pipe.unet is not None:
        return pipe.unet, "unet"
    raise AttributeError(
        "No denoising backbone found on the pipeline (expected pipe.transformer or pipe.unet)."
    )


def _clone(x):
    return x.detach().clone() if isinstance(x, torch.Tensor) else x


def _capture_all_inputs(pipe, prompt: str, num_steps: int = 1) -> dict[str, tuple]:
    """
    Run one full pipeline step (including VAE decode) and capture the actual
    call args/kwargs for every compile target via forward pre-hooks.

    Returns a dict keyed by component name:
      "backbone", "text_encoder", "text_encoder_2", "vae_decoder"
    Each value is (args_tuple, kwargs_dict) of cloned tensors.
    """
    backbone, _ = _get_backbone(pipe)
    captured: dict[str, tuple] = {}

    def make_hook(key: str):
        def _hook(module, args, kwargs):
            if key not in captured:  # only first call per component
                captured[key] = (
                    tuple(_clone(a) for a in args),
                    {k: _clone(v) for k, v in kwargs.items()},
                )

        return _hook

    hooks = [
        backbone.register_forward_pre_hook(make_hook("backbone"), with_kwargs=True)
    ]
    for attr in ("text_encoder", "text_encoder_2"):
        te = getattr(pipe, attr, None)
        if te is not None:
            hooks.append(
                te.register_forward_pre_hook(make_hook(attr), with_kwargs=True)
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

    try:
        # output_type="pil" triggers the VAE decode path so vae.decoder is called.
        pipe(prompt, num_inference_steps=num_steps, output_type="pil")
    finally:
        for h in hooks:
            h.remove()

    if "backbone" not in captured:
        raise RuntimeError("Failed to capture backbone forward inputs.")
    return captured


def _trt_inputs_from(args, kwargs) -> tuple[list, dict]:
    """Split captured (args, kwargs) into (flat_trt_inputs, kwarg_inputs) for compile_with_trt."""
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


class DiffusionStrategy(ModelStrategy):
    def __init__(self, cfg: RunConfig):
        super().__init__(cfg)
        self._pipe = None
        self._device = "cuda:0"
        self._model_dtype = model_dtype(cfg.precision, cfg.autocast)
        # Stashed for accuracy comparison: original PT backbone + captured inputs.
        self._pt_backbone = None
        self._captured_args: tuple = ()
        self._captured_kwargs: dict = {}
        # Original (pre-compilation) module refs needed for raw engine serialization.
        self._orig_text_encoder = None
        self._orig_text_encoder_2 = None
        self._orig_vae_decoder = None

    # ---------------------------------------------------------------------- #

    def load(self) -> None:
        from diffusers import AutoPipelineForText2Image

        print(f"[diffusion] Loading {self.cfg.model} ...")
        self._pipe = AutoPipelineForText2Image.from_pretrained(
            self.cfg.model,
            torch_dtype=self._model_dtype,
        ).to(self._device)
        self._pipe.set_progress_bar_config(disable=True)
        print("[diffusion] Pipeline loaded.")

    # ---------------------------------------------------------------------- #

    def compile(self) -> None:
        assert self._pipe is not None, "Call load() before compile()"

        # Save original module refs before any compilation so _save_companion_engines
        # can still serialize the original PyTorch modules as raw TRT engine bytes.
        self._orig_text_encoder = getattr(self._pipe, "text_encoder", None)
        self._orig_text_encoder_2 = getattr(self._pipe, "text_encoder_2", None)
        vae = getattr(self._pipe, "vae", None)
        self._orig_vae_decoder = (
            getattr(vae, "decoder", None) if vae is not None else None
        )

        if self.cfg.mode == "compile":
            self._compile_all_torch_compile()
        else:
            self._compile_all_export()

        if self.cfg.save_trt_engine and self.cfg.mode == "export":
            self._save_companion_engines()

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

        print(f"[diffusion] torch.compile pipe.{attr} ...")
        compiled_bb = torch.compile(backbone, backend="torch_tensorrt", options=opts)
        self._copy_backbone_attrs(backbone, compiled_bb)
        setattr(self._pipe, attr, compiled_bb)

        for te_attr in ("text_encoder", "text_encoder_2"):
            te = getattr(self._pipe, te_attr, None)
            if te is not None:
                print(f"[diffusion] torch.compile pipe.{te_attr} ...")
                setattr(
                    self._pipe,
                    te_attr,
                    torch.compile(te, backend="torch_tensorrt", options=opts),
                )

        vae = getattr(self._pipe, "vae", None)
        if vae is not None and getattr(vae, "decoder", None) is not None:
            print("[diffusion] torch.compile pipe.vae.decoder ...")
            self._pipe.vae.decoder = torch.compile(
                vae.decoder, backend="torch_tensorrt", options=opts
            )

        print(
            "[diffusion] Warming up (triggers TRT compilation for all components) ..."
        )
        with torch.no_grad():
            self._pipe("warmup", num_inference_steps=1, output_type="pil")
        torch.cuda.synchronize()
        print("[diffusion] All components compiled.")

    # ---------------------------------------------------------------------- #
    # export path
    # ---------------------------------------------------------------------- #

    def _compile_all_export(self) -> None:
        backbone, attr = _get_backbone(self._pipe)

        print("[diffusion] Capturing inputs from a 1-step inference ...")
        captured = _capture_all_inputs(
            self._pipe, prompt="a photo of an astronaut", num_steps=1
        )
        print(f"[diffusion] Captured: {sorted(captured.keys())}")

        # ---- backbone ----
        bb_args, bb_kwargs = captured["backbone"]
        self._pt_backbone = backbone
        self._captured_args = bb_args
        self._captured_kwargs = bb_kwargs

        print(f"[diffusion] Exporting + compiling pipe.{attr} ...")
        ep = safe_export(backbone, args=bb_args, kwargs=bb_kwargs)
        maybe_save_exported_program(self.cfg, ep, log_prefix=f"[diffusion:{attr}]")
        flat, kw_in = _trt_inputs_from(bb_args, bb_kwargs)
        compiled_bb = self._do_compile_with_trt(
            ep, flat, log_prefix=f"[diffusion:{attr}]"
        )
        self._copy_backbone_attrs(backbone, compiled_bb)
        setattr(self._pipe, attr, compiled_bb)

        if self.cfg.save_trt_engine:
            try:
                self._serialize_to_path(
                    ep,
                    _suffix_engine_path(self.cfg.save_trt_engine, attr),
                    arg_inputs=[
                        torch_tensorrt.Input(shape=t.shape, dtype=t.dtype)
                        for t in bb_args
                        if isinstance(t, torch.Tensor)
                    ],
                    kwarg_inputs=kw_in or None,
                    log_prefix=f"[diffusion:{attr}]",
                )
            except Exception as e:
                print(f"[diffusion:{attr}] Engine serialization failed: {e}")

        # ---- text encoders ----
        for te_attr in ("text_encoder", "text_encoder_2"):
            te = getattr(self._pipe, te_attr, None)
            if te is None or te_attr not in captured:
                if te is not None:
                    print(
                        f"[diffusion:{te_attr}] Not called during capture — keeping PyTorch."
                    )
                continue
            te_args, te_kwargs = captured[te_attr]
            print(f"[diffusion] Exporting + compiling pipe.{te_attr} ...")
            try:
                ep_te = safe_export(te, args=te_args, kwargs=te_kwargs)
                flat_te, _ = _trt_inputs_from(te_args, te_kwargs)
                compiled_te = self._do_compile_with_trt(
                    ep_te, flat_te, log_prefix=f"[diffusion:{te_attr}]"
                )
                setattr(self._pipe, te_attr, compiled_te)
            except Exception as e:
                print(f"[diffusion:{te_attr}] Compile failed ({e}) — keeping PyTorch.")

        # ---- VAE decoder ----
        vae = getattr(self._pipe, "vae", None)
        if vae is not None and getattr(vae, "decoder", None) is not None:
            if "vae_decoder" not in captured:
                print(
                    "[diffusion:vae_decoder] Not called during capture — keeping PyTorch."
                )
            else:
                vae_args, vae_kwargs = captured["vae_decoder"]
                print("[diffusion] Exporting + compiling pipe.vae.decoder ...")
                try:
                    ep_vae = safe_export(vae.decoder, args=vae_args, kwargs=vae_kwargs)
                    flat_vae, _ = _trt_inputs_from(vae_args, vae_kwargs)
                    compiled_vae = self._do_compile_with_trt(
                        ep_vae, flat_vae, log_prefix="[diffusion:vae_decoder]"
                    )
                    self._pipe.vae.decoder = compiled_vae
                except Exception as e:
                    print(
                        f"[diffusion:vae_decoder] Compile failed ({e}) — keeping PyTorch."
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
    # Multi-engine serialization (--save-trt-engine)
    # ---------------------------------------------------------------------- #

    def _serialize_to_path(
        self, ep, path, *, arg_inputs, kwarg_inputs=None, log_prefix: str
    ) -> None:
        print(f"{log_prefix} Serializing to {path} ...")
        engine_bytes = serialize_trt_engine(
            ep,
            arg_inputs=list(arg_inputs),
            kwarg_inputs=kwarg_inputs,
            precision=self.cfg.precision,
            autocast=self.cfg.autocast,
            min_block_size=self.cfg.min_block_size,
            debug=self.cfg.debug,
            offload_module_to_cpu=self.cfg.offload_module_to_cpu,
            optimization_level=self.cfg.optimization_level,
        )
        with open(path, "wb") as f:
            f.write(engine_bytes)
        print(f"{log_prefix} Wrote {len(engine_bytes)} bytes.")

    def _save_companion_engines(self) -> None:
        """
        Export and serialize the text_encoder and VAE decoder as separate TRT engines.
        Uses the original PyTorch modules saved before compilation.
        """
        base = self.cfg.save_trt_engine
        assert base is not None

        if self._orig_text_encoder is not None:
            try:
                self._serialize_text_encoder(
                    self._orig_text_encoder,
                    _suffix_engine_path(base, "text_encoder"),
                )
            except Exception as e:
                print(f"[diffusion:text_encoder] Serialization failed: {e}")

        if self._orig_text_encoder_2 is not None:
            try:
                self._serialize_text_encoder(
                    self._orig_text_encoder_2,
                    _suffix_engine_path(base, "text_encoder_2"),
                )
            except Exception as e:
                print(f"[diffusion:text_encoder_2] Serialization failed: {e}")

        if self._orig_vae_decoder is not None:
            try:
                self._serialize_vae_decoder(
                    self._orig_vae_decoder,
                    _suffix_engine_path(base, "vae_decoder"),
                )
            except Exception as e:
                print(f"[diffusion:vae_decoder] Serialization failed: {e}")

    def _serialize_text_encoder(self, te, path: str) -> None:
        tcfg = getattr(te, "config", None)
        max_len = getattr(tcfg, "max_position_embeddings", 77)
        vocab = getattr(tcfg, "vocab_size", 49408)
        input_ids = torch.randint(
            0, vocab, (self.cfg.batch_size, max_len), dtype=torch.int64
        ).to(self._device)
        ep = safe_export(te, args=(input_ids,))
        trt_inputs = [
            torch_tensorrt.Input(shape=input_ids.shape, dtype=input_ids.dtype)
        ]
        self._serialize_to_path(
            ep, path, arg_inputs=trt_inputs, log_prefix="[diffusion:text_encoder]"
        )

    def _serialize_vae_decoder(self, decoder, path: str) -> None:
        vae = self._pipe.vae
        vcfg = getattr(vae, "config", None)
        latent_channels = getattr(vcfg, "latent_channels", 4)
        block_out_channels = getattr(vcfg, "block_out_channels", [128, 256, 512, 512])
        spatial_factor = 2 ** (len(block_out_channels) - 1)
        h_lat = self.cfg.image_size // spatial_factor
        w_lat = self.cfg.image_size // spatial_factor
        latents = torch.randn(
            self.cfg.batch_size, latent_channels, h_lat, w_lat, dtype=self._model_dtype
        ).to(self._device)
        ep = safe_export(decoder, args=(latents,))
        trt_inputs = [torch_tensorrt.Input(shape=latents.shape, dtype=latents.dtype)]
        self._serialize_to_path(
            ep, path, arg_inputs=trt_inputs, log_prefix="[diffusion:vae_decoder]"
        )

    # ---------------------------------------------------------------------- #

    def benchmark(self) -> list[dict]:
        assert self._pipe is not None, "Call load() first"
        rows: list[dict] = []
        prompt = "a photo of an astronaut riding a horse on Mars"
        n_steps = self.cfg.num_inference_steps
        n_images = self.cfg.batch_size

        def _run():
            return self._pipe(
                prompt,
                num_inference_steps=n_steps,
                num_images_per_prompt=n_images,
                output_type="latent",
            )

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
            report_images_per_sec(
                n_images=n_images * self.cfg.iterations,
                elapsed_s=sum(timings),
                n_steps=n_steps * self.cfg.iterations,
                backend=f"torch_tensorrt[{self.cfg.mode}]",
                precision=self.cfg.precision,
            )
        )
        print_table(rows, title=f"Diffusion benchmark – {self.cfg.model}")
        return rows

    # ---------------------------------------------------------------------- #

    def _run_pt(self):
        if self._pt_backbone is None:
            raise RuntimeError(
                "Backbone not captured.  Accuracy is only supported when "
                "--mode export was used (the default)."
            )
        with torch.no_grad():
            return self._pt_backbone(*self._captured_args, **self._captured_kwargs)

    def _run_trt(self):
        backbone, _ = _get_backbone(self._pipe)
        with torch.no_grad():
            return backbone(*self._captured_args, **self._captured_kwargs)

    def generate(self) -> None:
        prompt = "a photo of an astronaut riding a horse on Mars"
        image = self._pipe(
            prompt,
            num_inference_steps=self.cfg.num_inference_steps,
            num_images_per_prompt=1,
        ).images[0]
        out_path = "output.png"
        image.save(out_path)
        print(f"[diffusion] Image saved to {out_path}")
