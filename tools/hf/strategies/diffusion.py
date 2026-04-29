"""
Diffusion strategy: Stable Diffusion, SDXL, FLUX, etc.

Compile only the denoising backbone (UNet or DiT transformer).  The rest
of the pipeline (scheduler, VAE, CLIP) stays in PyTorch.

UNet/DiT signatures vary widely across architectures:
  SD 1.x/2.x : (sample, timestep, encoder_hidden_states)
  SDXL       : ... + (added_cond_kwargs={"text_embeds", "time_ids"})
  FLUX       : (hidden_states, timestep, guidance, pooled_projections,
                encoder_hidden_states, txt_ids, img_ids)

Rather than hard-code dummy inputs per architecture, we run one short
inference pass and capture the actual backbone-call args via a forward
pre-hook.  Those captured tensors are then used as the export args.
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
    """
    Insert `.<suffix>` before the extension of base_path.
      foo.trt → foo.<suffix>.trt
      foo     → foo.<suffix>
    """
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


def _capture_backbone_inputs(pipe, prompt: str, num_steps: int = 1):
    """
    Run a single denoising step and capture the backbone's positional and
    keyword args via a pre-forward hook.  Returns (args, kwargs) cloned so
    they can be reused for export.
    """
    backbone, _ = _get_backbone(pipe)
    captured = {"args": None, "kwargs": None}

    def _hook(module, args, kwargs):
        captured["args"] = tuple(
            a.detach().clone() if isinstance(a, torch.Tensor) else a for a in args
        )
        captured["kwargs"] = {
            k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
            for k, v in kwargs.items()
        }

    handle = backbone.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        # 1 step is enough to capture the call signature.
        pipe(prompt, num_inference_steps=num_steps, output_type="latent")
    finally:
        handle.remove()

    if captured["args"] is None:
        raise RuntimeError("Failed to capture backbone forward inputs.")
    return captured["args"], captured["kwargs"]


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

    # ---------------------------------------------------------------------- #

    def load(self) -> None:
        from diffusers import AutoPipelineForText2Image

        print(f"[diffusion] Loading {self.cfg.model} ...")
        # When autocast=True we keep FP32 weights for the backbone; otherwise
        # use the target precision for both backbone and pipeline parts.
        torch_dtype = self._model_dtype
        self._pipe = AutoPipelineForText2Image.from_pretrained(
            self.cfg.model,
            torch_dtype=torch_dtype,
        ).to(self._device)
        self._pipe.set_progress_bar_config(disable=True)
        print("[diffusion] Pipeline loaded.")

    # ---------------------------------------------------------------------- #

    def compile(self) -> None:
        assert self._pipe is not None, "Call load() before compile()"
        backbone, attr = _get_backbone(self._pipe)
        print(f"[diffusion] Compiling pipe.{attr} ...")

        if self.cfg.mode == "compile":
            compiled = self._compile_torch_compile(backbone)
        else:
            compiled = self._compile_export(backbone)

        # Preserve attributes the diffusers pipeline reads on the backbone
        # (e.g. unet.config.sample_size, unet.dtype).
        for attr_name in ("config", "dtype", "add_embedding", "device"):
            if hasattr(backbone, attr_name) and not hasattr(compiled, attr_name):
                try:
                    setattr(compiled, attr_name, getattr(backbone, attr_name))
                except Exception:
                    pass

        setattr(self._pipe, attr, compiled)
        print(f"[diffusion] Compiled backbone swapped into pipe.{attr}")

        # When --save-trt-engine is set, also serialize text_encoder and
        # vae.decoder as separate engines so the full diffusion pipeline
        # can be deployed under the standalone TRT runtime.
        if self.cfg.save_trt_engine and self.cfg.mode == "export":
            self._save_companion_engines()

    def _compile_torch_compile(self, backbone: torch.nn.Module) -> torch.nn.Module:
        options = compile_kwargs(self.cfg.precision, self.cfg.autocast)
        options.update(
            {"min_block_size": self.cfg.min_block_size, "debug": self.cfg.debug}
        )

        compiled = torch.compile(backbone, backend="torch_tensorrt", options=options)

        # Trigger compilation by running a real pipeline step.
        with torch.no_grad():
            self._pipe("warmup", num_inference_steps=1, output_type="latent")
        torch.cuda.synchronize()
        return compiled

    def _compile_export(self, backbone: torch.nn.Module) -> torch.nn.Module:
        # Capture the actual backbone call signature from a one-step inference.
        print("[diffusion] Capturing backbone inputs from a 1-step inference ...")
        args, kwargs = _capture_backbone_inputs(
            self._pipe,
            prompt="a photo of an astronaut",
            num_steps=1,
        )
        print(
            f"[diffusion] Captured {len(args)} positional + {len(kwargs)} keyword args."
        )

        # Stash for accuracy comparison: keep the PT backbone reference and
        # the captured call args/kwargs so we can run both pre- and post-swap.
        self._pt_backbone = backbone
        self._captured_args = args
        self._captured_kwargs = kwargs

        # Static-shape export for the backbone.  Dynamic-shape support is
        # currently architecture-specific (FLUX has variable image-token count,
        # SD has fixed latent shape) — leave to a future per-arch override.
        print("[diffusion] torch.export.export ...")
        ep = safe_export(backbone, args=args, kwargs=kwargs)
        # Save the pre-TRT EP for the backbone if requested.  Companion EPs
        # for text_encoder/vae are saved under suffixed paths in
        # _save_companion_engines (only when --save-trt-engine is used).
        maybe_save_exported_program(self.cfg, ep, log_prefix="[diffusion:backbone]")

        # Split positional and keyword tensor inputs to match the EP's
        # input spec.  Non-tensor kwargs (None, bool, dict) were baked
        # into the EP as constants at export time and are not graph inputs.
        trt_arg_inputs = [
            torch_tensorrt.Input(shape=t.shape, dtype=t.dtype)
            for t in args
            if isinstance(t, torch.Tensor)
        ]
        # The EP's input spec includes ALL kwargs from the original call,
        # not just tensor kwargs (non-tensor ones became constants but are
        # still listed in the in_spec for tree flattening).  Provide an
        # entry for each: torch_tensorrt.Input for tensors, the original
        # value for everything else.
        trt_kwarg_inputs = {
            k: (
                torch_tensorrt.Input(shape=v.shape, dtype=v.dtype)
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in kwargs.items()
        }
        # For the compile() call we still need a flat list, but only
        # tensor-typed entries; compile()'s `inputs=` arg is positional.
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
                log_prefix="[diffusion]",
            )

        # Save the backbone engine to <base>.<attr>.trt; companion engines
        # (text_encoder, vae_decoder) are written separately by
        # _save_companion_engines() called from compile().
        # The backbone often hits "Failed to extract symbolic shape
        # expressions" because diffusion UNets/transformers have ops the
        # converter can't trace symbolically.  Treat that as non-fatal:
        # text_encoder + vae_decoder are useful even without the backbone.
        if self.cfg.save_trt_engine:
            _, backbone_attr = _get_backbone(self._pipe)
            try:
                self._serialize_to_path(
                    ep,
                    _suffix_engine_path(self.cfg.save_trt_engine, backbone_attr),
                    arg_inputs=trt_arg_inputs,
                    kwarg_inputs=trt_kwarg_inputs or None,
                    log_prefix=f"[diffusion:{backbone_attr}]",
                )
            except Exception as e:
                print(f"[diffusion:{backbone_attr}] FAILED: {e}")
                print(
                    f"[diffusion:{backbone_attr}] (continuing with companion engines)"
                )

        return compiled

    # ---------------------------------------------------------------------- #
    # Multi-engine save: text_encoder + backbone + vae_decoder
    # ---------------------------------------------------------------------- #

    def _serialize_to_path(
        self,
        ep,
        path: str,
        *,
        arg_inputs,
        kwarg_inputs=None,
        log_prefix: str,
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
        )
        with open(path, "wb") as f:
            f.write(engine_bytes)
        print(f"{log_prefix} Wrote {len(engine_bytes)} bytes.")

    def _save_companion_engines(self) -> None:
        """
        Export and serialize the text_encoder and VAE decoder as separate
        TRT engines.  Together with the backbone engine they cover the
        compute-heavy parts of a text-to-image pipeline; the scheduler and
        tokenizer remain in Python.
        """
        base = self.cfg.save_trt_engine
        assert base is not None

        # Text encoder ----------------------------------------------------- #
        if hasattr(self._pipe, "text_encoder") and self._pipe.text_encoder is not None:
            try:
                self._serialize_text_encoder(_suffix_engine_path(base, "text_encoder"))
            except Exception as e:
                print(f"[diffusion:text_encoder] FAILED: {e}")

        # VAE decoder ------------------------------------------------------ #
        if hasattr(self._pipe, "vae") and self._pipe.vae is not None:
            try:
                self._serialize_vae_decoder(_suffix_engine_path(base, "vae_decoder"))
            except Exception as e:
                print(f"[diffusion:vae_decoder] FAILED: {e}")

    def _serialize_text_encoder(self, path: str) -> None:
        text_encoder = self._pipe.text_encoder
        tcfg = getattr(text_encoder, "config", None)
        max_len = getattr(tcfg, "max_position_embeddings", 77)
        vocab = getattr(tcfg, "vocab_size", 49408)
        input_ids = torch.randint(
            0, vocab, (self.cfg.batch_size, max_len), dtype=torch.int64
        ).to(self._device)

        print("[diffusion:text_encoder] torch.export.export ...")
        ep = safe_export(text_encoder, args=(input_ids,))
        trt_inputs = [
            torch_tensorrt.Input(shape=input_ids.shape, dtype=input_ids.dtype)
        ]
        self._serialize_to_path(
            ep,
            path,
            arg_inputs=trt_inputs,
            log_prefix="[diffusion:text_encoder]",
        )

    def _serialize_vae_decoder(self, path: str) -> None:
        vae = self._pipe.vae
        # `vae.decode(z)` is what the pipeline calls; its underlying module
        # is `vae.decoder`.  We export `vae.decoder` directly to skip the
        # diffusers-specific output wrapping and tiling logic.
        decoder = getattr(vae, "decoder", None)
        if decoder is None:
            print("[diffusion:vae_decoder] No vae.decoder submodule; skipping.")
            return

        vcfg = getattr(vae, "config", None)
        latent_channels = getattr(vcfg, "latent_channels", 4)
        # Spatial downsampling factor = 2^(num_decoder_blocks - 1).
        # `vae.config.scaling_factor` is the latent value rescaling (float
        # like 0.18215), NOT the spatial factor — don't confuse them.
        block_out_channels = getattr(vcfg, "block_out_channels", [128, 256, 512, 512])
        spatial_factor = 2 ** (len(block_out_channels) - 1)
        h_lat = self.cfg.image_size // spatial_factor
        w_lat = self.cfg.image_size // spatial_factor

        latents = torch.randn(
            self.cfg.batch_size,
            latent_channels,
            h_lat,
            w_lat,
            dtype=self._model_dtype,
        ).to(self._device)

        print("[diffusion:vae_decoder] torch.export.export ...")
        ep = safe_export(decoder, args=(latents,))
        trt_inputs = [torch_tensorrt.Input(shape=latents.shape, dtype=latents.dtype)]
        self._serialize_to_path(
            ep,
            path,
            arg_inputs=trt_inputs,
            log_prefix="[diffusion:vae_decoder]",
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
        # The TRT module was swapped into pipe.unet/transformer; call it
        # the same way the captured forward was made.
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
