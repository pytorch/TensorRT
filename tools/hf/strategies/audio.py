"""
Audio strategy: Whisper (and similar encoder-decoder ASR models).

We compile only the **encoder**.  The decoder uses cross-attention over
encoder outputs and an autoregressive KV cache — those don't trace cleanly
with torch.export today.  The decoder stays in PyTorch.

For Whisper the encoder is the bulk of the compute on long audio, so this
still delivers most of the available speedup.

Benchmarking metric: real-time factor (RTF = inference_time / audio_duration).
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
from common.metrics import print_table, report_latency, report_rtf
from strategies.base import ModelStrategy, RunConfig

# Whisper standard mel-spectrogram parameters
_WHISPER_MEL_BINS = 80
_WHISPER_TIME_FRAMES = 3000  # 30 s at 100 frames/s


class AudioStrategy(ModelStrategy):
    def __init__(self, cfg: RunConfig):
        super().__init__(cfg)
        self._model = None
        self._processor = None
        self._trt_encoder: Optional[torch.nn.Module] = None
        self._pt_encoder: Optional[torch.nn.Module] = None  # for accuracy
        self._device = "cuda:0"
        self._mel_shape: Optional[tuple] = None
        self._model_dtype = model_dtype(cfg.precision, cfg.autocast)

    # ---------------------------------------------------------------------- #

    def load(self) -> None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        print(f"[audio] Loading {self.cfg.model} ...")
        self._model = (
            AutoModelForSpeechSeq2Seq.from_pretrained(
                self.cfg.model,
                torch_dtype=self._model_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            .eval()
            .to(self._device)
        )
        self._processor = AutoProcessor.from_pretrained(self.cfg.model)

        mc = self._model.config
        mel_bins = getattr(mc, "num_mel_bins", _WHISPER_MEL_BINS)
        max_src_pos = getattr(mc, "max_source_positions", _WHISPER_TIME_FRAMES // 2)
        time_frames = max_src_pos * 2

        self._mel_shape = (self.cfg.batch_size, mel_bins, time_frames)
        print(f"[audio] Model loaded.  Mel shape: {self._mel_shape}")

    # ---------------------------------------------------------------------- #

    def compile(self) -> None:
        assert self._model is not None, "Call load() before compile()"
        encoder = self._model.model.encoder
        self._pt_encoder = encoder  # keep PT reference for accuracy comparison

        if self.cfg.mode == "compile":
            self._trt_encoder = self._compile_torch_compile(encoder)
        else:
            self._trt_encoder = self._compile_export(encoder)

        # Swap the compiled encoder back into the model so .generate() uses it.
        self._model.model.encoder = self._trt_encoder
        print("[audio] Compiled encoder swapped into self._model.model.encoder")

        # ---- Decoder ----
        decoder = getattr(self._model.model, "decoder", None)
        if decoder is not None:
            self._compile_decoder(decoder)
        else:
            print("[audio] No decoder found — skipping decoder TRT.")

    def _compile_torch_compile(self, encoder: torch.nn.Module) -> torch.nn.Module:
        print("[audio] torch.compile encoder (backend='torch_tensorrt') ...")
        options = compile_kwargs(self.cfg.precision, self.cfg.autocast)
        options.update(
            {"min_block_size": self.cfg.min_block_size, "debug": self.cfg.debug}
        )

        compiled = torch.compile(encoder, backend="torch_tensorrt", options=options)
        dummy = torch.randn(*self._mel_shape, dtype=self._model_dtype).to(self._device)
        with torch.no_grad():
            _ = compiled(dummy)
        torch.cuda.synchronize()
        return compiled

    def _compile_export(self, encoder: torch.nn.Module) -> torch.nn.Module:
        dummy = torch.randn(*self._mel_shape, dtype=self._model_dtype).to(self._device)

        print("[audio] torch.export.export encoder ...")
        ep = safe_export(
            encoder, args=(dummy,), dynamic_shapes=({0: torch.export.Dim.AUTO},)
        )
        maybe_save_exported_program(self.cfg, ep, log_prefix="[audio]")

        print("[audio] torch_tensorrt.dynamo.compile encoder ...")
        trt_inputs = [torch_tensorrt.Input(shape=dummy.shape, dtype=dummy.dtype)]
        compiled = compile_with_trt(
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

        maybe_save_trt_module(
            self.cfg,
            compiled,
            arg_inputs=trt_inputs,
            log_prefix="[audio]",
        )
        maybe_save_trt_engine(self.cfg, ep, trt_inputs, log_prefix="[audio]")

        return compiled

    def _compile_decoder(self, decoder: torch.nn.Module) -> None:
        print(
            "[audio] torch.compile decoder (dynamic=True, backend='torch_tensorrt') ..."
        )
        opts = compile_kwargs(self.cfg.precision, self.cfg.autocast)
        opts.update(
            {"min_block_size": self.cfg.min_block_size, "debug": self.cfg.debug}
        )
        if self.cfg.optimization_level is not None:
            opts["optimization_level"] = self.cfg.optimization_level
        compiled = torch.compile(
            decoder, backend="torch_tensorrt", dynamic=True, options=opts
        )
        self._model.model.decoder = compiled
        print("[audio] Compiled decoder swapped into self._model.model.decoder")

    # ---------------------------------------------------------------------- #

    def benchmark(self) -> list[dict]:
        assert (
            self._model is not None and self._mel_shape is not None
        ), "Call load() first"
        rows: list[dict] = []

        dummy = torch.randn(*self._mel_shape, dtype=self._model_dtype).to(self._device)
        audio_duration_s = self.cfg.audio_duration_s

        encoder_for_bench = (
            self._trt_encoder
            if self._trt_encoder is not None
            else self._model.model.encoder
        )
        backend = (
            f"torch_tensorrt[{self.cfg.mode}]"
            if self._trt_encoder is not None
            else "pytorch"
        )

        def _run_encoder():
            with torch.no_grad():
                return encoder_for_bench(dummy)

        timings = warmup_and_time(_run_encoder, (), iterations=self.cfg.iterations)
        stats = compute_stats(timings, self.cfg.batch_size)

        rows.append(
            report_rtf(
                audio_duration_s=audio_duration_s,
                inference_s=stats["mean_latency_ms"] / 1000.0,
                backend=f"{backend} (encoder)",
                precision=self.cfg.precision,
            )
        )
        rows.append(
            report_latency(
                stats,
                backend=f"{backend} (encoder)",
                batch_size=self.cfg.batch_size,
                precision=self.cfg.precision,
            )
        )

        # End-to-end generate: encoder + decoder together.
        import numpy as np

        sr = 16000
        audio_np = np.zeros(int(sr * audio_duration_s), dtype=np.float32)
        inputs = self._processor(audio_np, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(self._model_dtype).to(self._device)

        def _run_generate():
            with torch.no_grad():
                return self._model.generate(input_features, max_new_tokens=128)

        gen_timings = warmup_and_time(_run_generate, (), iterations=self.cfg.iterations)
        gen_stats = compute_stats(gen_timings, self.cfg.batch_size)
        rows.append(
            report_rtf(
                audio_duration_s=audio_duration_s,
                inference_s=gen_stats["mean_latency_ms"] / 1000.0,
                backend=f"{backend} (generate)",
                precision=self.cfg.precision,
            )
        )
        rows.append(
            report_latency(
                gen_stats,
                backend=f"{backend} (generate)",
                batch_size=self.cfg.batch_size,
                precision=self.cfg.precision,
            )
        )

        print_table(rows, title=f"Audio benchmark – {self.cfg.model}")
        return rows

    # ---------------------------------------------------------------------- #

    def _run_pt(self):
        dummy = torch.randn(*self._mel_shape, dtype=self._model_dtype).to(self._device)
        # Stash for _run_trt to use the same input.
        self._last_dummy = dummy
        with torch.no_grad():
            return self._pt_encoder(dummy)

    def _run_trt(self):
        # Reuse the same dummy from _run_pt so PT vs TRT compare on identical input.
        dummy = getattr(self, "_last_dummy", None)
        if dummy is None:
            dummy = torch.randn(*self._mel_shape, dtype=self._model_dtype).to(
                self._device
            )
        with torch.no_grad():
            return self._trt_encoder(dummy)

    def generate(self) -> None:
        """Transcribe a silence clip end-to-end (TRT encoder + PyTorch decoder)."""
        import numpy as np

        sr = 16000
        audio = np.zeros(int(sr * self.cfg.audio_duration_s), dtype=np.float32)
        inputs = self._processor(audio, sampling_rate=sr, return_tensors="pt")
        input_features = inputs.input_features.to(self._model_dtype).to(self._device)

        with torch.no_grad():
            out = self._model.generate(input_features, max_new_tokens=128)
        text = self._processor.batch_decode(out, skip_special_tokens=True)[0]
        print(f"[audio] Transcription: {text!r}")
