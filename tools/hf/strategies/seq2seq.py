"""
Seq2seq strategy: T5, BART, mT5, Pegasus, ...

Compiles both the encoder and the decoder with TRT:

  Encoder  — torch.export + dynamo.compile (export mode) or torch.compile.
             Static shape; the main compute-heavy component.

  Decoder  — torch.compile with dynamic=True.
             Handles cross-attention over encoder hidden states and
             the autoregressive self-attention KV cache through dynamo's
             symbolic shapes; no custom static-cache wrapper required.

The compiled modules are swapped back into model.encoder / model.decoder so
HF's .generate() uses both automatically.

Benchmarking: encoder latency AND full end-to-end .generate() throughput.
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


class SeqToSeqStrategy(ModelStrategy):
    def __init__(self, cfg: RunConfig):
        super().__init__(cfg)
        self._model = None
        self._tokenizer = None
        self._trt_encoder: Optional[torch.nn.Module] = None
        self._pt_encoder: Optional[torch.nn.Module] = None  # for accuracy comparison
        self._dummy_inputs: Optional[tuple] = None
        self._device = "cuda:0"
        self._model_dtype = model_dtype(cfg.precision, cfg.autocast)
        self._encoder_owner = None  # set in compile()

    # ---------------------------------------------------------------------- #

    def load(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        print(f"[seq2seq] Loading {self.cfg.model} ...")
        self._model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                self.cfg.model,
                torch_dtype=self._model_dtype,
                ignore_mismatched_sizes=True,
            )
            .eval()
            .to(self._device)
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model)

        # Build encoder dummy inputs.  T5 encoder takes (input_ids, attention_mask).
        seq_len = self.cfg.isl
        vocab = self._model.config.vocab_size
        input_ids = torch.randint(
            0, vocab, (self.cfg.batch_size, seq_len), dtype=torch.int64
        ).to(self._device)
        attention_mask = torch.ones_like(input_ids)
        self._dummy_inputs = (input_ids, attention_mask)
        print(f"[seq2seq] Model loaded.  Encoder input shape: {input_ids.shape}")

    # ---------------------------------------------------------------------- #

    def compile(self) -> None:
        assert self._model is not None, "Call load() before compile()"
        # T5: model.encoder.  BART/Pegasus/mBART: model.model.encoder.
        if (
            hasattr(self._model, "encoder")
            and not isinstance(getattr(self._model, "encoder", None), type(None))
            and hasattr(self._model.encoder, "forward")
        ):
            encoder_owner = self._model
        elif hasattr(self._model, "model") and hasattr(self._model.model, "encoder"):
            encoder_owner = self._model.model
        else:
            raise AttributeError(
                f"Could not find encoder submodule on {type(self._model).__name__}"
            )
        encoder = encoder_owner.encoder
        self._encoder_owner = encoder_owner  # remember for swap-back
        self._pt_encoder = encoder  # keep PT reference for accuracy

        if self.cfg.mode == "compile":
            self._trt_encoder = self._compile_torch_compile(encoder)
        else:
            self._trt_encoder = self._compile_export(encoder)

        # Preserve attributes the parent model accesses on the encoder.
        for attr_name in ("config", "dtype", "main_input_name"):
            if hasattr(encoder, attr_name) and not hasattr(
                self._trt_encoder, attr_name
            ):
                try:
                    setattr(self._trt_encoder, attr_name, getattr(encoder, attr_name))
                except Exception:
                    pass

        self._encoder_owner.encoder = self._trt_encoder
        print(
            f"[seq2seq] Compiled encoder swapped into {type(self._encoder_owner).__name__}.encoder"
        )

        # ---- Decoder ----
        # torch.compile with dynamic=True handles the autoregressive KV cache
        # and cross-attention over encoder hidden states through dynamo's
        # symbolic shapes.  torch.export of the decoder is not used because
        # the dynamic past_key_values structure cannot be exported cleanly.
        decoder = getattr(self._encoder_owner, "decoder", None)
        if decoder is not None:
            self._compile_decoder(decoder)
        else:
            print("[seq2seq] No decoder found on encoder_owner — skipping decoder TRT.")

    def _compile_torch_compile(self, encoder: torch.nn.Module) -> torch.nn.Module:
        print("[seq2seq] torch.compile encoder (backend='torch_tensorrt') ...")
        options = compile_kwargs(self.cfg.precision, self.cfg.autocast)
        options.update(
            {"min_block_size": self.cfg.min_block_size, "debug": self.cfg.debug}
        )

        compiled = torch.compile(encoder, backend="torch_tensorrt", options=options)
        with torch.no_grad():
            _ = compiled(*self._dummy_inputs)
        torch.cuda.synchronize()
        return compiled

    def _compile_export(self, encoder: torch.nn.Module) -> torch.nn.Module:
        print("[seq2seq] torch.export.export encoder ...")
        # Dynamic batch + seq_len.
        ep = safe_export(
            encoder,
            args=self._dummy_inputs,
            dynamic_shapes=(
                {0: torch.export.Dim.AUTO, 1: torch.export.Dim.AUTO},
                {0: torch.export.Dim.AUTO, 1: torch.export.Dim.AUTO},
            ),
        )
        maybe_save_exported_program(self.cfg, ep, log_prefix="[seq2seq]")

        print("[seq2seq] torch_tensorrt.dynamo.compile encoder ...")
        trt_inputs = [
            torch_tensorrt.Input(shape=t.shape, dtype=t.dtype)
            for t in self._dummy_inputs
        ]
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
            log_prefix="[seq2seq]",
        )
        maybe_save_trt_engine(self.cfg, ep, trt_inputs, log_prefix="[seq2seq]")

        return compiled

    def _compile_decoder(self, decoder: torch.nn.Module) -> None:
        print(
            "[seq2seq] torch.compile decoder (dynamic=True, backend='torch_tensorrt') ..."
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
        self._encoder_owner.decoder = compiled
        print("[seq2seq] Compiled decoder swapped into encoder_owner.decoder")

    # ---------------------------------------------------------------------- #

    def benchmark(self) -> list[dict]:
        assert (
            self._dummy_inputs is not None and self._tokenizer is not None
        ), "Call load() first"
        rows: list[dict] = []
        encoder = (
            self._trt_encoder
            if self._trt_encoder is not None
            else self._encoder_owner.encoder
        )
        backend = (
            f"torch_tensorrt[{self.cfg.mode}]"
            if self._trt_encoder is not None
            else "pytorch"
        )

        def _run_encoder():
            with torch.no_grad():
                return encoder(*self._dummy_inputs)

        timings = warmup_and_time(_run_encoder, (), iterations=self.cfg.iterations)
        rows.append(
            report_latency(
                compute_stats(timings, self.cfg.batch_size),
                backend=f"{backend} (encoder)",
                batch_size=self.cfg.batch_size,
                precision=self.cfg.precision,
            )
        )

        # End-to-end generate: encoder + decoder together.
        inputs = self._tokenizer(self.cfg.prompt, return_tensors="pt", padding=True).to(
            self._device
        )
        num_tokens = min(self.cfg.num_tokens, 128)

        def _run_generate():
            with torch.no_grad():
                return self._model.generate(**inputs, max_new_tokens=num_tokens)

        gen_timings = warmup_and_time(_run_generate, (), iterations=self.cfg.iterations)
        gen_stats = compute_stats(gen_timings, self.cfg.batch_size)
        rows.append(
            report_latency(
                gen_stats,
                backend=f"{backend} (generate)",
                batch_size=self.cfg.batch_size,
                precision=self.cfg.precision,
            )
        )

        print_table(rows, title=f"Seq2seq benchmark – {self.cfg.model}")
        return rows

    # ---------------------------------------------------------------------- #

    def _run_pt(self):
        with torch.no_grad():
            return self._pt_encoder(*self._dummy_inputs)

    def _run_trt(self):
        with torch.no_grad():
            return self._trt_encoder(*self._dummy_inputs)

    def generate(self) -> None:
        """End-to-end translate/summarize: TRT encoder + PyTorch decoder."""
        assert self._tokenizer is not None and self._model is not None
        prompt = self.cfg.prompt
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        print(f"[seq2seq] Prompt: {prompt!r}")
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=min(self.cfg.num_tokens, 128),
            )
        text = self._tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"[seq2seq] Output: {text!r}")
