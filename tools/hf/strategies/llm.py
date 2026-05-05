"""
LLM strategy: decoder-only causal LMs (LLaMA, GPT-2, Qwen, Gemma, ...).

Three KV-cache / compilation modes are supported:

  (no --cache)        Prefill-only benchmark.  Exported with dynamic seq_len,
                      no KV cache in the graph.

  --cache static_v1   FX-graph lowering pass that injects KV cache tensors as
  --cache static_v2   explicit inputs/outputs (tools/llm/static_cache_v*.py).

  --cache hf_static   HuggingFace-native StaticCache via
                      TorchExportableModuleWithStaticCache (transformers ≥ 4.43).
                      The wrapper exposes a clean (input_ids, cache_position)
                      forward — no past_key_values in the exported graph.
                      Works with both --mode compile and --mode export.

                      Cache reset is implicit: the wrapper copies cache_position[0]
                      into cumulative_length at the start of every forward call, so
                      starting a new prefill with cache_position=arange(isl) always
                      positions the write pointer at 0 — no explicit reset needed.

                      export mode uses strict=True (strict=False triggers a PyTorch
                      internal bug in run_decompositions; see issue #4162).
"""

from __future__ import annotations

import sys
import timeit
from pathlib import Path
from typing import Optional

import torch
import torch_tensorrt
from common.bench import warmup_and_time
from common.compile import (
    compile_kwargs,
    compile_with_trt,
    maybe_save_exported_program,
    maybe_save_trt_engine,
    maybe_save_trt_module,
    model_dtype,
    safe_export,
)
from common.metrics import print_table, report_tokens_per_sec
from strategies.base import ModelStrategy, RunConfig

_TOOLS_LLM = Path(__file__).resolve().parent.parent.parent / "llm"
if str(_TOOLS_LLM) not in sys.path:
    sys.path.insert(0, str(_TOOLS_LLM))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _export_llm(model: torch.nn.Module, input_ids: torch.Tensor, max_seq_len: int):
    """Export an HF causal LM with dynamic seq_len (no KV cache in graph).

    Uses Dim.AUTO so the exporter infers the correct constraints from the
    model itself.  Explicit min/max values cause constraint violations across
    multiple model families:
      - LLaMA / Mistral / Qwen: RoPE scaling emits min(64*N, 256*N) guards
        that no user-specified range satisfies
      - All SDPA causal models: is_causal = True if q_len > 1 else False
        fails when min=1 is in range
    Dim.AUTO derives the tightest correct constraints automatically, avoids
    the fallback to prefer_deferred_runtime_asserts_over_guards (which silently
    produces wrong TRT graphs for LLaMA), and works for all model families.
    """
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).to(input_ids.device)
    return safe_export(
        model,
        args=(input_ids,),
        kwargs={"position_ids": position_ids},
        dynamic_shapes={
            "input_ids": {1: torch.export.Dim.AUTO},
            "position_ids": {1: torch.export.Dim.AUTO},
        },
    )


def _hf_static_decode_loop(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    num_new_tokens: int,
    device: torch.device,
) -> None:
    """
    Prefill + greedy decode using TorchExportableModuleWithStaticCache semantics.

    cache_position drives everything: the wrapper copies cache_position[0] into
    each layer's cumulative_length at the start of every forward call, so passing
    cache_position=arange(isl) for prefill implicitly resets the write pointer to 0.
    No separate reset() call is required between benchmark iterations.
    """
    isl = input_ids.shape[1]

    # Prefill — cache_position[0]=0 resets the write pointer inside the wrapper.
    # Use keyword arg: forward signature is (input_ids, inputs_embeds, cache_position).
    cache_pos = torch.arange(isl, device=device, dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids, cache_position=cache_pos)
    logits = out[0] if isinstance(out, (tuple, list)) else out

    for step in range(num_new_tokens):
        next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
        cache_pos = torch.tensor([isl + step], device=device, dtype=torch.long)
        with torch.no_grad():
            out = model(next_tok, cache_position=cache_pos)
        logits = out[0] if isinstance(out, (tuple, list)) else out


# --------------------------------------------------------------------------- #
# Strategy
# --------------------------------------------------------------------------- #


class LLMStrategy(ModelStrategy):
    def __init__(self, cfg: RunConfig):
        super().__init__(cfg)
        self._model: Optional[torch.nn.Module] = None
        self._trt_model: Optional[torch.nn.Module] = None
        self._tokenizer = None
        self._input_ids: Optional[torch.Tensor] = None
        self._device = torch.device("cuda:0")
        self._model_dtype = model_dtype(cfg.precision, cfg.autocast)
        self._hf_wrapper: Optional[torch.nn.Module] = None

    # ---------------------------------------------------------------------- #

    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[llm] Loading {self.cfg.model} ...")
        # hf_static: keep use_cache=True (default) so attention layers use the
        # StaticCache that TorchExportableModuleWithStaticCache injects.
        # static_v1/v2: use_cache=False — the FX lowering pass injects its own
        # cache tensors; HF's dynamic cache must not appear in the export.
        use_cache = self.cfg.cache == "hf_static"
        self._model = (
            AutoModelForCausalLM.from_pretrained(
                self.cfg.model,
                use_cache=use_cache,
                attn_implementation="sdpa",
                ignore_mismatched_sizes=True,
            )
            .eval()
            .to(self._model_dtype)
            .to(self._device)
        )

        # TorchExportableModuleWithStaticCache asserts both
        # generation_config.use_cache and generation_config.cache_implementation.
        # generation_config is a separate object from model.config; set both.
        if self.cfg.cache == "hf_static":
            self._model.generation_config.use_cache = True
            self._model.generation_config.cache_implementation = "static"

        try:
            from torchtrt_ext import register_sdpa

            register_sdpa.enable_sdpa_converter(self.cfg.model, self._model.config)
        except Exception:
            pass

        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._input_ids = torch.randint(
            1, 10000, (self.cfg.batch_size, self.cfg.isl), dtype=torch.int64
        ).to(self._device)
        print(f"[llm] Model loaded.  input_ids shape: {self._input_ids.shape}")

    # ---------------------------------------------------------------------- #

    def compile(self) -> None:
        assert self._model is not None, "Call load() before compile()"
        if self.cfg.cache == "hf_static":
            if self.cfg.mode == "compile":
                self._compile_torch_compile_hf_static()
            else:
                self._compile_export_hf_static()
        elif self.cfg.mode == "compile":
            self._compile_torch_compile()
        else:
            self._compile_export()

    # ---- vanilla compile / export ---------------------------------------- #

    def _compile_torch_compile(self) -> None:
        print("[llm] torch.compile(backend='torch_tensorrt') ...")
        options = compile_kwargs(self.cfg.precision, self.cfg.autocast)
        options.update(
            {"min_block_size": self.cfg.min_block_size, "debug": self.cfg.debug}
        )

        self._trt_model = torch.compile(
            self._model, backend="torch_tensorrt", options=options
        )
        with torch.no_grad():
            _ = self._trt_model(self._input_ids)
        torch.cuda.synchronize()
        print("[llm] torch.compile done.")

    def _compile_export(self) -> None:
        max_seq_len = self.cfg.isl + self.cfg.num_tokens

        if self.cfg.cache == "static_v1":
            import static_cache_v1  # noqa: F401

            print("[llm] Static KV cache v1 lowering pass registered.")
        elif self.cfg.cache == "static_v2":
            import static_cache_v2  # noqa: F401

            print("[llm] Static KV cache v2 lowering pass registered.")

        print(f"[llm] Exporting (max_seq_len={max_seq_len}) ...")
        ep = _export_llm(self._model, self._input_ids, max_seq_len)
        maybe_save_exported_program(self.cfg, ep, log_prefix="[llm]")

        position_ids = (
            torch.arange(self._input_ids.shape[1]).unsqueeze(0).to(self._device)
        )
        print("[llm] torch_tensorrt.dynamo.compile ...")
        self._trt_model = compile_with_trt(
            ep,
            inputs=[self._input_ids, position_ids],
            precision=self.cfg.precision,
            autocast=self.cfg.autocast,
            min_block_size=self.cfg.min_block_size,
            debug=self.cfg.debug,
            offload_module_to_cpu=self.cfg.offload_module_to_cpu,
            engine_cache_dir=self.cfg.engine_cache_dir,
            optimization_level=self.cfg.optimization_level,
        )
        torch.cuda.synchronize()
        print("[llm] TRT compile done.")

        arg_inputs = [
            torch_tensorrt.Input(shape=self._input_ids.shape, dtype=torch.int64)
        ]
        kwarg_inputs = {
            "position_ids": torch_tensorrt.Input(
                shape=position_ids.shape, dtype=torch.int64
            )
        }
        maybe_save_trt_module(
            self.cfg,
            self._trt_model,
            arg_inputs=arg_inputs,
            kwarg_inputs=kwarg_inputs,
            log_prefix="[llm]",
        )
        maybe_save_trt_engine(
            self.cfg, ep, arg_inputs, kwarg_inputs=kwarg_inputs, log_prefix="[llm]"
        )

    # ---- hf_static compile / export -------------------------------------- #

    def _compile_torch_compile_hf_static(self) -> None:
        """
        torch.compile path.  Wraps the model in TorchExportableModuleWithStaticCache
        to get a clean (input_ids, cache_position) → logits forward, then compiles
        the wrapper with torch_tensorrt.
        """
        from transformers.integrations.executorch import (
            TorchExportableModuleWithStaticCache,
        )

        max_seq_len = self.cfg.isl + self.cfg.num_tokens
        print(
            f"[llm:hf_static] Wrapping in TorchExportableModuleWithStaticCache (max_cache_len={max_seq_len}) ..."
        )
        wrapper = TorchExportableModuleWithStaticCache(
            self._model,
            batch_size=self.cfg.batch_size,
            max_cache_len=max_seq_len,
        )
        self._hf_wrapper = wrapper

        isl = self.cfg.isl
        max_seq_len = self.cfg.isl + self.cfg.num_tokens

        # Dynamic input specs covering both prefill (seq_len=isl) and single-token
        # decode (seq_len=1).  Passing these in options tells TRT to build one engine
        # over the full range rather than recompiling on each new shape.
        input_ids_spec = torch_tensorrt.Input(
            min_shape=(self.cfg.batch_size, 1),
            opt_shape=(self.cfg.batch_size, isl),
            max_shape=(self.cfg.batch_size, max_seq_len),
            dtype=torch.int64,
            name="input_ids",
        )
        cache_pos_spec = torch_tensorrt.Input(
            min_shape=(1,),
            opt_shape=(isl,),
            max_shape=(max_seq_len,),
            dtype=torch.int64,
            name="cache_position",
        )

        options = compile_kwargs(self.cfg.precision, self.cfg.autocast)
        options.update(
            {
                "min_block_size": self.cfg.min_block_size,
                "debug": self.cfg.debug,
                "inputs": [input_ids_spec, cache_pos_spec],
            }
        )

        print("[llm:hf_static] torch.compile(backend='torch_tensorrt') ...")
        # dynamic=True: tell dynamo to use symbolic shapes so it does not
        # re-emit guards and recompile when the decode step switches from
        # seq_len=isl to seq_len=1.  The TRT engine is still built once for
        # the full min→opt→max range via the `inputs` spec above.
        self._trt_model = torch.compile(
            wrapper, backend="torch_tensorrt", dynamic=True, options=options
        )

        # Warm up with prefill shape first, then single-token decode shape, so
        # TRT traces both code paths before the benchmark begins.
        cache_pos = torch.arange(isl, device=self._device, dtype=torch.long)
        decode_ids = self._input_ids[:, :1]
        decode_pos = torch.zeros(1, device=self._device, dtype=torch.long)
        with torch.no_grad():
            _ = self._trt_model(self._input_ids, cache_position=cache_pos)
            _ = self._trt_model(decode_ids, cache_position=decode_pos)
        torch.cuda.synchronize()
        print("[llm:hf_static] torch.compile done.")

    def _compile_export_hf_static(self) -> None:
        """
        torch.export + dynamo.compile path.

        TorchExportableModuleWithStaticCache bakes the StaticCache into the module
        so the exported graph has signature (input_ids, cache_position) → logits.
        The KV tensors and cumulative_length scalars are mutable buffers in the EP.

        strict=True is required: strict=False triggers an internal PyTorch bug in
        run_decompositions() (aot_stage2_export AssertionError). See issue #4162.
        """
        from transformers.integrations.executorch import (
            TorchExportableModuleWithStaticCache,
        )

        max_seq_len = self.cfg.isl + self.cfg.num_tokens
        isl = self.cfg.isl

        print(
            f"[llm:hf_static] Wrapping in TorchExportableModuleWithStaticCache (max_cache_len={max_seq_len}) ..."
        )
        wrapper = TorchExportableModuleWithStaticCache(
            self._model,
            batch_size=self.cfg.batch_size,
            max_cache_len=max_seq_len,
        )
        self._hf_wrapper = wrapper

        # Export with a single dynamic seq_len that covers both prefill (isl)
        # and single-token decode (1).  Both inputs share the same dim.
        seq_dim = torch.export.Dim("seq_len", min=1, max=max_seq_len)
        cache_pos = torch.arange(isl, device=self._device, dtype=torch.long)

        print(f"[llm:hf_static] torch.export.export (max_seq_len={max_seq_len}) ...")
        # strict=True is required: strict=False triggers an internal PyTorch bug
        # in run_decompositions() (aot_stage2_export AssertionError) when the EP
        # contains in-place buffer mutations from StaticCache.  See #4162.
        with torch.no_grad():
            ep = torch.export.export(
                wrapper,
                args=(),
                kwargs={"input_ids": self._input_ids, "cache_position": cache_pos},
                dynamic_shapes={
                    "input_ids": {1: seq_dim},
                    "cache_position": {0: seq_dim},
                },
                strict=True,
            )

        maybe_save_exported_program(self.cfg, ep, log_prefix="[llm:hf_static]")

        input_ids_spec = torch_tensorrt.Input(
            min_shape=(self.cfg.batch_size, 1),
            opt_shape=(self.cfg.batch_size, isl),
            max_shape=(self.cfg.batch_size, max_seq_len),
            dtype=torch.int64,
        )
        cache_pos_spec = torch_tensorrt.Input(
            min_shape=(1,),
            opt_shape=(isl,),
            max_shape=(max_seq_len,),
            dtype=torch.int64,
        )

        print("[llm:hf_static] torch_tensorrt.dynamo.compile ...")
        self._trt_model = compile_with_trt(
            ep,
            inputs=[input_ids_spec, cache_pos_spec],
            precision=self.cfg.precision,
            autocast=self.cfg.autocast,
            min_block_size=self.cfg.min_block_size,
            debug=self.cfg.debug,
            offload_module_to_cpu=self.cfg.offload_module_to_cpu,
            engine_cache_dir=self.cfg.engine_cache_dir,
            optimization_level=self.cfg.optimization_level,
        )
        torch.cuda.synchronize()
        print("[llm:hf_static] TRT compile done.")

    # ---------------------------------------------------------------------- #

    def benchmark(self) -> list[dict]:
        if self.cfg.cache == "hf_static":
            return self._benchmark_generation_hf_static()
        if self.cfg.cache:
            return self._benchmark_generation()
        return self._benchmark_prefill()

    def _benchmark_prefill(self) -> list[dict]:
        assert self._input_ids is not None
        position_ids = (
            torch.arange(self._input_ids.shape[1]).unsqueeze(0).to(self._device)
        )
        rows: list[dict] = []
        tokens_per_iter = self.cfg.batch_size * self.cfg.isl

        def _run_pt():
            with torch.no_grad():
                return self._model(self._input_ids, position_ids=position_ids)

        pt_t = warmup_and_time(_run_pt, (), iterations=self.cfg.iterations)
        rows.append(
            report_tokens_per_sec(
                total_tokens=tokens_per_iter * self.cfg.iterations,
                elapsed_s=sum(pt_t),
                backend="pytorch (prefill)",
                precision=self.cfg.precision,
            )
        )

        if self._trt_model is not None:

            def _run_trt():
                with torch.no_grad():
                    return self._trt_model(self._input_ids, position_ids)

            trt_t = warmup_and_time(_run_trt, (), iterations=self.cfg.iterations)
            rows.append(
                report_tokens_per_sec(
                    total_tokens=tokens_per_iter * self.cfg.iterations,
                    elapsed_s=sum(trt_t),
                    backend=f"torch_tensorrt[{self.cfg.mode}] (prefill)",
                    precision=self.cfg.precision,
                )
            )

        if self.cfg.inductor:
            print("[llm] torch.compile(backend='inductor', dynamic=True) ...")
            ind_model = torch.compile(self._model, backend="inductor", dynamic=True)

            def _run_ind():
                with torch.no_grad():
                    return ind_model(self._input_ids, position_ids=position_ids)

            ind_t = warmup_and_time(_run_ind, (), iterations=self.cfg.iterations)
            rows.append(
                report_tokens_per_sec(
                    total_tokens=tokens_per_iter * self.cfg.iterations,
                    elapsed_s=sum(ind_t),
                    backend="inductor (prefill)",
                    precision=self.cfg.precision,
                )
            )

        print_table(rows, title=f"LLM prefill benchmark – {self.cfg.model}")
        return rows

    def _benchmark_generation(self) -> list[dict]:
        """static_v1 / static_v2 generation benchmark."""
        assert self._input_ids is not None and self._tokenizer is not None
        from utils import generate_with_static_cache, time_generate

        max_out = self._input_ids.shape[1] + self.cfg.num_tokens
        rows: list[dict] = []

        def _pt_generate(model, input_ids, max_len, eos):
            with torch.no_grad():
                return model.generate(
                    input_ids, max_length=max_len, do_sample=False, use_cache=True
                )

        pt_timings = time_generate(
            _pt_generate,
            self._model,
            self._input_ids.clone(),
            max_out,
            self._tokenizer.eos_token_id,
            iterations=self.cfg.iterations,
        )
        pt_total_new = self.cfg.num_tokens * self.cfg.batch_size * self.cfg.iterations
        rows.append(
            report_tokens_per_sec(
                total_tokens=pt_total_new,
                elapsed_s=sum(pt_timings),
                backend="pytorch (generate)",
                precision=self.cfg.precision,
            )
        )

        if self._trt_model is not None:
            trt_timings = time_generate(
                generate_with_static_cache,
                self._trt_model,
                self._input_ids.clone(),
                max_out,
                self._tokenizer.eos_token_id,
                iterations=self.cfg.iterations,
            )
            rows.append(
                report_tokens_per_sec(
                    total_tokens=pt_total_new,
                    elapsed_s=sum(trt_timings),
                    backend=f"torch_tensorrt[{self.cfg.mode}] ({self.cfg.cache})",
                    precision=self.cfg.precision,
                )
            )

        print_table(rows, title=f"LLM generation benchmark – {self.cfg.model}")
        return rows

    def _benchmark_generation_hf_static(self) -> list[dict]:
        """
        hf_static generation benchmark.

        Both PT and TRT paths use the same _hf_static_decode_loop.  The wrapper's
        forward self-resets via cumulative_length.copy_(cache_position[0:1]),
        so no explicit cache reset is needed between benchmark iterations.
        """
        assert self._input_ids is not None and self._hf_wrapper is not None
        n_new = self.cfg.num_tokens
        rows: list[dict] = []
        total_new = n_new * self.cfg.batch_size * self.cfg.iterations

        # ---- PyTorch baseline ----
        def _pt_loop():
            _hf_static_decode_loop(
                self._hf_wrapper, self._input_ids, n_new, self._device
            )

        for _ in range(3):
            _pt_loop()
        pt_timings = []
        for _ in range(self.cfg.iterations):
            t0 = timeit.default_timer()
            _pt_loop()
            torch.cuda.synchronize()
            pt_timings.append(timeit.default_timer() - t0)

        rows.append(
            report_tokens_per_sec(
                total_tokens=total_new,
                elapsed_s=sum(pt_timings),
                backend="pytorch (hf_static)",
                precision=self.cfg.precision,
            )
        )

        # ---- TRT path ----
        if self._trt_model is not None:

            def _trt_loop():
                _hf_static_decode_loop(
                    self._trt_model, self._input_ids, n_new, self._device
                )

            for _ in range(3):
                _trt_loop()
            trt_timings = []
            for _ in range(self.cfg.iterations):
                t0 = timeit.default_timer()
                _trt_loop()
                torch.cuda.synchronize()
                trt_timings.append(timeit.default_timer() - t0)

            rows.append(
                report_tokens_per_sec(
                    total_tokens=total_new,
                    elapsed_s=sum(trt_timings),
                    backend=f"torch_tensorrt[{self.cfg.mode}] (hf_static)",
                    precision=self.cfg.precision,
                )
            )

        print_table(
            rows, title=f"LLM hf_static generation benchmark – {self.cfg.model}"
        )
        return rows

    # ---------------------------------------------------------------------- #

    def _run_pt(self):
        if self.cfg.cache == "hf_static":
            assert self._hf_wrapper is not None
            # Prefill from position 0 — the wrapper self-resets via cache_position.
            cache_pos = torch.arange(
                self.cfg.isl, device=self._device, dtype=torch.long
            )
            with torch.no_grad():
                return self._hf_wrapper(self._input_ids, cache_position=cache_pos)
        position_ids = (
            torch.arange(self._input_ids.shape[1]).unsqueeze(0).to(self._device)
        )
        with torch.no_grad():
            return self._model(self._input_ids, position_ids=position_ids)

    def _run_trt(self):
        if self.cfg.cache == "hf_static":
            cache_pos = torch.arange(
                self.cfg.isl, device=self._device, dtype=torch.long
            )
            with torch.no_grad():
                return self._trt_model(self._input_ids, cache_position=cache_pos)
        position_ids = (
            torch.arange(self._input_ids.shape[1]).unsqueeze(0).to(self._device)
        )
        with torch.no_grad():
            return self._trt_model(self._input_ids, position_ids)

    # ---------------------------------------------------------------------- #

    def generate(self) -> None:
        assert self._tokenizer is not None and self._model is not None
        inputs = self._tokenizer(self.cfg.prompt, return_tensors="pt").to(self._device)
        input_ids = inputs["input_ids"]
        print(f"[llm] Prompt: {self.cfg.prompt!r}")
        with torch.no_grad():
            out = self._model.generate(input_ids, max_new_tokens=self.cfg.num_tokens)
        text = self._tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"[llm] PyTorch output: {text!r}")
        if self.cfg.cache != "hf_static":
            print(
                "[llm] Note: TRT generation via --cache static_v1|static_v2 uses tools/llm/utils.generate_with_static_cache."
            )
