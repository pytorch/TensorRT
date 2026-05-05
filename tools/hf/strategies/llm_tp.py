"""
Tensor-parallel LLM strategy.

Activates automatically under torchtrtrun (when WORLD_SIZE > 1).  Mirrors
the pattern in tools/llm/tensor_parallel_llama_multinode.py:

  1. Load the HF model on each rank.
  2. Build a Megatron-style TP plan from the model's layer structure.
  3. parallelize_module() shards the model across the TP mesh.
  4. Patch attention head counts on each rank (sharded heads).
  5. Compile via torch.compile(backend="torch_tensorrt", dynamic=True)
     — torch.export does NOT trace cleanly with TP shards.
  6. Wrap inference in distributed_context() so NCCL is released
     cleanly before dist.destroy_process_group.

Currently supports Llama-family architectures (Llama, Qwen, Mistral,
Gemma — anything with `model.layers.{i}.self_attn.{q,k,v,o}_proj` and
`model.layers.{i}.mlp.{gate,up,down}_proj`).  GPT-2 / GPT-NeoX / OPT
use combined-QKV layouts and need a different plan — not yet wired.

Launch
------
  torchtrtrun --nproc_per_node=2 run_hf.py --model meta-llama/Llama-3.2-1B-Instruct
"""

from __future__ import annotations

import logging
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch_tensorrt
from common.bench import warmup_and_time
from common.compile import compile_kwargs, model_dtype
from common.dist import (
    barrier,
    build_device_mesh,
    is_master,
    local_rank,
    rank,
    world_size,
)
from common.metrics import print_table, report_tokens_per_sec
from strategies.base import RunConfig
from strategies.llm import LLMStrategy

# Make tools/llm/utils importable for generate / time_generate.
_TOOLS_LLM = Path(__file__).resolve().parent.parent.parent / "llm"
if str(_TOOLS_LLM) not in sys.path:
    sys.path.insert(0, str(_TOOLS_LLM))

logger = logging.getLogger("torchtrt.hf.llm_tp")


def _build_megatron_plan(model) -> dict:
    """
    Megatron-style column/row sharding for a Llama-family transformer.
    Returns a dict suitable for torch.distributed.tensor.parallel.parallelize_module.
    """
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
    )

    plan: dict[str, object] = {}
    n_layers = model.config.num_hidden_layers
    for i in range(n_layers):
        plan.update(
            {
                f"model.layers.{i}.self_attn.q_proj": ColwiseParallel(),
                f"model.layers.{i}.self_attn.k_proj": ColwiseParallel(),
                f"model.layers.{i}.self_attn.v_proj": ColwiseParallel(),
                f"model.layers.{i}.self_attn.o_proj": RowwiseParallel(),
                f"model.layers.{i}.mlp.gate_proj": ColwiseParallel(),
                f"model.layers.{i}.mlp.up_proj": ColwiseParallel(),
                f"model.layers.{i}.mlp.down_proj": RowwiseParallel(),
            }
        )
    return plan


def _patch_attention_head_counts(model, ws: int) -> None:
    """
    After column-sharding Q/K/V, each rank holds num_heads // world_size
    heads.  HF attention uses self.num_heads / self.num_key_value_heads to
    reshape activations — patch them so the shapes match the local shard.
    """
    n_attn = model.config.num_attention_heads
    n_kv = getattr(model.config, "num_key_value_heads", n_attn)
    if n_attn % ws != 0 or n_kv % ws != 0:
        raise ValueError(
            f"Cannot shard model: num_attention_heads ({n_attn}) and "
            f"num_key_value_heads ({n_kv}) must be divisible by world_size ({ws})."
        )
    for layer in model.model.layers:
        layer.self_attn.num_heads = n_attn // ws
        layer.self_attn.num_key_value_heads = n_kv // ws


class LLMTPStrategy(LLMStrategy):
    """
    Tensor-parallel variant of LLMStrategy.  Inherits load-time behavior
    (HF model loading, tokenizer, dummy inputs) and overrides:
      - load(): also parallelize the model + patch attn head counts.
      - compile(): forces mode='compile' (torch.export doesn't work for TP).
      - benchmark(): per-rank timing + barrier; only rank 0 prints.
    """

    def __init__(self, cfg: RunConfig):
        super().__init__(cfg)
        self._mesh = None

    # ---------------------------------------------------------------------- #

    def load(self) -> None:
        super().load()  # HF model + tokenizer + dummy inputs

        # Build a 1-D TP device mesh and shard the model.
        self._mesh = build_device_mesh(("tp",))
        ws = world_size()
        from torch.distributed.tensor.parallel import parallelize_module

        if not hasattr(self._model, "model") or not hasattr(
            self._model.model, "layers"
        ):
            raise NotImplementedError(
                f"TP plan only supports Llama-family layouts "
                f"(model.model.layers.{{i}}.self_attn / mlp).  Got: "
                f"{type(self._model).__name__}.  GPT-2 / OPT / NeoX support TBD."
            )

        plan = _build_megatron_plan(self._model)
        parallelize_module(self._model, self._mesh, plan)
        _patch_attention_head_counts(self._model, ws)

        if is_master():
            print(f"[llm_tp] Sharded across {ws} ranks (TP).")
        barrier()

    # ---------------------------------------------------------------------- #

    def compile(self) -> None:
        # torch.export + TP shards = mode-mismatch errors and broken DTensor
        # specs.  Force the torch.compile path; it traces lazily on first
        # forward pass and works correctly with DTensors.
        if self.cfg.mode != "compile":
            if is_master():
                print(
                    "[llm_tp] Forcing --mode compile for tensor-parallel run "
                    "(torch.export does not currently work with TP)."
                )
            object.__setattr__(self.cfg, "mode", "compile")
        self._compile_torch_compile_tp()

    def _compile_torch_compile_tp(self) -> None:
        """
        torch.compile with TRT backend, dynamic seq_len, and engine build
        triggered explicitly so we can barrier across ranks.
        """
        if is_master():
            print("[llm_tp] torch.compile(backend='torch_tensorrt', dynamic=True) ...")

        opts = compile_kwargs(self.cfg.precision, self.cfg.autocast)
        opts.update(
            {
                "min_block_size": self.cfg.min_block_size,
                "debug": self.cfg.debug,
                "device": self._device,
                "disable_tf32": True,
                "use_python_runtime": False,
                "assume_dynamic_shape_support": True,
            }
        )

        with torch_tensorrt.logging.debug() if self.cfg.debug else nullcontext():
            self._trt_model = torch.compile(
                self._model,
                backend="torch_tensorrt",
                dynamic=True,
                options=opts,
            )

        # Trigger the engine build with one forward + barrier so all ranks
        # finish compilation before benchmark/generation starts.
        warmup = self._input_ids.clone()
        torch._dynamo.mark_dynamic(warmup, 1)
        position_ids = torch.arange(warmup.shape[1]).unsqueeze(0).to(self._device)
        torch._dynamo.mark_dynamic(position_ids, 1)
        with torch.no_grad():
            _ = self._trt_model(warmup, position_ids=position_ids)
        torch.cuda.synchronize()
        barrier()
        if is_master():
            print(f"[llm_tp] All {world_size()} ranks finished engine build.")

    # ---------------------------------------------------------------------- #

    def benchmark(self) -> list[dict]:
        """
        Distributed benchmark: each rank runs the same workload, we time
        rank 0 and barrier between iterations.  generate() is the proper
        autoregressive loop (TP + KV cache live in the framework's compiled
        path — no static_v1/v2 lowering).
        """
        from utils import generate, time_generate  # tools/llm/utils.py

        max_out = self._input_ids.shape[1] + self.cfg.num_tokens
        rows: list[dict] = []
        tot_new = self.cfg.num_tokens * self.cfg.batch_size * self.cfg.iterations

        # PyTorch-TP baseline (uncompiled, runs DTensor ops directly).
        if is_master():
            print("[llm_tp] PyTorch-TP baseline ...")
        pt_t = time_generate(
            generate,
            self._model,
            self._input_ids.clone(),
            max_out,
            self._tokenizer.eos_token_id,
            iterations=self.cfg.iterations,
        )
        barrier()
        if is_master():
            rows.append(
                report_tokens_per_sec(
                    total_tokens=tot_new,
                    elapsed_s=sum(pt_t),
                    backend="pytorch-tp (generate)",
                    precision=self.cfg.precision,
                )
            )

        # TRT-TP (compiled).  Use distributed_context so NCCL is released
        # cleanly before destroy_process_group during teardown.
        import torch.distributed as dist
        from torch_tensorrt.distributed import distributed_context

        if is_master():
            print("[llm_tp] TRT-TP timing ...")
        with distributed_context(dist.group.WORLD, self._trt_model) as trt_model:
            trt_t = time_generate(
                lambda m, ids, mlen, eos: generate(
                    m, ids, mlen, eos, dynamic_seqlen_range=(1, max_out)
                ),
                trt_model,
                self._input_ids.clone(),
                max_out,
                self._tokenizer.eos_token_id,
                iterations=self.cfg.iterations,
            )
        barrier()

        if is_master():
            rows.append(
                report_tokens_per_sec(
                    total_tokens=tot_new,
                    elapsed_s=sum(trt_t),
                    backend=f"torch_tensorrt-tp (generate, ws={world_size()})",
                    precision=self.cfg.precision,
                )
            )
            print_table(rows, title=f"LLM TP benchmark – {self.cfg.model}")

        return rows if is_master() else []

    # ---------------------------------------------------------------------- #

    def generate(self) -> None:
        """One greedy decode + print on rank 0."""
        from utils import generate

        ids = self._tokenizer(self.cfg.prompt, return_tensors="pt")["input_ids"].to(
            self._device
        )
        max_out = ids.shape[1] + self.cfg.num_tokens
        out = generate(
            self._trt_model, ids.clone(), max_out, self._tokenizer.eos_token_id
        )
        if is_master():
            text = self._tokenizer.decode(out[0], skip_special_tokens=True)
            print(f"[llm_tp] Output: {text!r}")
