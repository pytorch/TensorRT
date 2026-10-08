# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Compare saved TP engines with a full eager reference at the selected precision.

Run after exporting engines with tensor_parallel_llama_export.py. Each GPU must
also fit the unsharded reference model. The engine's sequence/cache capacity must
cover the prompt and requested steps. For the measured Qwen configuration::

    torchtrtrun --nproc_per_node=2 check_tensor_parallel_llama_export.py \
        --model Qwen/Qwen2.5-0.5B-Instruct --save-dir /tmp/llm_tp_engines \
        --cache static_v2 --steps 8 --tolerance-profile qwen2.5-0.5b-fp16

For freshly exported FP32 engines, pass --precision fp32 and leave the tolerance
profile at its strict default (atol=1e-3, rtol=1e-4). The checker does not change
a saved engine's precision.

Without --prompt, use token IDs [4, 5, 6, 7] to reproduce the original regression.
Every implementation follows the full reference's next token. Token agreement is
reported separately; this numerical check does not measure free-running quality.
"""

import argparse
import gc
import json

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast  # noqa: F401

import torch_tensorrt
from torch_tensorrt.distributed._nccl_utils import initialize_nccl_comm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument(
        "--cache", choices=["", "static_v1", "static_v2"], default="static_v2"
    )
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--prompt", help="Optional text instead of the fixed regression token IDs"
    )
    parser.add_argument(
        "--tolerance-profile",
        choices=["strict", "qwen2.5-0.5b-fp16"],
        default="strict",
        help="Keep strict tolerances unless explicitly checking the measured Qwen configuration.",
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.precision == "fp32" and args.tolerance_profile != "strict":
        parser.error("The relaxed Qwen profile is for FP16; use strict checks for FP32")

    # This example initializes the process group and selects the rank's CUDA device.
    import tensor_parallel_llama_export as example
    from utils import get_zeroed_static_cache_inputs

    rank, world = dist.get_rank(), dist.get_world_size()
    device = example.DEVICE
    atol = rtol = 0.02
    if args.precision == "fp32":
        # FP32 Qwen TP=2 replay on B300 / TRT 11.3: 896 natural-prompt steps
        # passed atol=rtol=1e-4; the 8 synthetic-token steps needed atol up to
        # 0.000465 at rtol=1e-4. Allow 1e-3 absolute error for accumulated
        # rounding, while keeping relative error and token agreement visible.
        atol, rtol = 1e-3, 1e-4
    if args.tolerance_profile == "qwen2.5-0.5b-fp16":
        if args.cache != "static_v2" or world != 2:
            parser.error("The Qwen FP16 tolerance profile requires static_v2 and TP=2")
        # Qwen2.5-0.5B-Instruct, static_v2, TP=2, batch=1, FP16 autocast:
        # 896 steps across 10 prompts plus the original 8-step regression on
        # B300 / TRT 11.3.0.99 required atol up to 0.0699 at rtol=0.02. Use
        # 0.08 for margin. Eager TP also differed from full eager in FP16; explicit
        # FP32 controls agreed closely. Two FP16 top-score ties changed argmax,
        # so logit closeness does not imply identical token choices. Replaying
        # those contexts in FP32 made TRT and full eager agree on both tokens.
        # This opt-in budget is checkpoint-specific, not a wider default for
        # all models.
        atol = 0.08

    initialize_nccl_comm()
    program = torch_tensorrt.load(example._rank_path(args.save_dir, rank, world))
    loaded = program.module()
    reference = (
        AutoModelForCausalLM.from_pretrained(
            args.model, use_cache=False, attn_implementation="sdpa"
        )
        .to(device)
        .eval()
    )
    if args.precision == "fp32":
        reference = reference.float()
    if args.tolerance_profile == "qwen2.5-0.5b-fp16":
        config = reference.config
        # Reject other Qwen sizes and tiny fixtures even when loading a local
        # checkpoint directory whose name does not contain the HF model ID.
        if not (
            config.model_type == "qwen2"
            and config.hidden_size == 896
            and config.num_hidden_layers == 24
            and config.num_attention_heads == 14
            and config.num_key_value_heads == 2
            and config.vocab_size == 151936
        ):
            parser.error("The Qwen tolerance profile is only for Qwen2.5-0.5B-Instruct")

    token_ids = [4, 5, 6, 7]
    if args.prompt is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        token_ids = tokenizer(args.prompt)["input_ids"]
    if len(token_ids) < 2:
        parser.error("The prompt must contain at least two tokens")
    sequence = torch.tensor([token_ids], dtype=torch.int64, device=device)
    dist.broadcast(sequence, src=0)
    current = sequence
    kv = get_zeroed_static_cache_inputs(loaded, device=device) if args.cache else ()
    if args.precision == "fp32" and any(t.dtype != torch.float32 for t in kv):
        parser.error(
            "The saved cache is not FP32; re-export the engine with --precision fp32"
        )
    if args.cache:
        # The two scalar placeholders bound cache start/end indices. Fail with
        # a clear message before executing beyond the saved cache capacity.
        for node in program.graph.nodes:
            value = node.meta.get("val")
            if node.op == "placeholder" and isinstance(value, torch.SymInt):
                bounds = program.range_constraints[value.node.expr]
                if len(token_ids) + args.steps - 1 > bounds.upper:
                    parser.error(
                        "Requested steps exceed the saved cache capacity; re-export with more capacity"
                    )

    token_matches = 0
    with torch.inference_mode(), example._precision_context(args):
        for step in range(args.steps):
            end = sequence.shape[1]
            start = 0 if step == 0 else end - 1
            positions = torch.arange(end, device=device).unsqueeze(0)
            if args.cache:
                outputs = loaded(current, positions[:, start:], *kv, start, end)
                actual, kv = outputs[0][:, -1, :], outputs[1:]
            else:
                outputs = loaded(sequence, position_ids=positions)
                actual = example._extract_logits(outputs)[:, -1, :]
            expected = reference(sequence, position_ids=positions).logits[:, -1, :]
            matches = bool(torch.equal(actual.argmax(-1), expected.argmax(-1)))
            token_matches += int(matches)
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "step": step,
                        "precision": args.precision,
                        "atol": atol,
                        "rtol": rtol,
                        "max_logit_difference": (actual.float() - expected.float())
                        .abs()
                        .max()
                        .item(),
                        "top1_equal": matches,
                        "reference_top2_margin": float(
                            expected.float().topk(2).values.diff(dim=-1).abs().item()
                        ),
                    }
                ),
                flush=True,
            )
            torch.testing.assert_close(
                actual.float(), expected.float(), atol=atol, rtol=rtol
            )
            current = expected.argmax(-1, keepdim=True)
            dist.broadcast(current, src=0)
            sequence = torch.cat([sequence, current], dim=1)

    print(
        json.dumps(
            {
                "rank": rank,
                "accuracy": "passed",
                "steps": args.steps,
                "token_matches": token_matches,
            }
        ),
        flush=True,
    )
    # Release the engine before destroying the communicator it uses.
    del loaded, program, reference, outputs, actual, expected, kv
    gc.collect()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
