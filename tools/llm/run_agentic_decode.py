# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
.. _run_agentic_decode:

Tuning LLM decode with an Auto-Tuner session
============================================

Runs a ``torch_tensorrt.autotune`` tuning session on the batch-1 decode of
a Hugging Face causal LM (default ``Qwen/Qwen3-0.6B``): it measures the
documented ``tools/llm`` baseline, lets a planner propose one change at a time
(KV cache lowering, precision, builder options, CUDA graphs, ...), measures
every candidate in a fresh worker process against an FP32 reference, and keeps
only validated gains. The session directory holds every trial, the report and the
exported decode engine; rerunning the command resumes it.

.. code-block:: bash

    python tools/llm/run_agentic_decode.py --session /tmp/qwen3-decode --max-trials 10
    python -m torch_tensorrt.autotune status /tmp/qwen3-decode

``--planner llm`` asks the logged-in ``claude`` (or ``--provider codex``) CLI to
choose the experiments; the rule-based planner needs no LLM. ``--guidance FILE``
gives the LLM planner Markdown guidance (``guidance.md`` in the session);
``--noise-runs`` and ``--head-to-head-pairs`` set how the session separates
gains from run-to-run noise; ``--quiet-gpu-s 0`` and a looser ``--max-spread``
run it on a GPU that is never idle. ``--prompt`` and ``--prefill`` choose the
token layout and the static_v3 prefill (see ``agentic_decode.py``); their
``auto`` defaults suit both short and long contexts. Every other option is
``python -m torch_tensorrt.autotune run``'s; the budget defaults to 10 trials:

.. code-block:: bash

    python tools/llm/run_agentic_decode.py --session /tmp/qwen3-decode-8k \
        --isl 8192 --osl 64 --max-trials 8
"""

import argparse
import json
import sys
from pathlib import Path

from torch_tensorrt.autotune.__main__ import main as autotune_main


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[1],
        epilog="Every other option is `python -m torch_tensorrt.autotune run`'s "
        "(--session, --planner, --guidance, --noise-runs, ...).",
        allow_abbrev=False,
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--isl", type=int, default=128, help="prompt tokens")
    parser.add_argument("--osl", type=int, default=64, help="decode tokens per call")
    parser.add_argument(
        "--prompt",
        choices=("auto", "text", "recall"),
        default="auto",
        help="token layout; 'recall' makes decode depend on the whole context",
    )
    parser.add_argument(
        "--prefill",
        choices=("auto", "engine", "eager"),
        default="auto",
        help="static_v3 prefill: through the engine one token at a time, or eager",
    )
    parser.add_argument("--target-ms", type=float, help="stop at this ms/token")
    parser.add_argument("--min-top1", type=float, help="minimum top-1 agreement")
    parser.add_argument("--max-kl", type=float, help="maximum mean KL divergence")
    args, rest = parser.parse_known_args()

    workload_args = {"model": args.model, "isl": args.isl, "osl": args.osl}
    # Recorded only when set, so earlier sessions resume with the same workload.
    if args.prompt != "auto":
        workload_args["prompt"] = args.prompt
    if args.prefill != "auto":
        workload_args["prefill"] = args.prefill
    argv = [
        "run",
        "--max-trials",
        "10",
        "--workload",
        f"{Path(__file__).with_name('agentic_decode.py')}:create_case",
        "--args",
        json.dumps(workload_args),
    ]
    if args.target_ms is not None:
        argv += ["--target", str(args.target_ms)]
    if args.min_top1 is not None:
        argv += ["--accuracy", "top1_agreement", ">=", str(args.min_top1)]
    if args.max_kl is not None:
        argv += ["--accuracy", "mean_kl", "<=", str(args.max_kl)]
    return autotune_main([*argv, *rest])


if __name__ == "__main__":
    sys.exit(main())
