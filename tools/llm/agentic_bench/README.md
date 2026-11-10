# Reproducing the Auto-Tuner benchmarks

This directory holds what the benchmark of Auto-Tuner sessions (`torch_tensorrt.autotune`, which
the published sessions ran under its earlier name, `torch_tensorrt.agentic`) on eight batch-1
workloads ran:

- `briefs/<bench>.md`: the guidance the published LLM-planner session read (`--guidance`;
  `--brief` before the rename), byte for byte. `briefs/SHA256SUMS` lists the SHA-256 that each session recorded as `brief_sha256`
  in its `session.json`; `sha256sum -c SHA256SUMS`, run in `briefs/`, checks the files against
  them. [What the briefs hold](#what-the-briefs-hold) describes their contents, measurements
  included.
- `briefs/<bench>-hybrid.md`: the same guidance followed by a section on the hybrid knobs
  (`kernels.inductor`, and `runtime.step_graph` for decode), written for the hybrid follow-up
  sessions (section 5). The published LLM-planner sessions did not see that section.
- `specs/<bench>.json`: the head-to-head of the session's config against the TensorRT baseline
  and PyTorch Inductor, for `tools/llm/bench_compare.py`.
- `specs/<bench>-hybrid.json`: the head-to-head of the hybrid variants (`kernels.inductor`:
  Inductor's kernels inside the TensorRT config) built on the session's config.
- `specs/<bench>-kernels.json` (six benchmarks, and `qwen3-128` and `qwen3-8k` for Qwen3
  decode): where one region's kernels come from, TensorRT, Inductor or PR 4914's
  `KernelGenerator` (section 6); `specs/qwen3-128-session-kernels.json` and
  `specs/qwen3-8k-session-kernels.json` run the Qwen3 ones with the kernels an Auto-Tuner session
  prepared.

Every spec is a template: the workload is relative to the repository root and the reference
outputs are relative to the output directory you give `bench_compare.py`. A spec's
`head_to_head` lists the configurations its head-to-head runs (what `bench_compare.py all` runs
by default); the rest of its `configs` is the menu they were chosen from. Each spec's
`trt-session` is the config the published session chose (for ModernBERT, the second
sessions, which read the guidance here); `--session DIR` replaces it with the config your session
chose (`winner.config` in its `report.json`), and the hybrid configurations, which extend
`trt-session`, follow it.

## The benchmarks

The unit is what one measured latency covers. Every candidate is checked against outputs of
the FP32 eager Hugging Face model (no TF32) on its first call and again after timing; the limits
are the workloads' defaults (`Case.accuracy_defaults`), which the sessions used.

| Benchmark | Workload (`--workload`, `--args`) | Unit | Accuracy check |
|---|---|---|---|
| `qwen3-decode-8k` | `tools/llm/agentic_decode.py:create_case`, `{"model": "Qwen/Qwen3-0.6B", "isl": 8192, "osl": 64, "prompt": "recall", "prefill": "eager"}` | one decoded token: an untimed 8192-token prefill, then 64 teacher-forced tokens per call | logits at every decode position: top-1 agreement >= 0.97, mean KL <= 0.005 |
| `gemma3-prefill-2k` | `tools/llm/agentic_prefill.py:create_case`, `{"model": "unsloth/gemma-3-1b-it", "length": 2048}` | one 2048-token prefill (logits at 64 positions) | top-1 >= 0.90, mean KL <= 0.005, largest KL at one position <= 0.1 |
| `gemma3-prefill-8k` | the same with `"length": 8192` | one 8192-token prefill | the same |
| `modernbert-512` | `tools/llm/agentic_encoder.py:create_case`, `{"model": "answerdotai/ModernBERT-base", "length": 512}` | one encoder forward, `last_hidden_state` | every token's hidden state: tight limits on the bulk of the tokens and on 64-token blocks, loose ones on a few ill-conditioned tokens (`EncoderCase.accuracy_defaults`) |
| `modernbert-8k` | the same with `"length": 8192` | one encoder forward | the same, calibrated for 8192 tokens |
| `qwen35-prefill-2k` | `tools/llm/qwen35_prefill.py:create_case`, `{"model": "Qwen/Qwen3.5-0.8B", "length": 2048, "positions": 64}` | one 2048-token prefill (logits at 64 positions) | top-1 >= 0.95, mean KL <= 0.002, largest KL <= 0.01 |
| `nemotron-decode` | `tools/llm/nemotron_decode.py:create_case`, `{"model": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "isl": 128, "osl": 64}` | one decoded token: an untimed 128-token prefill, then 64 teacher-forced tokens per call | top-1 >= 0.97, mean KL <= 0.002 at every decode position |
| `nemotron-prefill-2k` | `tools/llm/nemotron_decode.py:create_prefill_case`, `{"model": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "length": 2048, "positions": 64}` | one 2048-token prefill (logits at 64 positions) | top-1 >= 0.85, mean KL <= 0.002 |

The prompts are fixed public-domain or generated text (see each workload's docstring and the
brief). `unsloth/gemma-3-1b-it` is an ungated copy of `google/gemma-3-1b-it` with the same
weights.

| Benchmark | Brief | Head-to-head | Hybrid brief | Hybrid head-to-head |
|---|---|---|---|---|
| `qwen3-decode-8k` | [briefs/qwen3-decode-8k.md](briefs/qwen3-decode-8k.md) | [specs/qwen3-decode-8k.json](specs/qwen3-decode-8k.json) | [briefs/qwen3-decode-8k-hybrid.md](briefs/qwen3-decode-8k-hybrid.md) | [specs/qwen3-decode-8k-hybrid.json](specs/qwen3-decode-8k-hybrid.json) |
| `gemma3-prefill-2k` | [briefs/gemma3-prefill-2k.md](briefs/gemma3-prefill-2k.md) | [specs/gemma3-prefill-2k.json](specs/gemma3-prefill-2k.json) | [briefs/gemma3-prefill-2k-hybrid.md](briefs/gemma3-prefill-2k-hybrid.md) | [specs/gemma3-prefill-2k-hybrid.json](specs/gemma3-prefill-2k-hybrid.json) |
| `gemma3-prefill-8k` | [briefs/gemma3-prefill-8k.md](briefs/gemma3-prefill-8k.md) | [specs/gemma3-prefill-8k.json](specs/gemma3-prefill-8k.json) | [briefs/gemma3-prefill-8k-hybrid.md](briefs/gemma3-prefill-8k-hybrid.md) | [specs/gemma3-prefill-8k-hybrid.json](specs/gemma3-prefill-8k-hybrid.json) |
| `modernbert-512` | [briefs/modernbert-512.md](briefs/modernbert-512.md) | [specs/modernbert-512.json](specs/modernbert-512.json) | [briefs/modernbert-512-hybrid.md](briefs/modernbert-512-hybrid.md) | [specs/modernbert-512-hybrid.json](specs/modernbert-512-hybrid.json) |
| `modernbert-8k` | [briefs/modernbert-8k.md](briefs/modernbert-8k.md) | [specs/modernbert-8k.json](specs/modernbert-8k.json) | [briefs/modernbert-8k-hybrid.md](briefs/modernbert-8k-hybrid.md) | [specs/modernbert-8k-hybrid.json](specs/modernbert-8k-hybrid.json) |
| `qwen35-prefill-2k` | [briefs/qwen35-prefill-2k.md](briefs/qwen35-prefill-2k.md) | [specs/qwen35-prefill-2k.json](specs/qwen35-prefill-2k.json) | [briefs/qwen35-prefill-2k-hybrid.md](briefs/qwen35-prefill-2k-hybrid.md) | [specs/qwen35-prefill-2k-hybrid.json](specs/qwen35-prefill-2k-hybrid.json) |
| `nemotron-decode` | [briefs/nemotron-decode.md](briefs/nemotron-decode.md) | [specs/nemotron-decode.json](specs/nemotron-decode.json) | [briefs/nemotron-decode-hybrid.md](briefs/nemotron-decode-hybrid.md) | [specs/nemotron-decode-hybrid.json](specs/nemotron-decode-hybrid.json) |
| `nemotron-prefill-2k` | [briefs/nemotron-prefill-2k.md](briefs/nemotron-prefill-2k.md) | [specs/nemotron-prefill-2k.json](specs/nemotron-prefill-2k.json) | [briefs/nemotron-prefill-2k-hybrid.md](briefs/nemotron-prefill-2k-hybrid.md) | [specs/nemotron-prefill-2k-hybrid.json](specs/nemotron-prefill-2k-hybrid.json) |

## What the briefs hold

A brief describes the workload, the accuracy check and its calibration, what bounds the latency,
and each knob: what it changes, what it costs, and which values were left out of the workload
and why. Its numbers are smoke measurements taken while it was written, mostly single runs on
a GPU shared with other jobs; none is a result of the sessions or of the head-to-heads. Every
brief has the baseline's profile and measured GPU rates (bandwidth, GEMM throughput), and all
but the two Qwen briefs give PyTorch eager and Inductor latencies as reference points. Several
also report what single knob settings measured, so a planner can take those as given instead
of measuring them:

| Brief | Measured besides the baseline and the PyTorch reference points |
|---|---|
| `qwen3-decode-8k` | a plain `workload.cache=static_v3` engine: GPU time per token and its kernels; Inductor's flex-decoding attention and SDPA's memory-efficient kernel, per layer; build routes and optimization levels that build the same static_v3 engine as the defaults, and values that fail to build; build times per tool value |
| `gemma3-prefill-2k`, `gemma3-prefill-8k` | per-layer attention of TensorRT engines with position-range masks (BF16 and FP16) against PyTorch's flash attention, and one-layer masked-attention micro-benchmarks; on one FP16 recipe, the optimization and tiling levels, CUDA graphs and pre-allocated outputs within 0.5% of the default build, and other build routes within 1% (left out); at 8k, the MLP in sequence chunks, 2% and 7% slower (left out); among the reference points, Inductor with the workload's `torch-flash` lowering |
| `modernbert-512`, `modernbert-8k` | among the reference points, Inductor with `workload.local_attention=banded128`; build routes that build the same engine as no route or as another route, with or without `engine.decompose_attention`; at 8k, how many GEMM launches each `-cask:fp16_acc=off` route moves to FP32 accumulation |
| `qwen35-prefill-2k` | `workload.gdn=functional` with every other knob at its default: latency, kernels and build time; the fused kernels around `kernels.gdn_scan` regions in loop and scan builds; the Triton scan kernel alone per layer (IEEE and TF32 dots) against the unrolled loop; optimization level 5 and tiling `fast` on a functional recipe (build time, accuracy) |
| `nemotron-decode` | build routes and optimization levels that build the baseline's kernels unchanged, and values that fail to build (left out); the build time with `engine.use_fp32_acc`; the patch gate's calibration on two rewrites |
| `nemotron-prefill-2k` | among the reference points, Inductor with the workload's own `workload.ssd=grouped` and `workload.chunk_size=128`, which TensorRT configs use too; `runtime.cudagraphs` and optimization level 5 as no-ops on the baseline (left out); every knob value builds one engine and passes the limits |

The published ModernBERT sessions are the second ones: the first sessions read briefs that
listed measured results of candidate configs, and that guidance was cleaned before the second
sessions ran. The planner ablation (section 5) runs sessions without guidance, to separate what
the LLM planner finds on its own from what its guidance tells it.

## Setup

- The published runs used one GeForce RTX 5090 (32 GB), driver 610.57, CUDA 13.2, TensorRT
  11.4.0.106, PyTorch 2.16.0.dev20261015+cu132, transformers 5.18.0 and Python 3.11, with
  Torch-TensorRT built from the checkout.
- The eight models in the Hugging Face cache, downloaded beforehand (for example
  `hf download Qwen/Qwen3-0.6B`): every command below runs offline (`HF_HUB_OFFLINE=1`).
- For the LLM planner, the `claude` CLI, logged in (`--provider claude`; `--provider codex`
  works too).
- One GPU lock for every GPU job of the machine, so that no two of them measure at the same
  time. `bench_compare.py` takes `$BENCH_GPU_LOCK` (default `/tmp/torch_tensorrt_gpu.lock`)
  around each run; the sessions below run under the same lock with `flock`.

Run every command from the repository root, in bash, after:

```bash
export BENCH_GPU_LOCK=/tmp/torch_tensorrt_gpu.lock   # bench_compare.py's default
export HF_HUB_OFFLINE=1
OUT=/path/to/results
KIT=tools/llm/agentic_bench
field() {  # SPEC KEY: the spec's workload or its arguments, as run_agentic.py takes them
  python -c 'import json, sys; v = json.load(open(sys.argv[1]))[sys.argv[2]]; print(v if isinstance(v, str) else json.dumps(v))' "$1" "$2"
}
agentic() {  # BENCH DIRECTORY FLAGS...: a run_agentic.py session of BENCH's workload
  local spec=$KIT/specs/$1.json dir=$2 work=$OUT/work/$(basename "$2")
  shift 2
  mkdir -p "$work/tmp"
  python tools/llm/bench_compare.py quiet --quiet-min-free-mb 12000 --quiet-timeout-s 1800
  TMPDIR=$work/tmp TORCHINDUCTOR_CACHE_DIR=$work/inductor TRITON_CACHE_DIR=$work/triton \
    flock "$BENCH_GPU_LOCK" python -P tools/llm/run_agentic.py \
      --workload "$(field "$spec" workload)" --args "$(field "$spec" args)" \
      --session "$dir" "$@"
}
MEASURE="--noise-runs 2 --head-to-head-pairs 2 --final-check-pairs 2"
```

Each session waits for an idle GPU (at most 3% busy with 12 GB free for 10 s), holds the lock
while it runs, and gets compiler caches and a `TMPDIR` of its own.

## 1. The LLM-planner sessions

One session per benchmark, with the LLM planner, the benchmark's guidance, two noise runs, two
head-to-head pairs, two final-check pairs and the benchmark's budget:

```bash
session() {  # BENCH BUDGET...: the LLM-planner session of BENCH
  agentic "$1" "$OUT/sessions/$1" --planner llm --provider claude \
    --guidance "$KIT/briefs/$1.md" $MEASURE "${@:2}"
}
session modernbert-512 --max-trials 8 --patience 4 --max-time-s 3600 --trial-timeout-s 1200
session modernbert-8k --max-trials 8 --patience 4 --max-time-s 3600 --trial-timeout-s 1800 --max-spread 0.04
session gemma3-prefill-2k --max-trials 8 --patience 4 --max-time-s 4500
session gemma3-prefill-8k --max-trials 8 --patience 4 --max-time-s 4500
session qwen3-decode-8k --max-trials 8 --patience 4 --max-time-s 4500 --trial-timeout-s 1800
session nemotron-decode --max-trials 10 --patience 4 --max-time-s 4500 --trial-timeout-s 1800 --quiet-gpu-s 120 --min-improvement 0.01
session nemotron-prefill-2k --max-trials 8 --patience 4 --max-time-s 4500 --trial-timeout-s 1800 --quiet-gpu-s 120
session qwen35-prefill-2k --max-trials 10 --patience 4 --max-time-s 7200 --trial-timeout-s 2400
```

Rerunning a command resumes its session. A session directory holds `report.md`, `report.json`
(the winner's config under `winner.config`), `trials.tsv`, every trial's results, the guidance it
read (`guidance.md`), the FP32 reference outputs (`reference.pt`) and the exported artifact (see
the `torch_tensorrt.autotune` documentation). The published sessions, run before the rename,
name these `selected.recipe`, `ledger.tsv` and `program.md` (whose SHA-256 `session.json`
records as `brief_sha256`).

The published sessions ran on earlier commits, with fewer knobs: `kernels.inductor` (every
workload) and `runtime.step_graph` (decode) did not exist yet, and their guidance does not
mention them. At this commit the workloads offer them, so a new session can choose them, without
guidance from `briefs/<bench>.md`. The knobs a published session had are in its
`session.json` (`case.tools`); this kit has no option to restrict a workload's knobs to them.

## 2. The reference outputs

Every head-to-head run is checked against the same FP32 reference outputs. A session's
`reference.pt` holds them for its workload and arguments, so the commands below pass it with
`--reference`. Without it, `bench_compare.py all` computes `reference.pt` in its output
directory first (under the lock); to compute it beforehand:

```bash
python tools/llm/bench_compare.py reference $KIT/specs/qwen3-decode-8k.json $OUT/h2h/qwen3-decode-8k
```

## 3. The head-to-head

Four rounds of fresh processes, each run after an idle-GPU wait and under the lock (see
`bench_compare.py`), of the configurations in the spec's `head_to_head`:

```bash
compare() {  # BENCH: the session's config against the baseline and Inductor
  python tools/llm/bench_compare.py all "$KIT/specs/$1.json" "$OUT/h2h/$1" \
    --session "$OUT/sessions/$1" --reference "$OUT/sessions/$1/reference.pt" \
    --rounds 4 --quiet-min-free-mb 12000
}
for bench in modernbert-512 modernbert-8k gemma3-prefill-2k gemma3-prefill-8k \
             qwen3-decode-8k nemotron-decode nemotron-prefill-2k qwen35-prefill-2k; do
  compare $bench
done
```

Without a session of your own, leave out `--session` and `--reference`: `trt-session` is then
the published session's config, and the driver computes the reference.

| Benchmark | `head_to_head` of `specs/<bench>.json` |
|---|---|
| `qwen3-decode-8k` | `trt-baseline`, `trt-session`, `inductor-max-autotune-flex`, `inductor-max-autotune-flex-cdt` |
| `gemma3-prefill-2k` | `trt-baseline`, `trt-session`, `inductor-fp16-fp16acc-max-autotune-torchflash-cdt`, `inductor-max-autotune-torchflash` |
| `gemma3-prefill-8k` | `trt-baseline`, `trt-session`, `inductor-fp16-fp16acc-max-autotune-torchflash`, `inductor-max-autotune-torchflash` |
| `modernbert-512` | `trt-baseline`, `trt-session`, `inductor-max-autotune-flex-fp16acc`, `inductor-max-autotune-flex-cdt` |
| `modernbert-8k` | `trt-baseline`, `trt-session`, `inductor-max-autotune-flex-cdt-fp16acc`, `inductor-max-autotune-flex` |
| `qwen35-prefill-2k` | `trt-baseline`, `trt-session`, `inductor-functional-causal-tf32-fp16-fp16acc-default`, `inductor-functional-causal-max-autotune` |
| `nemotron-decode` | `trt-baseline`, `trt-session`, `inductor-max-autotune-flex`, `inductor-max-autotune` |
| `nemotron-prefill-2k` | `trt-baseline`, `trt-session`, `inductor-max-autotune-fp16-fp16acc-tf32-ssd-grouped-chunk128`, `inductor-max-autotune-fp16-fp16acc-ssd-grouped-chunk128`, `inductor-max-autotune-ssd-grouped` |

These are the configurations of the published head-to-heads, except that the
`nemotron-prefill-2k` list adds `inductor-max-autotune-fp16-fp16acc-tf32-ssd-grouped-chunk128`.
Because the session's config enables TF32, that configuration was measured afterwards, in a
separate four-round head-to-head with `trt-session` and the same Inductor configuration
without TF32.

`trt-baseline` is the documented Torch-TensorRT compile, the sessions' starting point. The
Inductor configurations compile the same Hugging Face model, with the same inputs, precision
and outputs, through `torch.compile` (`bench.backend=inductor`). Their names give their
settings: the `torch.compile` mode (`default`, `max-autotune`); `flex`, FlexAttention
(`bench.attn=flex_attention`); `cdt`, coordinate-descent tuning; `fp16`, FP16 weights;
`fp16acc`, FP16 GEMMs that accumulate in FP16 (`bench.fp16_accumulation`); `tf32`, TF32 for FP32
matrix products; `torchflash`, the workload's flash-attention lowering; `functional-causal`
(Qwen3.5), the functional Gated DeltaNet with causal attention; `ssd-grouped`, `chunk128`
(Nemotron), the grouped SSD scan and its chunk length. The spec has each one's exact recipe and
`bench` keys.

Compare at matched precision. TensorRT's FP16 engines accumulate GEMMs in FP16 by default, so
each head-to-head includes the fastest Inductor configuration with the same allowances
(`bench.fp16_accumulation`, and TF32 where the session's config enables it) next to one at
PyTorch's default precision.

The rounds run the configurations in the rows of a Williams Latin square, which balances the
order (every configuration equally often in every position and right after every other one)
over `n` rounds for an even number `n` of configurations and over `2n` for an odd one. The
four-configuration lists are balanced at `--rounds 4`; the five-configuration lists need
`--rounds 10` and the three-configuration one `--rounds 6`. The published runs used four
rounds throughout.

`$OUT/h2h/<bench>` then holds:

- `spec.json`: the resolved spec (absolute paths, the session's config and accuracy limits).
- `r<N>/<config>.json`: one run's result: latency quartiles and samples, accuracy on the first
  call and after timing, the limits it missed (`accuracy_violations`), measurement stability,
  structure, profile, the environment, the workload's files, the region kernels the build
  generated or loaded (`kernels`, with their origin and digests) and the reference's SHA-256.
  `r<N>/<config>.log` is its log.
- `summary.json`: per configuration, the median over rounds of the runs' median latencies, the
  smallest and largest run, the runs that succeeded, every limit a run missed, unstable runs,
  failures and the ratio to `trt-session`; and the reference outputs and code versions the
  runs used (the table warns when there is more than one of either).

`python tools/llm/bench_compare.py summary $OUT/h2h/<bench> [--base trt-baseline]` prints the
table again without changing the directory (`--write` also rewrites `summary.json`). `--resume`
continues an interrupted head-to-head, or adds rounds or configurations; it runs again the runs
that crashed or timed out.

How this protocol differs from the published runs, which used an earlier version of the
driver:

- Every run has its own `TMPDIR`, emptied before it starts. The published driver gave the runs
  of a round one shared `TMPDIR`, where Torch-TensorRT keeps TensorRT's timing cache, so all but
  the round's first TensorRT build reused timing results. Build times (`build_s`), and possibly
  TensorRT's tactic choices, can differ; the latency measurement itself is the same.
- The published driver ran the configurations of a round in a rotated order, reversed every
  other round, which for two configurations repeats the same order in every round; this one
  uses the balanced orders above.
- If the driver stops, the run in progress stops with it and is measured again on `--resume`,
  and runs that crashed or timed out are measured again too.

## 4. The hybrid head-to-head

```bash
hybrid() {  # BENCH: Inductor's kernels inside the session's config
  python tools/llm/bench_compare.py all "$KIT/specs/$1-hybrid.json" "$OUT/hybrid/$1" \
    --session "$OUT/sessions/$1" --reference "$OUT/sessions/$1/reference.pt" \
    --rounds 4 --quiet-min-free-mb 12000
}
for bench in modernbert-512 modernbert-8k gemma3-prefill-2k gemma3-prefill-8k \
             qwen3-decode-8k nemotron-decode nemotron-prefill-2k qwen35-prefill-2k; do
  hybrid $bench
done
```

| Benchmark | `head_to_head` of `specs/<bench>-hybrid.json` |
|---|---|
| `qwen3-decode-8k` | `trt-session`, `trt-session-step-graph`, `trt-hybrid-attention-flex`, `trt-hybrid-attention`, `inductor-max-autotune-flex` |
| `gemma3-prefill-2k` | `trt-session`, `trt-session-cudagraphs`, `trt-hybrid-attention-fa2-global`, `trt-hybrid-rope-plugin`, `inductor-fp16-fp16acc-max-autotune-torchflash-cdt` |
| `gemma3-prefill-8k` | `trt-session`, `trt-session-cudagraphs`, `trt-hybrid-attention-rope-inductor`, `trt-hybrid-attention-fa2-all`, `inductor-fp16-fp16acc-max-autotune-torchflash` |
| `modernbert-512` | `trt-session`, `trt-hybrid-rope-kernel`, `trt-decompose-cudagraphs`, `trt-decompose-hybrid-rope-kernel`, `inductor-max-autotune-flex-fp16acc` |
| `modernbert-8k` | `trt-session`, `trt-hybrid-rope-kernel`, `trt-decompose-cudagraphs`, `trt-decompose-hybrid-rope-kernel`, `inductor-max-autotune-flex-cdt-fp16acc` |
| `qwen35-prefill-2k` | `trt-session`, `trt-hybrid-attention`, `trt-hybrid-attention+prologue-decay`, `inductor-functional-causal-tf32-fp16-fp16acc-default` |
| `nemotron-decode` | `trt-session`, `trt-session-whole-graph`, `trt-hybrid-lm-head`, `inductor-max-autotune-flex` |
| `nemotron-prefill-2k` | `trt-session`, `trt-hybrid-mamba-inductor`, `inductor-max-autotune-fp16-fp16acc-tf32-ssd-grouped-chunk128` |

These are the configurations of the published hybrid head-to-heads, which also ran four rounds
(balanced only for the four-configuration lists; see section 3). A `trt-hybrid-<value>`
configuration is the session's config with `kernels.inductor=<value>` and the settings that
value requires; the `trt-session-*` configurations are controls with the hybrids' capture
(CUDA graphs, the decode step graph) and no hybrid. The notes in each spec say what every
configuration changes. Hybrids with Inductor regions cannot be saved (`torch_tensorrt.save`
refuses them), so a session that keeps one writes its config but no artifact.

## 5. Follow-up sessions

Hybrid sessions start from a session's config, in a new directory, with the hybrid knobs
available, the hybrid guidance and a smaller budget (plus `--quiet-gpu-s 120` for the Nemotron
benchmarks and `--trial-timeout-s 2400` for Qwen3.5):

```bash
winner() {  # BENCH: the config its session chose, as JSON
  python -c 'import json, sys; print(json.dumps(json.load(open(sys.argv[1]))["winner"]["config"]))' "$OUT/sessions/$1/report.json"
}
agentic qwen3-decode-8k "$OUT/sessions/qwen3-decode-8k-hybrid" --baseline "$(winner qwen3-decode-8k)" \
  --planner llm --provider claude --guidance "$KIT/briefs/qwen3-decode-8k-hybrid.md" $MEASURE \
  --max-trials 5 --patience 3 --max-time-s 3600
```

A planner ablation runs the sessions of section 1 again, with the same budgets, in new
directories: with the rule-based planner (`--planner rule-based`, no guidance) and with the LLM
planner without `--guidance`. For example:

```bash
agentic qwen3-decode-8k "$OUT/sessions/qwen3-decode-8k-rule-based" --planner rule-based $MEASURE \
  --max-trials 8 --patience 4 --max-time-s 4500 --trial-timeout-s 1800
```

These commands search the knobs of this commit, so compare their results with sessions you
run at this commit. The published follow-up sessions differ in two ways that the kit does not
reproduce: the ablation arms searched the knobs of the LLM-planner session they are compared with
(a wrapper factory, not part of the kit, kept the knobs listed in that session's
`session.json` and built every other knob at its default), and the hybrid and ablation sessions
ran with the final export turned off, to save disk space. The export runs after the search and
does not change it.

## 6. Where the kernels come from

`specs/<bench>-kernels.json` runs one kind of region of the model with kernels from three
sources: TensorRT's own kernels (`trt-alone`: TensorRT alone, with CUDA graphs, on the base of
the best hybrid of section 4), that best hybrid (Inductor's kernels), and kernels that PR 4914's
`torch_tensorrt.region.KernelGenerator` writes for the region during the build, on the
`trt-alone` base; the fourth configuration is the strongest Inductor one of the hybrid spec.
The generated values are knob values of the workloads (`kernels.generated`,
`kernels.gdn_scan=generated`, `kernels.decode_blocks=generated`), and every workload builds its
generator with the same settings (`tools/llm/region_generation.py`).

| Spec | Regions with generated kernels | `head_to_head` |
|---|---|---|
| [specs/gemma3-prefill-2k-kernels.json](specs/gemma3-prefill-2k-kernels.json) | each layer's q and k RMSNorm + RoPE (`kernels.generated=qk-norm-rope`) | `trt-alone`, `trt-hybrid-rope-plugin`, `trt-generated-qk-norm-rope`, `inductor-fp16-fp16acc-max-autotune-torchflash-cdt` |
| [specs/gemma3-prefill-8k-kernels.json](specs/gemma3-prefill-8k-kernels.json) | the same | `trt-alone`, `trt-hybrid-attention-rope-inductor`, `trt-generated-qk-norm-rope`, `inductor-fp16-fp16acc-max-autotune-torchflash` |
| [specs/modernbert-512-kernels.json](specs/modernbert-512-kernels.json) | each layer's QKV split + RoPE (`kernels.generated=split-rope`) | `trt-alone`, `trt-decompose-hybrid-rope-kernel`, `trt-decompose-generated-split-rope`, `inductor-max-autotune-flex-fp16acc` |
| [specs/modernbert-8k-kernels.json](specs/modernbert-8k-kernels.json) | the same | `trt-alone`, `trt-decompose-hybrid-rope-kernel`, `trt-decompose-generated-split-rope`, `inductor-max-autotune-flex-cdt-fp16acc` |
| [specs/qwen35-prefill-2k-kernels.json](specs/qwen35-prefill-2k-kernels.json) | each Gated DeltaNet layer's state scan (`kernels.gdn_scan=generated`) | `trt-alone`, `trt-hybrid-attention+prologue-decay`, `trt-scan-generated`, `inductor-functional-causal-tf32-fp16-fp16acc-default` |
| [specs/nemotron-decode-kernels.json](specs/nemotron-decode-kernels.json) | each MLP block: RMSNorm + up GEMV + ReLU², and down GEMV + residual add (`kernels.generated=mlp`) | `trt-alone`, `trt-hybrid-lm-head`, `trt-generated-mlp`, `inductor-max-autotune-flex` |
| [specs/qwen3-128-kernels.json](specs/qwen3-128-kernels.json), [specs/qwen3-8k-kernels.json](specs/qwen3-8k-kernels.json) | the Qwen3-0.6B decoder blocks (`kernels.decode_blocks=generated`), after 128 and 8192 prompt tokens | `trt-default`, `trt-static_v3-graphs`, `trt-blocks-handwritten`, `trt-blocks-generated`, `inductor-max-autotune-flex` |
| [specs/qwen3-128-session-kernels.json](specs/qwen3-128-session-kernels.json), [specs/qwen3-8k-session-kernels.json](specs/qwen3-8k-session-kernels.json) | the same, with the kernels an Auto-Tuner session prepared (see below) | the same |

`trt-alone` extends `trt-session`, the published session's config, so `--session DIR` moves
the whole comparison onto your session's config; ModernBERT's base (decomposed attention) is not
the session's config, and its session-based configurations are in the spec, off the
head-to-head. Each spec's notes say what every configuration changes, and its configurations
off the head-to-head are the region-matched alternatives (for example Inductor's hosted kernels
for the same regions).

A generated kernel is cached in the workload's kernel cache (its `kernel_cache` argument, else
`$TORCHTRT_KERNEL_CACHE`, else `/dev/shm/exp/kcache`), keyed by the region, the generator's
settings and notes, and the engine options. A build that finds no kernel for its key generates
one, with the `claude` CLI logged in, which can take up to 30 minutes per distinct region. Fill
the cache with the configuration's own build before timing it, so that the head-to-head reuses
exactly the kernels that build made:

```bash
kernels() {  # BENCH GENERATED_CONFIG: fill the kernel cache, then the head-to-head
  local spec=$KIT/specs/$1-kernels.json dir=$OUT/kernels/$1
  python tools/llm/bench_compare.py reference "$spec" "$dir"
  flock "$BENCH_GPU_LOCK" python tools/llm/bench_compare.py run "$spec" "$2" "$dir/fill.json" \
    --reference "$dir/reference.pt"
  python tools/llm/bench_compare.py all "$spec" "$dir" --rounds 4 --quiet-min-free-mb 12000
}
kernels gemma3-prefill-2k trt-generated-qk-norm-rope
```

`run` has no time limit. The `timeout_s` of the six specs other than the Qwen3 ones also
leaves room for the generator's 1800 s per distinct region, should a timed run find the cache
empty. An Auto-Tuner session fills the cache on its own: it prepares each generated value once,
before its search (see the knobs' `prepare`), and holds later runs to the kernels it made.

The `*-session-kernels` specs time the decoder blocks that the Qwen3 session of the deck's
Part 2 generated. Point the workload at that session's kernel cache and switch generation off,
so that every build loads those kernels and none is generated during timing (a kernel missing
from the cache then fails the build):

```bash
TORCHTRT_KERNEL_CACHE=/path/to/the/session/kernel_cache TORCH_TENSORRT_REGION_GENERATION=0 \
  python tools/llm/bench_compare.py all $KIT/specs/qwen3-128-session-kernels.json \
  $OUT/kernels/qwen3-128-session --rounds 4 --quiet-min-free-mb 12000
```
