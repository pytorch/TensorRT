# Optimizing LLMs in Torch-TensorRT

This directory provides utilities and scripts for compiling, optimizing, and benchmarking Large Language Models (LLMs) and Visual Language Models (VLMs) using Torch-TensorRT, with a focus on efficient inference on NVIDIA GPUs. The main entry points are `run_llm.py` for text-only LLMs and `run_vlm.py` for vision-language models. Note that this is an **experimental release** and APIs may change in future versions.

### Key Features

- **Model Support:** Works with popular LLMs such as Llama-3, Qwen2.5, etc.
- **VLM Support:** Supports Visual Language Models like Qwen2.5-VL and Eagle2.
- **Precision Modes:** Supports FP16, BF16, and FP32.
- **Quantization:** Supports FP8 and NVFP4 quantization formats for reduced memory usage and improved inference speed.
- **KV Cache:** Supports static and dynamic KV cache for efficient autoregressive decoding.
- **Benchmarking:** Measures and compares throughput and latency for PyTorch and TensorRT backends.
- **Custom Attention:** Registers and converts custom scaled dot-product attention (SDPA) for compatibility with TensorRT.


### Supported Models

We have officially verified support for the following models:

| Model Series | HF Model Card | Precision | KV Cache Supported ? |
|--------------|---------------|-----------|-------------------|
| GPT-2 | gpt2<br>gpt2-medium | FP16, FP32 | Yes |
| LLaMA 2 | meta-llama/Llama-2-7b-chat-hf | FP16, FP32 | Yes |
| LLaMA 3.1 | meta-llama/Llama-3.1-8B-Instruct | FP16, FP32 | Yes |
| LLaMA 3.2 | meta-llama/Llama-3.2-1B-Instruct<br>meta-llama/Llama-3.2-3B-Instruct | FP16, FP32 | Yes |
| Qwen 2.5 | Qwen/Qwen2.5-0.5B-Instruct<br>Qwen/Qwen2.5-1.5B-Instruct<br>Qwen/Qwen2.5-4B-Instruct<br>Qwen/Qwen2.5-7B-Instruct | FP16, FP32 | Yes |
| Qwen 3 | Qwen/Qwen3-0.6B<br>Qwen/Qwen3-1.7B<br>Qwen/Qwen3-4B<br>Qwen/Qwen3-8B | FP16, FP32 | Yes |
| Gemma 3 | google/gemma-3-1b-it | FP16, FP32 | Yes |

### Supported VLM Models

| Model Series | HF Model Card | Precision | KV Cache Supported ? |
|--------------|---------------|-----------|-------------------|
| Qwen 2.5 VL | Qwen/Qwen2.5-VL-3B-Instruct | FP16, FP32 | Yes |
| Eagle2 | nvidia/Eagle2-2B | FP16, FP32 | Yes |

### Usage

#### Text-only LLMs: `run_llm.py`

```bash
python run_llm.py --model meta-llama/Llama-3.2-1B-Instruct --prompt "What is parallel programming?" --model_precision FP16 --num_tokens 128 --cache static_v2 --benchmark
```

#### Vision Language Models: `run_vlm.py`

```bash
python run_vlm.py --model nvidia/Eagle2-2B --precision FP16 --num_tokens 128 --cache static_v1 --enable_pytorch_run --benchmark
```

#### Key Arguments

- `--model`: Name or path of the HuggingFace LLM/VLM.
- `--tokenizer`: (Optional) Tokenizer name; defaults to model.
- `--prompt`: Input prompt for generation.
- `--image_path`: (Optional) Path to input image file for VLM models. If not provided, will use a sample image.
- `--model_precision`: Precision of model weight/buffer (`FP16`, `BF16`, `FP32`).
- `--quant_format`: (Optional) Quantization format (`int8`,  `fp8`, `nvfp4`) to apply.
- `--quant_algo`: (Optional) Quantization algorithm (`max`, `smoothquant`), by default it is `max`.
- `--weight_only`: (Optional) weight only quantization flag, by default it False.
- `--num_tokens`: Number of output tokens to generate.
- `--cache`: KV cache type (`static_v1`, `static_v2`, or empty for no KV caching).
- `--benchmark`: Enable benchmarking mode.
- `--enable_pytorch_run`: Also run and compare PyTorch baseline.

### Tensor-parallel export: FP16 and FP32

[`tensor_parallel_llm_export.py`](tensor_parallel_llm_export.py) exports
Llama/Qwen models into per-rank TensorRT engines. `--precision fp16` is the default
and uses FP16 autocast. `--precision fp32` converts floating model weights and
buffers to FP32 before tracing and disables autocast. TensorRT compilation disables
TF32, and the FP32 eager reference uses PyTorch's highest float32 matmul precision.

Run these commands from the repository root with two GPUs. Export a cached FP32
Qwen model, then reload it in a separate process:

```bash
torchtrtrun --nproc_per_node=2 tools/llm/tensor_parallel_llm_export.py \
    --mode export --model Qwen/Qwen2.5-0.5B-Instruct \
    --precision fp32 --cache static_v2 --save_dir /tmp/qwen_tp_fp32 \
    --prompt "What is tensor parallelism?" --num_tokens 128

torchtrtrun --nproc_per_node=2 tools/llm/tensor_parallel_llm_export.py \
    --mode load --model Qwen/Qwen2.5-0.5B-Instruct \
    --precision fp32 --cache static_v2 --save_dir /tmp/qwen_tp_fp32 \
    --prompt "What is tensor parallelism?" --num_tokens 128
```

Both `static_v1` and `static_v2` are supported. To run v1, replace
`--cache static_v2` with `--cache static_v1` in both commands and use a separate
save directory such as `/tmp/qwen_tp_fp32_v1`.

Use `--precision fp16` and a different save directory for FP16. Changing the flag
in load mode does not convert an existing engine: re-export when changing
precision, and keep the load flag consistent with the saved engine. The saved
sequence/cache capacity must cover the inference prompt and requested tokens.
FP32 uses more memory. Performance measurements for both precisions are reported
in [Benchmark experiment results](#benchmark-experiment-results).

#### Accuracy check

Compare the saved cached engines with a full, unsharded eager reference using the
same token history at every step. Each GPU must also fit the reference model.
Save this snippet as `/tmp/check_tp_accuracy.py`; its settings match the FP32
export command above. The same snippet works with `static_v1` engines by setting
`engine_dir` to their save directory. `start` and `end` select the cache positions
to update.

```python
import gc

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

import tensor_parallel_llm_export as example  # Initializes NCCL and the device.
import torch_tensorrt
from torch_tensorrt.distributed._nccl_utils import initialize_nccl_comm
from utils import get_zeroed_static_cache_inputs

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
engine_dir = "/tmp/qwen_tp_fp32"
precision = "fp32"
prompt = "What is tensor parallelism?"
steps = 128
atol, rtol = 0.001, 0.0001

assert precision in ("fp16", "fp32") and steps > 0
rank, world = dist.get_rank(), dist.get_world_size()
device = example.DEVICE
initialize_nccl_comm()
program = torch_tensorrt.load(example._rank_path(engine_dir, rank, world))
loaded = program.module()
reference = (
    AutoModelForCausalLM.from_pretrained(
        model_id, use_cache=False, attn_implementation="sdpa"
    )
    .to(device)
    .eval()
)
if precision == "fp32":
    reference = reference.float()
    torch.set_float32_matmul_precision("highest")

tokenizer = AutoTokenizer.from_pretrained(model_id)
sequence = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
dist.broadcast(sequence, src=0)
current = sequence
kv = get_zeroed_static_cache_inputs(loaded, device=device)
if precision == "fp32":
    assert all(t.dtype == torch.float32 for t in kv), "Re-export in FP32"
for node in program.graph.nodes:
    value = node.meta.get("val")
    if node.op == "placeholder" and isinstance(value, torch.SymInt):
        capacity = program.range_constraints[value.node.expr].upper
        assert sequence.shape[1] + steps - 1 <= capacity, "Re-export with more capacity"

max_difference, token_matches = 0.0, 0
with torch.inference_mode(), torch.autocast(
    "cuda", dtype=torch.float16, enabled=precision == "fp16"
):
    for step in range(steps):
        end = sequence.shape[1]
        start = 0 if step == 0 else end - 1
        positions = torch.arange(end, device=device).unsqueeze(0)
        outputs = loaded(current, positions[:, start:], *kv, start, end)
        actual, kv = outputs[0][:, -1, :], outputs[1:]
        expected = reference(sequence, position_ids=positions).logits[:, -1, :]
        difference = (actual.float() - expected.float()).abs().max().item()
        max_difference = max(max_difference, difference)
        token_matches += int(torch.equal(actual.argmax(-1), expected.argmax(-1)))
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=atol, rtol=rtol
        )
        # Feed the reference's token to both paths for the next comparison.
        current = expected.argmax(-1, keepdim=True)
        dist.broadcast(current, src=0)
        sequence = torch.cat([sequence, current], dim=1)

print(
    f"rank={rank}: max logit difference={max_difference:.7f}; "
    f"matching tokens={token_matches}/{steps}",
    flush=True,
)
# Release the engine before destroying its communicator.
del loaded, program, reference, outputs, actual, expected, kv
gc.collect()
dist.barrier()
dist.destroy_process_group()
```

Run it from the repository root:

```bash
PYTHONPATH=tools/llm torchtrtrun --nproc_per_node=2 /tmp/check_tp_accuracy.py
```

For FP16 engines, set `precision = "fp16"` and the matching `engine_dir`.
Use `atol=rtol=0.02` as the default FP16 check; the measured
Qwen2.5-0.5B-Instruct / `static_v2` / TP=2 / batch=1 configuration needs
`atol=0.08, rtol=0.02`. The FP32 snippet uses `atol=0.001, rtol=0.0001` to allow
small floating-point differences between TensorRT and PyTorch eager.
The numerical tolerance does not require identical next-token choices, which
are reported separately. Export and load do not take tolerance arguments.

#### Precision findings

Settings: pretrained Qwen2.5-0.5B-Instruct, B300 GPUs, TensorRT 11.3.0.99,
PyTorch 2.15.0.dev20261005+cu130, Transformers 5.14.1, TP=2, batch size 1,
with both `static_v1` and `static_v2`. Each cache/precision combination was checked
on the same 904 contexts: six prompts x 64 steps, four x 128 steps, and eight
steps from synthetic tokens `[4, 5, 6, 7]`. All runs followed the original FP16
reference's token histories; matching results across the two ranks are counted once.

**TRT versus full eager reference**

Each TRT engine was compared with eager PyTorch at the same precision. The logit
checks below use `atol=0.08, rtol=0.02` for FP16 and `atol=0.001, rtol=0.0001`
for FP32.

| Cache | Precision | Largest absolute logit difference | Matching next-token choices | Passing logit checks |
|---|---|---:|---:|---:|
| `static_v1` | FP16 autocast | 0.0957031 | 902 / 904 | 903 / 904 |
| `static_v1` | FP32 | 0.0006673 | 904 / 904 | 904 / 904 |
| `static_v2` | FP16 autocast | 0.0805664 | 902 / 904 | 904 / 904 |
| `static_v2` | FP32 | 0.0006111 | 904 / 904 | 904 / 904 |

For `static_v1` FP16, one logit at the eighth synthetic-input step exceeded the
numerical tolerance (required `atol` approximately 0.08412 at `rtol=0.02`), while
the next-token choice still matched. The tolerance remains unchanged. FP32
matched all token choices for both cache variants, with small logit differences.

**Prompt and generated text**

Both cache variants had the same two FP16 token mismatches, where the full eager
reference's top scores tied:

| Prompt and generated text before the differing token | FP16 eager | FP16 TRT | FP32 eager and TRT |
|---|---|---|---|
| Implement an LRU cache: "This implementation uses a ..." | `list` | `dictionary` | `dictionary` |
| Plan a library schedule: "a reading event for children from 3 PM to ..." | `5` | `6` | `6` |

FP16 TRT matched both FP32 implementations at those positions. These comparisons
measure numerical agreement; the library prompt did not specify the event's
ending time, so matching `6` does not establish factual correctness.

#### Benchmark experiment results

Settings: Qwen/Qwen2.5-0.5B-Instruct (revision
`7ae557604adf67be50417f59c2c2f167def9a775`), 2 x NVIDIA B300 SXM6, TP=2,
batch size 1, TensorRT 11.3.0.99, PyTorch 2.15.0.dev20261005+cu130, and
Transformers 5.14.1. These measurements used the Python source changes with
existing Torch-TensorRT `2.15.0.dev0+bc747c4ad6` runtime binaries.

All four saved engines used the same sequence profile: min=1, opt=128, max=384.
The profile accepts sequence lengths from 1 to 384; `opt=128` is the target length
for TensorRT kernel tuning. Both tested input lengths, **128 and 256 tokens**,
are within that range. The 384-token KV cache accommodates a 256-token prompt
plus 128 generated tokens. Inputs were seeded synthetic token IDs (seed 0),
broadcast to both ranks. Each run generated exactly 128 tokens without
stopping at EOS. Cases ran sequentially on the same GPU pair. Each case had two
runs, each with three warmup generations and ten measured generations per input
length. `static_v2` FP16 had a third run to investigate timing variation; all
30 samples are retained for that case.

Wall-clock timings synchronize CUDA at the phase boundaries and include token
selection. The first token comes from prefill; decode throughput counts the
remaining 127 tokens. Each sample uses the slower TP rank, and the table reports
medians across the measured samples. Timings exclude engine/model loading,
compilation, tokenization, and explicit cache creation. Eager cache growth during
forward execution is included.

The performance baseline is **cached, tensor-parallel eager PyTorch** using
Hugging Face `DynamicCache`. For the FP16 baseline, the checkpoint's BF16 model
weights are converted to FP16 once, before timing starts. Buffers that were already
FP32, such as the frequencies used for rotary position embeddings, keep that dtype.
These buffers are non-weight tensors. Autocast is disabled because the weights
have already been prepared in FP16; the model's explicit dtype conversions still apply.
The FP32 baseline uses FP32 parameters and highest float32 matmul precision.
The full, unsharded eager model remains the reference for the accuracy checks above.

First-token latency and decode throughput in this table use the 128-token input.
Both total-time columns include prefill and generation of 128 output tokens.

| Backend | Precision | First token (ms) | Decode (tokens/s) | Total: 128-token input (ms) | Total: 256-token input (ms) |
|---|---|---:|---:|---:|---:|
| Cached eager PyTorch | FP16 | 17.60 | 60.2 | 2,126 | 2,139 |
| TRT `static_v1` | FP16 | 4.55 | 239.2 | 535 | 536 |
| TRT `static_v2` | FP16 | 4.51 | 246.4 | 520 | 523 |
| Cached eager PyTorch | FP32 | 17.34 | 71.0 | 1,806 | 1,804 |
| TRT `static_v1` | FP32 | 8.20 | 156.0 | 822 | 845 |
| TRT `static_v2` | FP32 | 8.18 | 155.1 | 827 | 851 |

For the 128-token input, TRT reduced total generation time by approximately
4x in FP16 and 2.2x in FP32 relative to cached eager at the same precision.
These results describe this model, hardware, and workload.

All four new engines also passed logit checks against full eager at 16 positions
each (eight steps at each input length), with matching next-token choices. Those
short checks do not replace the extended [precision findings](#precision-findings).

**Benchmark changes**

- The export/load benchmark calls pass named arguments to `time_generate_split`.
- `--seed` makes benchmark inputs repeatable, and rank 0 broadcasts them to every shard.
- `--warmup` controls untimed generations; benchmark sizes and iteration counts are validated.
- Time to first token includes the first `argmax`; throughput counts all batch elements.
- The exported input profile uses the requested batch size.

**Commands**

Run these Bash commands from the repository root. Select the same two GPUs for
all timed cases. Pin the model snapshot and use separate engine directories for
each cache/precision combination:

```bash
export CUDA_VISIBLE_DEVICES=1,2
export PYTHONPATH="${PWD}/tools/llm${PYTHONPATH:+:${PYTHONPATH}}"
TP_BENCH_MODEL=$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775"))')
TP_BENCH_ENGINES=/tmp/qwen_tp_benchmark_engines
TP_BENCH_RESULTS=/tmp/qwen_tp_benchmark_results

for precision in fp16 fp32; do
    for cache in static_v1 static_v2; do
        torchtrtrun --nproc_per_node=2 tools/llm/tensor_parallel_llm_export.py \
            --mode export --model "$TP_BENCH_MODEL" \
            --precision "$precision" --cache "$cache" \
            --save_dir "$TP_BENCH_ENGINES/$cache-$precision" \
            --benchmark --batch_size 1 --isl 128 --num_tokens 256 \
            --seed 0 --warmup 0 --iterations 1
    done
done
```

Export uses `--isl 128 --num_tokens 256` to create the opt=128, max=384 profile.
Ignore the timing printed during export: the measured workloads below generate
128 tokens, and load the saved engines in separate processes.

Save the following snippet as `/tmp/benchmark_tp.py`. It measures both input
lengths, supports the cached eager baseline and either saved TRT cache variant,
and writes one JSON file per case/run. It prints pooled medians as runs finish.
Use a fresh results directory when changing the model, hardware, or settings.

```python
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, DynamicCache

import tensor_parallel_llm_export as example  # Initializes NCCL and the device.
from torch_tensorrt.distributed._nccl_utils import initialize_nccl_comm
from utils import _timed_generate_static_cache

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument(
    "--backend", choices=["eager", "static_v1", "static_v2"], required=True
)
parser.add_argument("--precision", choices=["fp16", "fp32"], required=True)
parser.add_argument("--engine-root", required=True)
parser.add_argument("--results", required=True)
parser.add_argument("--round", type=int, required=True)
parser.add_argument("--warmup", type=int, default=3)
parser.add_argument("--iterations", type=int, default=10)
args = parser.parse_args()
assert args.warmup >= 0 and args.iterations > 0
rank, world, device = example.rank, example.world_size, example.DEVICE
assert world == 2
initialize_nccl_comm()
tokenizer = AutoTokenizer.from_pretrained(args.model)
torch.set_float32_matmul_precision("highest")

if args.backend == "eager":
    model = example.get_exportable_model(args, rank, world)
    if args.precision == "fp16":
        # Convert weights once; retain FP32 buffers such as rotary frequencies.
        for parameter in model.parameters():
            parameter.data = parameter.data.to(torch.float16)
else:
    directory = Path(args.engine_root) / f"{args.backend}-{args.precision}"
    model = torch.export.load(example._rank_path(directory, rank, world)).module()


def timed_eager(inputs):
    cache = DynamicCache(config=model.config)
    length = inputs.shape[1]
    positions = torch.arange(length, device=device).unsqueeze(0)
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = model(
        inputs, position_ids=positions, past_key_values=cache, use_cache=True
    )
    token = output.logits[:, -1, :].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    ttft = time.perf_counter() - start
    start = time.perf_counter()
    for index in range(length, length + 127):
        positions = torch.tensor([[index]], device=device, dtype=torch.int64)
        output = model(
            token, position_ids=positions, past_key_values=cache, use_cache=True
        )
        token = output.logits[:, -1, :].argmax(-1, keepdim=True)
    torch.cuda.synchronize()
    decode = time.perf_counter() - start
    assert cache.get_seq_length() == length + 127
    return {"ttft_s": ttft, "decode_s": decode, "total_s": ttft + decode}


rows = []
with torch.no_grad(), torch.autocast(
    "cuda",
    dtype=torch.float16,
    enabled=args.backend != "eager" and args.precision == "fp16",
):
    for length in (128, 256):
        inputs = torch.randint(
            tokenizer.vocab_size,
            (1, length),
            generator=torch.Generator().manual_seed(0),
        ).to(device)
        dist.broadcast(inputs, src=0)
        for iteration in range(-args.warmup, args.iterations):
            dist.barrier()  # Outside the timed region; every rank must participate.
            if args.backend == "eager":
                timing = timed_eager(inputs)
            else:
                timing = _timed_generate_static_cache(
                    model, inputs, length + 128, tokenizer.eos_token_id
                )
                assert timing["decode_tokens"] == 127
            if iteration >= 0:
                rows.append({"isl": length, **timing})
        dist.barrier()

# Aggregate after timing. Each sample uses the slower TP rank for each metric.
per_rank = [None] * world
dist.all_gather_object(per_rank, rows)
if rank == 0:
    merged = [
        {
            "isl": row["isl"],
            **{
                key: max(rank_rows[index][key] for rank_rows in per_rank)
                for key in ("ttft_s", "decode_s", "total_s")
            },
        }
        for index, row in enumerate(rows)
    ]
    result_dir = Path(args.results)
    result_dir.mkdir(parents=True, exist_ok=True)
    case = f"{args.backend}-{args.precision}"
    (result_dir / f"{case}-round{args.round}.json").write_text(
        json.dumps(merged, indent=2)
    )
    pooled = [
        row
        for path in sorted(result_dir.glob(f"{case}-round*.json"))
        for row in json.loads(path.read_text())
    ]
    for length in (128, 256):
        samples = [row for row in pooled if row["isl"] == length]
        print(
            {
                "case": case,
                "input_tokens": length,
                "output_tokens": 128,
                "samples": len(samples),
                "median_ttft_ms": 1000
                * statistics.median(row["ttft_s"] for row in samples),
                "median_decode_tokens_s": statistics.median(
                    127 / row["decode_s"] for row in samples
                ),
                "median_total_ms": 1000
                * statistics.median(row["total_s"] for row in samples),
            },
            flush=True,
        )

del model
dist.barrier()
dist.destroy_process_group()
os._exit(0)  # Same shutdown convention as the export example.
```

Run two rounds in opposite case order, with each process completing before the
next starts:

```bash
for round in 0 1; do
    cases=(eager:fp16 static_v1:fp16 static_v2:fp16 eager:fp32 static_v1:fp32 static_v2:fp32)
    if [ "$round" -eq 1 ]; then
        cases=(static_v2:fp32 static_v1:fp32 eager:fp32 static_v2:fp16 static_v1:fp16 eager:fp16)
    fi
    for case in "${cases[@]}"; do
        torchtrtrun --nproc_per_node=2 /tmp/benchmark_tp.py \
            --model "$TP_BENCH_MODEL" --backend "${case%:*}" --precision "${case#*:}" \
            --engine-root "$TP_BENCH_ENGINES" --results "$TP_BENCH_RESULTS" \
            --round "$round" --warmup 3 --iterations 10
    done
done

# The experiment included one additional static_v2 FP16 run.
torchtrtrun --nproc_per_node=2 /tmp/benchmark_tp.py \
    --model "$TP_BENCH_MODEL" --backend static_v2 --precision fp16 \
    --engine-root "$TP_BENCH_ENGINES" --results "$TP_BENCH_RESULTS" \
    --round 2 --warmup 3 --iterations 10
```

For a quick TRT-only check, the export script's load mode also exposes the timing
helpers directly. This prints rank 0's summary, including **mean** throughput;
the comparison snippet above instead pools the slower rank's samples and reports
**median** throughput:

```bash
torchtrtrun --nproc_per_node=2 tools/llm/tensor_parallel_llm_export.py \
    --mode load --model "$TP_BENCH_MODEL" --precision fp16 --cache static_v2 \
    --save_dir "$TP_BENCH_ENGINES/static_v2-fp16" \
    --benchmark --batch_size 1 --isl 128 --num_tokens 128 \
    --seed 0 --warmup 3 --iterations 10
```

### Quantization

Torch-TensorRT supports quantization to reduce model memory footprint and improve inference performance:

#### Using Pre-quantized Models

To use pre-quantized models from HuggingFace:
If a model contains quantization configuration (detected automatically), the model's linear layers are converted to TensorRT quantized versions using the specified quantization algorithm (e.g., FP8, NVFP4). The quantization algorithm type is displayed during conversion.

**Note:** The `--quant_format` option will raise an error if it's used with pre-quantized models, as quantization cannot be applied to models that are already quantized.

```bash
python run_llm.py --model nvidia/Llama-3.1-8B-Instruct-FP8 --prompt "What is parallel programming?" --model_precision FP16 --num_tokens 128

python run_llm.py --model google/gemma-3-1b-it  --prompt "What is parallel programming?" --model_precision FP16 --quant_format int8 --quant_algo smoothquant --num_tokens 128

python run_llm.py --model google/gemma-3-1b-it  --prompt "What is parallel programming?" --model_precision FP16 --quant_format int8 --weight-only --num_tokens 128
```

**Expected output:**
```
Model is FP8 pre-quantized hf model. Quantized linear layers are applied
```

#### Applying quantization by ModelOpt

To apply quantization to non-quantized models using ModelOpt:
The `--quant_format` option calls `mtq.quantize()` to apply ModelOpt post-training quantization to the model.

```bash
python run_llm.py --model meta-llama/Llama-3.1-8B --quant_format fp8 --prompt "What is parallel programming?" --model_precision FP16 --num_tokens 128
```

#### Quantization Requirements

- **ModelOpt Library**: Required for quantization operations
- **FP8**: Supported on Hopper and Blackwell-generation GPUs.
- **NVFP4**: Supported on Blackwell-generation GPUs.

### Caching Strategies

- **Static Cache v1/v2:** Adds static KV cache tensors as model inputs/outputs for efficient reuse.
- **Static Cache v3** (`static_cache_v3.py`): a shape-static, single-token decode engine for batch 1.
  v1/v2 slice the cache to the current length, so TensorRT synchronizes with the host twice per
  layer and step and the engine cannot be captured in a CUDA graph. v3 writes the new key and value
  in place (`index_copy`, lowered to `IKVCacheUpdateLayer`) and attends over the whole cache with a
  position mask and native grouped-query attention. Prefill runs one token at a time. Used by the
  agentic decode workload below.
- **No Cache:** Standard autoregressive decoding.

Please read our tutorial on how static cache is implemented.

### Fused decoder-block kernels (Qwen3)

`qwen3_regions.py` runs whole Qwen3 decoder blocks as fused Triton kernels inside the
TensorRT decode engine, through `torch_tensorrt.region.fused_kernel` (`qwen3_kernels.py` holds
the kernels). Each block replaces several small TensorRT kernels and the gaps between them:

| Block | Fuses |
|---|---|
| `attn_in` | [residual add +] input RMSNorm + q/k/v GEMVs (three outputs) |
| `qk_rope` | q/k per-head RMSNorm + RoPE, with the softmax scale folded into q |
| `attn_out` | o_proj GEMV + residual add |
| `mlp_in` | [residual add +] post-attention RMSNorm + gate/up GEMVs + SiLU(gate) * up |
| `mlp_out` | down GEMV + residual add |
| `final` | final RMSNorm + lm_head GEMV |

```python
import qwen3_regions

qwen3_regions.patch_qwen3(model, qwen3_regions.PRESETS["layers"])  # a loaded FP16 Qwen3ForCausalLM
decode = qwen3_regions.compile_decode(model, max_seq_len=192, device=torch.device("cuda:0"))
torch_tensorrt.runtime.set_cudagraphs_mode(True)  # required, see below
```

The KV-cache write and attention stay in TensorRT (`static_cache_v3`). The kernels accumulate in
FP32 and round each output once. On an RTX 5090 (FP16, batch 1, CUDA graphs) the `layers` preset
decodes Qwen3-0.6B in 1.02-1.03 ms/token against 1.18 ms/token without regions (339 to 202
kernels per token), with top-1 agreement 1.0 and half the KL divergence. The `final` block is
neutral: TensorRT's lm_head GEMV already runs at DRAM bandwidth. Constraints:

- Run under CUDA graphs: TensorRT 11.4 reloads a QDP plugin's module on every enqueue otherwise.
- Batch 1, single-token decode, FP16 weights, Qwen3 (no q/k/v biases) with a power-of-two hidden
  size and head size (Qwen3-0.6B, 1.7B and 8B; not 4B, 14B or 32B); `BLOCK_N` and `BLOCK_K` are
  powers of two that divide each GEMV's rows and columns.
- Regions build with `offload_module_to_cpu=False`: release the PyTorch weights after compiling
  (`torch_tensorrt.dynamo.utils.deallocate_module(model)`), as `agentic_decode.py` does; the engine
  holds its own copy.
- Loads carry `tl.multiple_of(..., 16)` hints (the plugin compile path drops Triton's alignment
  specialization) and an L2 `evict_first` policy for weights (otherwise the streamed weights
  evict the KV cache and slow TensorRT's own kernels).

### Auto-Tuner sessions: decode

`run_agentic_decode.py` runs an Auto-Tuner session (`torch_tensorrt.autotune`) on batch-1 decode
(see the `torch_tensorrt.autotune` documentation). The workload (`agentic_decode.py`) measures
milliseconds per decoded token; decoding is teacher-forced on a fixed public-domain text, and every
step's logits are checked against an FP32 eager reference (top-1 agreement >= 0.97 and mean KL
divergence <= 0.005 by default). The session starts from the documented configuration (FP16,
`static_v2`) and searches the KV cache lowering, precision, FP32 accumulation, TensorRT build
routes and builder levels, the runtime modes (CUDA graphs per engine, or one CUDA graph for the
whole decode step with `runtime.step_graph`) and, for Qwen3 models, the fused decoder blocks and
Inductor's kernels (`kernels.inductor`, see [Hybrid kernels](#hybrid-kernels-inductors-kernels-in-tensorrt-configs)),
one validated change at a time.

```bash
python run_agentic_decode.py --session /tmp/qwen3-decode --max-trials 8
python run_agentic_decode.py --session /tmp/qwen3-decode-llm --max-trials 8 --planner llm
python -m torch_tensorrt.autotune status /tmp/qwen3-decode
```

Each trial runs in a fresh process; rerunning a command resumes its session. The session
directory holds every trial's results, the report (`report.md`) and the exported decode engine
(`artifact/decode.ep` with the winner's `config.json`).

#### Results: Qwen3-0.6B decode on an RTX 5090

Batch 1, 128 prompt tokens, 64 decoded tokens per measured call; TensorRT 11.4, Torch-TensorRT
Python runtime, driver 610.57, a GPU shared with another process. All sessions started from the
documented configuration (FP16, `static_v2`) and checked every candidate against FP32 eager logits.
They ran the code before the rename (`torch_tensorrt.agentic`, the same steps under older names).
Sessions A and B ran before `kernels.decode_blocks`, the build routes, noise runs and
head-to-head pairs existed; session C had all of them.

| Session | Planner | Trials | Baseline | Best | Final check | Best config |
|---|---|---|---|---|---|---|
| A | rule-based | 10 | 2.922 ms/token | 1.251 ms/token | +56.9% (2.34x tokens/s) | `static_v3`, `use_fp32_acc=False` |
| B | LLM (`claude`) | 6, then the planner stopped | 2.923 ms/token | 1.235 ms/token | +57.1% (2.37x tokens/s) | `static_v3`, CUDA graphs, `use_fp32_acc=False` |
| C | LLM (`claude`) with guidance | 5, then the planner stopped | 2.947 ms/token | 1.031 ms/token | +64.7% (2.86x tokens/s) | `static_v3`, CUDA graphs, `kernels.decode_blocks=all` |

Top-1 agreement stayed 1.0 (mean KL 1.9e-5 to 4.5e-5). "Final check" is the median of two fresh
baseline/best worker pairs. Process GPU memory fell from 2160 MB to about 2020 MB. (Session C
recorded 3176 MB for the fused blocks: the workload left the PyTorch weights on the GPU, since
region builds cannot offload them; it now releases them, and a fused build measures 2018 MB.)

Session C, step by step: three fresh-process runs of the baseline moved 0.8%, so the noise
threshold stayed at the 2% minimum. The LLM planner paired `static_v3` with CUDA graphs (1.29 ms/token,
+55.5% in two head-to-head pairs), then chose the fused decoder blocks (1.031 ms/token, +20.5% in its
head-to-head; 73% of memory-bandwidth-bound speed). Two edits of the blocks' launch settings (edited
file `decode_launch`) and one attention build route measured 0.3-1.4% slower and were rejected; the
planner then stopped, judging that the remaining gap in its slowest kernel was worth at most about 6%.

What the sessions measured:

- The baseline spends 31% of each token waiting on the host: `static_v2` makes TensorRT copy
  data-dependent sizes to the host twice per layer and step (58 device-to-host copies per token).
  Both planners chose `static_v3` first (about 1.38-1.41 ms/token, no host copies).
- Turning off FP32 accumulation for FP16 matrix multiplications gained another 9-12% within the
  accuracy bounds.
- No gain beyond noise: CUDA graphs (decode is GPU-bound after `static_v3`), pre-allocated outputs,
  builder optimization levels 4 and 5. Slower: BF16 (8-21%), FP32 (50%). TensorRT 11.4 failed to
  build the engine with tiling optimization (`moderate`, `fast`); the sessions recorded the error.
- Without the fused blocks, about 70% of the remaining GPU time is cuBLAS GEMV over FP16 weights
  and the rest is many small kernels (norms, residual adds, RoPE, SiLU). With them (session C),
  the Triton block kernels take about 85% of GPU time at 1.2-1.7 TB/s and TensorRT's decode
  attention 11%. The remaining levers are programmatic dependent launch between kernels, fusing the
  KV-cache write and attention into the q/k block, and weight quantization (ModelOpt).
- The exported `decode.ep` reloads in a fresh process with the same accuracy. Called through
  `torch_tensorrt.load(...).module()` it decoded at 1.50 ms/token, 0.25 ms more than in the
  session, because the loaded program checks its 58 inputs on every call.

### Auto-Tuner sessions on other workloads: `run_agentic.py`

`run_agentic.py` runs a session on any workload factory, `path/to/file.py:function` called with
JSON keyword arguments, so it serves every workload below (`run_agentic_decode.py` keeps the
decode workload's own options). Both are `python -m torch_tensorrt.autotune run` with a default
budget of 10 trials, and take its options. Rerunning a command resumes its session.

```bash
python tools/llm/run_agentic.py --session /tmp/gemma3-prefill-2k \
    --workload tools/llm/agentic_prefill.py:create_case \
    --args '{"model": "unsloth/gemma-3-1b-it", "length": 2048}' \
    --planner llm --provider claude --guidance my_guidance.md --max-trials 8 --patience 4
python -m torch_tensorrt.autotune status /tmp/gemma3-prefill-2k
```

- Planner: `--planner rule-based` (the default; no LLM needed) or `--planner llm`, which asks
  the logged-in `claude` CLI (`--provider claude`, or `codex`, or a `module:attribute` callable;
  `--llm-model` picks the model) to choose each experiment. `--guidance FILE` gives the LLM
  planner Markdown guidance, kept as `guidance.md` in the session. Keep measured results of
  candidate configs out of guidance: the planner tends to take them as given instead of
  measuring them.
- Goal and limits: `--metric latency_ms|throughput`, `--min-improvement`, `--target`,
  `--accuracy METRIC OP VALUE` (repeatable; other metrics keep the workload's defaults),
  `--max-memory-mb`, and `--baseline CONFIG` (JSON knob changes) to start from a config instead
  of the workload's baseline.
- Budget: `--max-trials`, `--max-time-s` (bounds the search; the final check and the export run
  after it), `--patience` (consecutive trials without an accepted change), `--trial-timeout-s`
  (kills a worker that builds or measures for longer) and `--prepare-timeout-s` (the same for
  preparing a generated value's kernels).
- Measurement: `--noise-runs` (fresh-process runs of the baseline that measure the noise),
  `--head-to-head-pairs` and `--final-check-pairs` (fresh best/candidate pairs before a change is
  accepted, and baseline/winner pairs at the end), `--quiet-gpu-s` (how long each worker may wait for an idle GPU; 0 turns the wait
  off), `--max-spread` and `--retries` (when a measurement counts as stable).

The workloads:

| Workload | Factory | Unit | Knobs besides the standard engine and runtime ones |
|---|---|---|---|
| Causal-LM decode (`agentic_decode.py`) | `create_case(model, isl, osl, prompt, prefill, kernel_cache)` | one decoded token, teacher-forced after an untimed prefill | `workload.cache`, `engine.precision`, `engine.use_fp32_acc`, `engine.build_route`, `runtime.step_graph`; Qwen3: `kernels.decode_blocks`, `kernels.inductor`; edited files `decode_launch`, `decode_kernels` |
| Causal-LM prefill (`agentic_prefill.py`; Gemma 3, Qwen3) | `create_case(model, length, positions, prompt, kernel_cache)` | one prefill, logits at `positions` evenly spaced positions | `workload.masks`, `workload.global_attention`, `workload.local_attention`, `engine.precision`, `engine.build_route`, `kernels.inductor`, `kernels.generated` |
| Encoder forward (`agentic_encoder.py`; ModernBERT) | `create_case(model, length, kernel_cache)` | one forward, `last_hidden_state` | `workload.local_attention` (banded), `graph.fold_attention_masks`, `engine.decompose_attention`, `engine.precision`, `engine.use_fp32_acc`, `engine.build_route`, `kernels.inductor`, `kernels.generated` |
| Qwen3.5 prefill (`qwen35_prefill.py`, Gated DeltaNet in `qwen35_gdn.py`) | `create_case(model, length, positions, kernel_cache)` | one prefill, logits at `positions` positions | `workload.gdn` (Hugging Face's fallback or a functional rewrite), `workload.conv`, `workload.chunk_size`, `kernels.gdn_scan` (a hand-written or generated Triton scan in the engine), `workload.attention`, `engine.precision`, `engine.use_fp32_acc`, `engine.disable_tf32`, `engine.decompose_attention`, `kernels.inductor`; edited file `gdn` |
| Nemotron-H decode (`nemotron_decode.py`; Mamba-2 hybrid) | `create_case(model, isl, osl, kernel_cache)` | one decoded token | `engine.precision`, `engine.use_fp32_acc`, `workload.ssm_state`, `workload.state_update`, `engine.build_route`, `runtime.cudagraphs`, `kernels.inductor`, `kernels.generated`; edited file `decode_step` |
| Nemotron-H prefill (`nemotron_decode.py`) | `create_prefill_case(model, length, positions)` | one prefill | `engine.precision`, `engine.use_fp32_acc`, `engine.tf32`, `workload.ssd`, `workload.chunk_size`, `kernels.inductor` |

Every factory also takes a `baseline` config. Every workload checks each candidate against an
FP32 eager reference, with default limits (`accuracy_defaults`) that correct builds pass and
deliberately broken ones fail (the guidance files in `agentic_bench/briefs` give the calibration),
and its module docstring describes its knobs and its baseline, a plain Torch-TensorRT compile (the
documented `tools/llm` configuration where there is one). Every workload also accepts the
head-to-head driver's `bench.*` values (below), which are not knobs: a session never sets
them.

### Hybrid kernels: Inductor's kernels in TensorRT configs

`kernels.inductor` (default `none`, so existing configs build as before) hands chosen
operations of a TensorRT config to the kernels PyTorch Inductor generates for them, in one of
two ways:

- **Hosted Inductor kernels** (`torch_tensorrt.region.fused_kernel`). At build time
  `torch.compile` compiles the subgraph for the workload's shapes, and the Triton kernel
  Inductor generates runs unchanged inside the TensorRT engine as a QDP plugin, with the launch
  settings Inductor's autotuner chose. Before the engine is built, the build launches the
  kernel once through Triton, with the plugin's arguments and launch settings, and checks its
  outputs against Inductor's own (or eager) results. That check runs Triton's compilation of
  the kernel, not the plugin's ahead-of-time binary, which only the candidate's accuracy check
  covers. The engine stays whole, but TensorRT cannot fuse across a plugin.
  Values: decode `attention-flex` (Inductor's flex-decoding kernels), prefill `rope-plugin` (q/k
  RMSNorm and RoPE) and encoder `rope-kernel` and `rope-kernel-eager-numerics` (the QKV split and
  RoPE).
- **Inductor regions** (`torch_tensorrt.region.execute_in_torch(backend="inductor")`). The
  scope stays out of TensorRT and Inductor compiles it at its first call, which the build makes.
  Every region ends one engine and starts the next (two boundaries per region), and the whole
  program, engines and regions, is captured in one CUDA graph (except with Nemotron prefill's
  `mamba-inductor`, which the host calls engine by engine). Values: decode `lm-head`,
  `attention` and `attention-math`; prefill `attention-inductor*` and `attention-rope-inductor*`;
  encoder `inductor-regions`; Qwen3.5 `attention`, `prologue`, `prologue-decay` and their
  combinations; Nemotron `lm-head` and `mlp-inductor` (decode) and `mamba-inductor` and
  `mamba-inductor-graph` (prefill).

The prefill workload's `attention-fa2-*` values run PyTorch's flash-attention operator (the
kernel Inductor calls for attention) between engines, without a region.

Requirements, which the knobs enforce before anything is built (a planner's change brings
them along; the decode knob lists them per value):

| Workload | Values | Require |
|---|---|---|
| Decode | `attention-flex` | `workload.cache=static_v3`, `engine.precision=fp16`, `runtime.cudagraphs` |
| Decode | `lm-head`, `attention`, `attention-math` | the same and `runtime.step_graph`: per-engine CUDA graphs would run the regions eagerly between engines |
| Prefill, encoder | every value | `runtime.cudagraphs` |
| Qwen3.5 | every value | `workload.gdn=functional`, `workload.attention=causal`, `runtime.cudagraphs` |
| Nemotron decode | every value | `workload.state_update=inplace`, `runtime.cudagraphs` (the workload records the whole step as one CUDA graph) |
| Nemotron prefill | none | `mamba-inductor-graph` records the forward as one CUDA graph itself |

Limits:

- Region builds cannot be saved: `torch_tensorrt.save` refuses `execute_in_torch` regions whose
  backend is `inductor`. Their candidates' export raises `NotImplementedError`, so a session that
  keeps one records the export as unsupported and writes its config but no artifact; rebuild the
  config instead.
- Both ways compile through strict export (`torch_tensorrt.compile(..., ir="dynamo",
  strict=True)`) without module offloading, so the model stays on the GPU during the build;
  Inductor's compilation and autotuning lengthen it.
- TensorRT 11.4 QDP plugins need CUDA graph replay (outside it, every enqueue reloads the plugin
  module) and at most 48 KB of shared memory.
- The hosted kernels read Inductor's generated code and launch settings through private
  Inductor internals; a shape that Inductor compiles differently is refused at build.
- More engines and regions need more GPU memory: each engine reserves its own activation memory,
  and the tensors that cross a boundary stay allocated (see each knob's description).
- Under the head-to-head's PyTorch backends (`bench.backend` `eager` or `inductor`) the prefill,
  encoder, Qwen3.5 and Nemotron prefill workloads refuse the knob. The two decode workloads
  build the PyTorch model from `engine.precision` and the `bench.*` values alone there and
  ignore every other knob, `kernels.inductor` included; they also ignore `bench.*` keys they do
  not know, which the other workloads refuse. Check a spec's PyTorch configurations against the
  workload's docstring.
- A session records its workload's knobs (each with its description), its editable files'
  sources and the workload's files, and refuses to resume after any of them changed. That
  includes a session created before its workload had the knob, or with an earlier description
  of it: start such sessions in a new directory.

### Generated kernels: regions whose kernel an LLM writes

Some regions can also run a kernel that PR 4914's `torch_tensorrt.region.KernelGenerator`
writes during the build: `fused_kernel(generator=...)` around the region's PyTorch code. The
generator has an LLM author (the `claude` CLI) propose Triton kernels, checks each against the
region's code, times it against TensorRT's own build of the region, and keeps one only if it is
at least as fast; the kernel then runs inside the engine as a QDP plugin. So the same region can
run TensorRT's kernels (the knob's default), a hand-written or Inductor-generated kernel (the
values above, and `kernels.gdn_scan=triton*`), or a generated one:

| Workload | Value | Regions (inputs) | Requires |
|---|---|---|---|
| Decode (Qwen3) | `kernels.decode_blocks=generated` | the `layers` blocks (see [Fused decoder-block kernels](#fused-decoder-block-kernels-qwen3)) | `workload.cache=static_v3`, `engine.precision=fp16`, `runtime.cudagraphs` |
| Encoder (ModernBERT) | `kernels.generated=split-rope` | `split_rope_<layer>`: each layer's QKV split + RoPE (`qkv`, `inv_freq`) | `runtime.cudagraphs`, `kernels.inductor=none` |
| Prefill (Gemma 3, Qwen3) | `kernels.generated=qk-norm-rope` | `q_norm_rope.<layer>`, `k_norm_rope.<layer>`: q/k RMSNorm + RoPE (`qkv`, the norm's weight, `cos`, `sin`) | `runtime.cudagraphs`, `kernels.inductor=none` |
| Qwen3.5 | `kernels.gdn_scan=generated` | `gdn_scan`: each layer's state scan (`w`, `u`, `k_decayed`, `gamma`) | `workload.gdn=functional`, `runtime.cudagraphs` |
| Nemotron decode | `kernels.generated=mlp` | `mlp_in_<layer>`: RMSNorm + up GEMV + ReLU² (`hidden`, the norm's weight, `up_proj.weight`); `mlp_out_<layer>`: down GEMV + residual add (the activation, `down_proj.weight`, `hidden`) | `runtime.cudagraphs`, `kernels.inductor=none` |

Every workload builds its generator with the same settings (`region_generation.py`:
`make_generator`), plus a notes file of measured facts about its regions where one applies
(`*_kernel_notes.md`); Qwen3.5's scan also accepts the error that its FP32 products have with
TF32, the precision `triton-tf32` runs at. Identical regions of one build share one kernel.
Accepted kernels are cached in the workload's `kernel_cache` directory (the factory argument,
else `$TORCHTRT_KERNEL_CACHE`, else `/dev/shm/exp/kcache`), keyed by the region, the generator's
settings and notes, and the compilation's engine options (precision, accumulation, TF32,
optimization level, build route...): a build reuses a cached kernel only with the same options,
and otherwise generates one, for up to 30 minutes per distinct region. Generate kernels with
the build that will use them, the workload's own: an Auto-Tuner session prepares each generated
value once, before its search (the knobs' `prepare`), and pins later runs to those kernels; for a
head-to-head, run the configuration once with `bench_compare.py run` first (see
[`agentic_bench/`](agentic_bench/README.md), section 6).

### Head-to-head benchmarks and the reproduction kit

`bench_compare.py` measures configurations of one workload, TensorRT configs and the same model
run eagerly or compiled by Inductor (the `bench.*` values: `bench.backend`, `bench.mode`,
`bench.attn`, `bench.inductor` and workload-specific keys), with the evaluator the sessions use.
Every run is a fresh process with cold compiler caches; rounds run the configurations in
balanced orders; each run waits for an idle GPU under a GPU lock that it inherits (`--lock`,
`$BENCH_GPU_LOCK`, and `--quiet-command`, `$BENCH_GPU_QUIET`, are configurable) and stops with
the driver; and every run is checked against the same FP32 reference outputs. `--session DIR`
measures the config an Auto-Tuner session chose.

```bash
python tools/llm/bench_compare.py all tools/llm/agentic_bench/specs/qwen3-decode-8k.json \
    /tmp/h2h/qwen3-decode-8k --session /tmp/sessions/qwen3-decode-8k --rounds 4
python tools/llm/bench_compare.py summary /tmp/h2h/qwen3-decode-8k
```

[`agentic_bench/`](agentic_bench/README.md) is the reproduction kit of the benchmark of Auto-Tuner
sessions on eight workloads (Qwen3-0.6B decode at 8k context, Gemma 3 1B prefill at 2k and 8k
tokens, ModernBERT-base at 512 and 8k tokens, Qwen3.5-0.8B prefill and Nemotron-3-Nano-4B decode
and prefill): the guidance each session read, byte for byte, the guidance of the hybrid
follow-up sessions, the head-to-head and hybrid head-to-head specs, and the exact commands.

## Extension

This codebase can be extended to
- Add new models by specifying their HuggingFace name.
- Implement new cache strategies by adding FX graph passes.
- Customize SDPA conversion for new attention mechanisms.

## Limitations
- We do not currently support sliding window attention (used in Gemma3 and Qwen 3 models) yet.
- **Flash Attention Limitation**: Some models (e.g., Eagle2-2B) internally use flash attention operations (`torch.ops.flash_attn._flash_attn_forward.default`) which require the `flash-attn` package to be installed. Without flash-attn, these models will fail to load or run properly.
- **Qwen2.5‑VL vision is not compiled (LLM-only)**: We only compile the language model for Qwen2.5‑VL. The vision encoder is skipped because its `get_window_index` relies on dynamic Python operations.

## Requirements

- Torch-TensorRT 2.8.0
- Transformers v4.52.3
- For VLM models (run_vlm.py):
  - `pip install qwen-vl-utils` (for Qwen2.5-VL-3B-Instruct model)
  - **Flash Attention**: For models using flash attention operations (e.g., Eagle2-2B), install one of the following:
    - **Fast installation (recommended)**: `pip install flash-attn==2.8.1` (pre-built wheel, should work)
    - **Source build (slow)**: `pip install flash-attn --no-build-isolation -v` (fallback if pre-built wheels fail)
