"""
.. _multi_optimization_profiles:

Multiple Optimization Profiles: Prefill vs Decode for Gemma-3
=============================================================

Autoregressive LLMs run in two very different shape *regimes* that share one set
of weights (and ideally one engine):

- **prefill**: the prompt is processed in one shot, so the sequence length
  ``seq`` is large, and
- **decode**: tokens are generated one at a time, so ``seq == 1``.

A single dynamic range ``seq in [1, max]`` works, but TensorRT can only tune
kernels for **one** ``opt`` point. Tuning for the prefill length leaves decode
(the latency-critical, most-frequently-executed phase) running on kernels picked
for a sequence it never sees.

``torch_tensorrt.Input(profiles=[...])`` declares **N optimization profiles** on
a single input. The engine is built once (a single ``torch.export`` over the
union of all profiles), each profile gets its own TensorRT kernel tuning, and you
select the active profile per call (by index, or ``"auto"``).

This example compiles `google/gemma-3-1b-it
<https://huggingface.co/google/gemma-3-1b-it>`_ **twice** -- once with a single
profile and once with separate prefill/decode profiles -- and compares the decode
and prefill latency of the two engines.

.. note::

   ``google/gemma-3-1b-it`` is a **gated** model: you must accept its license on
   the Hugging Face Hub and authenticate (``hf auth login`` or the ``HF_TOKEN``
   environment variable) before running this example. It downloads ~2 GB of
   weights on first use and requires a CUDA GPU.

.. note::

   This uses the Ahead-Of-Time (AOT) ``torch.export`` + ``dynamo.compile`` path.
   Runtime profile selection works with whichever TensorRT runtime (C++ or
   Python) the installed Torch-TensorRT build provides.
"""

# %%
# Imports and Setup
# ^^^^^^^^^^^^^^^^^^
#
# The HuggingFace attention path needs a TensorRT-friendly SDPA lowering. The
# reusable LLM helpers ``register_sdpa`` (a Gemma-3-specific SDPA pass) and
# ``export_llm`` live under ``tools/llm`` in the Torch-TensorRT repo, so we add
# that directory to ``sys.path``.

import sys
import timeit
from pathlib import Path

import torch
import torch_tensorrt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "llm"))

MODEL_ID = "google/gemma-3-1b-it"
DEVICE = torch.device("cuda:0")

# The two regimes we benchmark.
MAX_SEQ = 256  # largest prompt the engine must support
PREFILL_SEQ = 128
DECODE_SEQ = 1
DECODE_IDX, PREFILL_IDX = 0, 1


# %%
# Load the Model
# ^^^^^^^^^^^^^^
#
# Load with ``use_cache=False`` (this example recomputes over the full sequence
# rather than using a KV cache, which keeps the export simple) and the ``sdpa``
# attention implementation, then register the Gemma-3 SDPA lowering pass.
def load_model():
    from transformers import AutoModelForCausalLM

    with torch.no_grad():
        model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                use_cache=False,
                attn_implementation="sdpa",
                ignore_mismatched_sizes=True,
            )
            .eval()
            .cuda()
            .to(torch.float16)
        )
    from torchtrt_ext import register_sdpa

    register_sdpa.enable_sdpa_converter(MODEL_ID, model.config)
    return model


try:
    model = load_model()
except Exception as e:  # gated/no-auth/no-GPU environments (e.g. CI docs build)
    print(f"Skipping example: could not load {MODEL_ID} ({type(e).__name__}: {e}).")
    print("Accept the license and authenticate (hf auth login / HF_TOKEN) to run.")
    sys.exit(0)


def make_inputs(seq_len: int):
    ids = torch.randint(1, 10000, (1, seq_len), dtype=torch.int64, device=DEVICE)
    position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0)
    return ids, position_ids


# %%
# Declaring the Optimization Profiles
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``profiles`` is an ordered list; the list index is the optimization-profile
# index used at runtime. Both model inputs (``input_ids`` and ``position_ids``)
# are dynamic over ``seq``, so each gets a profiled ``Input`` with identical
# profiles:
#
# - index ``0`` -> **decode**: ``seq`` pinned to 1 (a fully static profile)
# - index ``1`` -> **prefill**: ``seq`` in ``[1, MAX_SEQ]``, tuned at ``PREFILL_SEQ``
#
# Profile order matters for auto-selection: the profiles overlap at ``seq == 1``
# and auto-selection picks the *first* profile whose ``[min, max]`` accepts the
# input, so declaring ``decode`` first lets it win the ``seq == 1`` overlap.
profiles = [
    {"min_shape": (1, 1), "opt_shape": (1, 1), "max_shape": (1, 1)},  # decode
    {
        "min_shape": (1, 1),
        "opt_shape": (1, PREFILL_SEQ),
        "max_shape": (1, MAX_SEQ),
    },  # prefill
]
multi_profile_inputs = [
    torch_tensorrt.Input(dtype=torch.int64, profiles=profiles),  # input_ids
    torch_tensorrt.Input(dtype=torch.int64, profiles=profiles),  # position_ids
]

# %%
# Export Once, Compile Twice
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``export_llm`` traces the model over a dynamic ``seq`` range. We reuse the
# exported program for both the single-profile baseline (tuned at the prefill
# length, the conventional choice) and the multi-profile engine.
from utils import export_llm  # noqa: E402

example_ids, _ = make_inputs(PREFILL_SEQ)
with torch.inference_mode():
    exported = export_llm(model, example_ids, min_seq_len=1, max_seq_len=MAX_SEQ)

# ``offload_module_to_cpu`` must stay False here: it is currently incompatible
# with the multi-profile ``Input(profiles=...)`` path (CPU/CUDA device mismatch).
common = dict(
    use_fp32_acc=True,
    disable_tf32=True,
    offload_module_to_cpu=False,
    min_block_size=1,
    require_full_compilation=True,
    device=DEVICE,
)

print("Compiling single-profile engine (tuned at prefill length) ...")
bench_ids, bench_pos = make_inputs(PREFILL_SEQ)
trt_single = torch_tensorrt.dynamo.compile(
    exported, inputs=[bench_ids, bench_pos], **common
)

print("Compiling multi-profile engine (decode + prefill) ...")
trt_multi = torch_tensorrt.dynamo.compile(
    exported, arg_inputs=multi_profile_inputs, **common
)


# %%
# Correctness
# ^^^^^^^^^^^
#
# FP16 logits over Gemma's 262K-token vocabulary are noisy, so we compare the
# *predicted token* (argmax) rather than raw logits.
def logits(out):
    return (out.logits if hasattr(out, "logits") else out).float()


from torch_tensorrt.runtime import optimization_profile  # noqa: E402

decode_ids, decode_pos = make_inputs(DECODE_SEQ)
prefill_ids, prefill_pos = make_inputs(PREFILL_SEQ)

with torch.inference_mode():
    ref_decode = logits(model(decode_ids, position_ids=decode_pos))
    ref_prefill = logits(model(prefill_ids, position_ids=prefill_pos))

    with optimization_profile(trt_multi, DECODE_IDX):
        trt_decode = logits(trt_multi(decode_ids, decode_pos))
    with optimization_profile(trt_multi, PREFILL_IDX):
        trt_prefill = logits(trt_multi(prefill_ids, prefill_pos))


def top1_match(a, b):
    return (a.argmax(-1) == b.argmax(-1)).float().mean().item()


print(f"decode  top-1 token match vs eager: {top1_match(trt_decode, ref_decode):.1%}")
print(f"prefill top-1 token match vs eager: {top1_match(trt_prefill, ref_prefill):.1%}")


# %%
# Latency Comparison
# ^^^^^^^^^^^^^^^^^^^
#
# Time each regime on each engine. For the multi-profile engine we pin the
# matching profile around the loop (the realistic serving pattern). We report the
# min over several rounds to reduce noise.
def benchmark(run, iters: int = 50, warmup: int = 20, rounds: int = 3) -> float:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        start = timeit.default_timer()
        for _ in range(iters):
            run()
        torch.cuda.synchronize()
        best = min(best, (timeit.default_timer() - start) / iters * 1000)  # ms/call
    return best


with torch.inference_mode():
    single_decode = benchmark(lambda: trt_single(decode_ids, decode_pos))
    single_prefill = benchmark(lambda: trt_single(prefill_ids, prefill_pos))
    with optimization_profile(trt_multi, DECODE_IDX):
        multi_decode = benchmark(lambda: trt_multi(decode_ids, decode_pos))
    with optimization_profile(trt_multi, PREFILL_IDX):
        multi_prefill = benchmark(lambda: trt_multi(prefill_ids, prefill_pos))

# %%
# Results. Decode is the win: the multi-profile engine dedicates a *static*
# profile (``seq`` pinned to 1) to decode, so TensorRT specializes that path
# instead of serving it from kernels tuned for the long prefill length. Prefill
# is unchanged (both engines tune it at the same ``opt``).
print("\nPer-call latency (ms), batch=1")
print(f"{'regime':<20}{'single-profile':>16}{'multi-profile':>16}{'speedup':>10}")
print("-" * 62)
print(
    f"{f'decode (seq={DECODE_SEQ})':<20}{single_decode:>16.3f}"
    f"{multi_decode:>16.3f}{single_decode / multi_decode:>9.2f}x"
)
print(
    f"{f'prefill (seq={PREFILL_SEQ})':<20}{single_prefill:>16.3f}"
    f"{multi_prefill:>16.3f}{single_prefill / multi_prefill:>9.2f}x"
)

# %%
# Summary
# ^^^^^^^
#
# - Declare ``N`` profiles on an ``Input`` with
#   ``profiles=[{min_shape, opt_shape, max_shape}, ...]``
#   (one per dynamic model input -- here ``input_ids`` and ``position_ids``).
# - One export + one engine; each profile gets its own TensorRT kernel tuning.
# - Select at runtime by **index** (``optimization_profile(m, i)``) or let
#   ``"auto"`` pick the first profile that fits the input shapes.
# - Dedicating a static ``seq == 1`` profile to decode lets TensorRT tune that
#   latency-critical path independently of the prefill length.

print("Done.")
