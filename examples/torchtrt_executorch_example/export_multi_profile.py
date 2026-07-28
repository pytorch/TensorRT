"""
.. _executorch_export_multi_profile:

Saving a Multi-Optimization-Profile Gemma-3 Model in ExecuTorch Format (.pte)
=============================================================================

Autoregressive LLMs run in two very different shape *regimes* that share one set
of weights:

- **prefill**: the prompt is processed in one shot, so the sequence length
  ``seq`` is large, and
- **decode**: tokens are generated one at a time, so ``seq == 1``.

A single dynamic range ``seq in [1, max]`` works, but TensorRT can only tune
kernels for **one** ``opt`` point. Tuning for the prefill length leaves decode --
the latency-critical, most-frequently-executed phase -- running on kernels
picked for a sequence it never sees.

``torch_tensorrt.Input(profiles=[...])`` declares **N optimization profiles** on
a single input. The engine is built **once** (a single ``torch.export`` over the
union of all profiles) and each profile gets its own TensorRT kernel tuning:

- profile ``0`` -> **decode**: ``seq`` pinned to 1 (a fully static profile)
- profile ``1`` -> **prefill**: ``seq`` in ``[1, MAX_SEQ]``, tuned at ``PREFILL_SEQ``

Run the result with ``examples/executorch_reference_runner``, which selects a
profile per call with ``OptimizationProfileGuard``, and measure what that
selection is worth with ``example_executorch_multi_profile_benchmark``.

By default this exports a **randomly initialized mini Gemma-3**: the real
architecture (sliding-window and full attention, the Gemma-3 SDPA lowering) at a
few million parameters, so the whole export takes about a minute and needs no
download or Hugging Face account. Only the shapes matter for demonstrating
optimization profiles, and the weights never leave the engine.

Pass ``--weights google/gemma-3-1b-it`` for the real 1B model. That is the
configuration the latency numbers in the runner README were measured on, and it
takes considerably longer: the ``.pte`` serialization step costs roughly 3.6
seconds per megabyte of engine, so a ~2 GB engine is a couple of hours.

.. note::

   ``google/gemma-3-1b-it`` is **gated**: accept its license on the Hugging Face
   Hub and authenticate (``hf auth login`` or the ``HF_TOKEN`` environment
   variable) first, or point ``--weights`` at an ungated mirror of the same
   architecture. A CUDA GPU is required either way.

Prerequisites
-------------
Install Torch-TensorRT with the ExecuTorch extra before running this example::

    pip install -e ".[executorch]"

See https://pytorch.org/executorch/stable/getting-started-setup.html for details.
"""

# %%
# Imports and Setup
# ^^^^^^^^^^^^^^^^^^
#
# ``export_llm``, a reusable helper that traces a decoder over a dynamic
# sequence length, lives under ``tools/llm`` in the Torch-TensorRT repo, so we
# add that directory to ``sys.path``.

import argparse
import sys
import time
from pathlib import Path

import torch
import torch_tensorrt

_start = time.time()


def stamp(phase: str) -> None:
    """Each phase's cost, since export time is the first thing people ask about."""
    print(f"[{time.time() - _start:6.1f}s] {phase}", flush=True)


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "llm"))

MODEL_ID = "google/gemma-3-1b-it"
DEVICE = torch.device("cuda:0")

# The two regimes, matching examples/dynamo/multi_optimization_profiles.py.
MAX_SEQ = 256  # largest prompt the engine must support
PREFILL_SEQ = 128
DECODE_SEQ = 1
DECODE_IDX, PREFILL_IDX = 0, 1

# Gemma-3 shrunk to a few million parameters: the real layer structure, only
# narrower and shallower. ``sliding_window`` keeps the 1B model's 512, which is
# wider than MAX_SEQ, so as in the real model the window never binds over the
# exported range and every layer attends to the whole prefix. Narrowing it below
# MAX_SEQ would make the sliding layers genuinely windowed, and the engine would
# then need ``attn_bias_is_causal=False`` to keep the mask instead of assuming
# plain causality.
MINI_CONFIG = dict(
    vocab_size=2048,
    hidden_size=320,
    intermediate_size=640,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=80,
    max_position_embeddings=512,
    sliding_window=512,
    layer_types=["sliding_attention", "sliding_attention", "full_attention"],
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path",
    default="model_gemma3_multi_profile.pte",
    help="Path to save the .pte file",
)
parser.add_argument(
    "--weights",
    default=None,
    help=(
        "Hugging Face repo to load pretrained weights from, e.g. "
        f"{MODEL_ID}. Omit to export a randomly initialized mini Gemma-3, "
        "which needs no download and exports in about a minute."
    ),
)
args = parser.parse_args()


# %%
# The Exported Method
# ^^^^^^^^^^^^^^^^^^^^
#
# The wrapper fixes the ``.pte``'s method signature to two ``[1, seq]`` inputs
# and one output, and returns only the **last** position's logits -- the row a
# sampler actually reads. That keeps the output shape static at ``[1, vocab]``
# whatever ``seq`` is, so ExecuTorch plans one small buffer instead of one sized
# for ``MAX_SEQ``, and a large device-to-host copy does not end up dominating
# the very latency this example is meant to measure.
class NextTokenLogits(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        out = self.model(input_ids=input_ids, position_ids=position_ids)
        return out.logits[:, -1, :]


# %%
# Build the Model
# ^^^^^^^^^^^^^^^^
#
# Either way the model runs in fp16 with ``use_cache=False`` (this example
# recomputes over the full sequence rather than using a KV cache, which keeps
# the export simple). ``attn_implementation="sdpa"`` makes HuggingFace emit
# ``scaled_dot_product_attention``, which Torch-TensorRT converts to a single
# TensorRT attention layer; no SDPA lowering pass is needed.
def build_model() -> torch.nn.Module:
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    with torch.no_grad():
        if args.weights:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                args.weights,
                use_cache=False,
                attn_implementation="sdpa",
                ignore_mismatched_sizes=True,
            )
        else:
            config = Gemma3TextConfig(
                use_cache=False, attn_implementation="sdpa", **MINI_CONFIG
            )
            model = Gemma3ForCausalLM(config)
        model = model.eval().cuda().to(torch.float16)

    params = sum(p.numel() for p in model.parameters())
    stamp(
        f"model built: Gemma-3 ({args.weights or 'mini, random init'}), {params / 1e6:.1f}M params"
    )
    return model


try:
    model = build_model()
except Exception as e:  # no GPU, or gated/unauthenticated --weights
    print(f"Skipping example: could not build the model ({type(e).__name__}: {e}).")
    print("A CUDA GPU is required. With --weights, accept the model license and")
    print("authenticate (hf auth login / HF_TOKEN), or use an ungated mirror.")
    sys.exit(0)


# %%
# Declaring the Optimization Profiles
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``profiles`` is an ordered list and the list index *is* the optimization
# profile index selected at runtime. There are no profile names. Both model
# inputs are dynamic over ``seq``, so each gets a profiled ``Input`` with
# identical profiles.
#
# The ranges overlap at ``seq == 1``: a decode-sized input is valid under both
# profiles. That overlap is why auto-selection is history-dependent (it keeps
# the loaded profile while it still fits) and why prefill/decode serving should
# pin a profile explicitly rather than rely on auto.
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
# Export Bounds Must Cover Every Profile
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ExecuTorch plans the ``.pte``'s memory from the ``torch.export`` input domain,
# not from the TensorRT profiles, and it plans for the upper bound. So the
# exported ``Dim`` has to span the *union* of all profile ranges -- here
# ``[1, MAX_SEQ]``, the prefill maximum -- or a profile accepting a larger input
# than the plan allows for would overrun its buffer.
from utils import export_llm  # noqa: E402

vocab = model.config.get_text_config().vocab_size
example_ids = torch.randint(
    1, vocab, (1, PREFILL_SEQ), dtype=torch.int64, device=DEVICE
)
with torch.inference_mode():
    exported = export_llm(
        NextTokenLogits(model), example_ids, min_seq_len=1, max_seq_len=MAX_SEQ
    )
stamp("torch.export done")

# %%
# Compile Once
# ^^^^^^^^^^^^^
#
# One export, one compile, one engine holding both profiles. Nothing about the
# profiles is chosen here beyond their bounds; which one runs is a runtime
# decision made per call by the C++ runner.
#
# ``offload_module_to_cpu`` must stay False: it is currently incompatible with
# the multi-profile ``Input(profiles=...)`` path (CPU/CUDA device mismatch).
print("Compiling multi-profile engine (decode + prefill) ...")
with torch.inference_mode():
    trt_gm = torch_tensorrt.dynamo.compile(
        exported,
        arg_inputs=multi_profile_inputs,
        use_fp32_acc=True,
        disable_tf32=True,
        offload_module_to_cpu=False,
        min_block_size=1,
        require_full_compilation=True,
        device=DEVICE,
    )
stamp("TensorRT engine built")


# %%
# Save as ExecuTorch .pte format
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# All profiles live inside the serialized engine, so the TR01 blob format is
# unchanged and the runtime rediscovers count and bounds at load. This step
# scales with the size of the engine, and dominates the export for large models.
position_ids = torch.arange(PREFILL_SEQ, device=DEVICE).unsqueeze(0)
torch_tensorrt.save(
    trt_gm,
    args.model_path,
    output_format="executorch",
    arg_inputs=(example_ids, position_ids),
    retrace=False,
)
stamp("saved .pte")

size_mb = Path(args.model_path).stat().st_size / 1e6
print(f"\nSaved {args.model_path} ({size_mb:.1f} MB) with {len(profiles)} profiles.")
print(f"  profile {DECODE_IDX} (decode):  seq == {DECODE_SEQ}")
print(
    f"  profile {PREFILL_IDX} (prefill): seq in [1, {MAX_SEQ}], tuned at {PREFILL_SEQ}"
)
