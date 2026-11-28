"""
Qwen3 Reranker compiled with Torch-TensorRT

Qwen3-Reranker is a causal LM used for reranking: it takes a formatted
(instruction, query, document) prompt and returns logits for the last
token.  The relevance score is softmax( [logit_no, logit_yes] )[:, 1].

This example:
  1. Loads Qwen/Qwen3-Reranker-0.6B (or a larger variant via --model).
  2. Runs a baseline PyTorch forward pass.
  3. Exports the model with torch.export.export() with dynamic seq_len.
  4. Compiles to TensorRT via torch_tensorrt.dynamo.compile().
  5. Verifies that the last-token logits match between PyTorch and TRT.
  6. (Optional) Benchmarks latency of both backends.

Usage
-----
# Basic run (quality check)
python examples/dynamo/torch_export_qwen3_reranker.py

# With larger model and BF16
python examples/dynamo/torch_export_qwen3_reranker.py --model Qwen/Qwen3-Reranker-4B --precision BF16

# Benchmark mode
python examples/dynamo/torch_export_qwen3_reranker.py --benchmark --iterations 20
"""

import argparse
import sys
import timeit
from contextlib import nullcontext
from pathlib import Path

import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make tools/llm importable so we can reuse export_llm / register_sdpa
_TOOLS_LLM = Path(__file__).resolve().parent.parent.parent / "tools" / "llm"
if str(_TOOLS_LLM) not in sys.path:
    sys.path.insert(0, str(_TOOLS_LLM))

from torchtrt_ext import register_sdpa  # noqa: E402
from utils import export_llm  # noqa: E402

DEVICE = torch.device("cuda:0")

# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_pair(instruction: str, query: str, document: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


def build_inputs(
    tokenizer,
    queries: list[str],
    documents: list[str],
    instruction: str,
    max_length: int = 8192,
) -> dict:
    """Tokenize (query, document) pairs with the reranker prefix/suffix."""
    prefix_ids = tokenizer.encode(_PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(_SUFFIX, add_special_tokens=False)
    body_max = max_length - len(prefix_ids) - len(suffix_ids)

    pairs = [format_pair(instruction, q, d) for q, d in zip(queries, documents)]
    encoded = tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=body_max,
    )
    for i, ids in enumerate(encoded["input_ids"]):
        encoded["input_ids"][i] = prefix_ids + ids + suffix_ids

    batch = tokenizer.pad(
        encoded, padding=True, return_tensors="pt", max_length=max_length
    )
    return {k: v.to(DEVICE) for k, v in batch.items()}


def compute_scores(
    logits: torch.Tensor, token_true_id: int, token_false_id: int
) -> list[float]:
    """Convert last-token logits to yes-probability scores."""
    last = logits[:, -1, :]
    yes_logit = last[:, token_true_id]
    no_logit = last[:, token_false_id]
    stacked = torch.stack([no_logit, yes_logit], dim=1)
    return torch.nn.functional.softmax(stacked, dim=1)[:, 1].tolist()


# ---------------------------------------------------------------------------
# Model wrapper: drop attention_mask, add position_ids
# ---------------------------------------------------------------------------


class RerankerForExport(torch.nn.Module):
    """
    Thin wrapper around the causal-LM so that torch.export sees a clean
    (input_ids, position_ids) signature – matching what export_llm / the
    existing TRT pipeline expects.

    The Qwen3 model internally builds position_ids from the sequence length
    when they are not supplied, but passing them explicitly lets us mark the
    sequence-length dimension as dynamic during export.

    NOTE: We drop attention_mask here because the reranker pairs are
    right-padded during batching.  If you need left-padding (e.g. for
    generation), pass attention_mask explicitly and adjust the export
    dynamic_shapes accordingly.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, position_ids=position_ids)
        return out.logits


# ---------------------------------------------------------------------------
# Export + compile
# ---------------------------------------------------------------------------


def export_reranker(model_wrapper: RerankerForExport, input_ids: torch.Tensor):
    """Export the wrapped reranker with a dynamic sequence-length dimension."""
    max_seq_len = input_ids.shape[1]
    ep = export_llm(model_wrapper, input_ids, max_seq_len=max_seq_len)
    return ep


def compile_torchtrt(
    ep, input_ids: torch.Tensor, precision: str, debug: bool, min_block_size: int
):
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).to(DEVICE)

    if precision == "FP16":
        enabled_precisions = {torch.float32}
        use_fp32_acc = True
    elif precision == "BF16":
        enabled_precisions = {torch.bfloat16}
        use_fp32_acc = False
    else:  # FP32
        enabled_precisions = {torch.float32}
        use_fp32_acc = False

    with torch_tensorrt.logging.debug() if debug else nullcontext():
        trt_model = torch_tensorrt.dynamo.compile(
            ep,
            inputs=[input_ids, position_ids],
            enabled_precisions=enabled_precisions,
            use_explicit_typing=True,
            use_fp32_acc=use_fp32_acc,
            device=DEVICE,
            disable_tf32=True,
            use_python_runtime=True,
            debug=debug,
            offload_module_to_cpu=True,
            min_block_size=min_block_size,
        )
    return trt_model


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------


def benchmark(fn, *args, iterations: int = 10, label: str = "") -> float:
    # Warmup
    fn(*args)
    torch.cuda.synchronize()

    total = 0.0
    for _ in range(iterations):
        t0 = timeit.default_timer()
        fn(*args)
        torch.cuda.synchronize()
        total += timeit.default_timer() - t0

    avg_ms = total / iterations * 1000
    print(f"[{label}] avg latency over {iterations} iters: {avg_ms:.2f} ms")
    return avg_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Compile Qwen3-Reranker with Torch-TensorRT")
    p.add_argument("--model", default="Qwen/Qwen3-Reranker-0.6B", help="HF model name")
    p.add_argument("--precision", default="FP16", choices=["FP16", "BF16", "FP32"])
    p.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max token length for test inputs (keep small for quick iteration)",
    )
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--min_block_size", type=int, default=1)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load tokenizer & model
    # ------------------------------------------------------------------
    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    print(f"  token_true_id (yes) = {token_true_id}, token_false_id (no) = {token_false_id}")

    base_model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            use_cache=False,
            attn_implementation="sdpa",
        )
        .eval()
        .cuda()
    )

    # Register custom SDPA converter (handles Qwen3 like other models in run_llm.py)
    register_sdpa.enable_sdpa_converter(args.model, base_model.config)

    dtype_map = {"FP16": torch.float16, "BF16": torch.bfloat16, "FP32": torch.float32}
    base_model = base_model.to(dtype_map[args.precision])

    # ------------------------------------------------------------------
    # 2. Build test inputs
    # ------------------------------------------------------------------
    instruction = "Given a web search query, retrieve relevant passages that answer the query"
    queries = [
        "What is the capital of China?",
        "How does photosynthesis work?",
    ]
    documents = [
        "The capital of China is Beijing.",
        "Photosynthesis is the process by which plants convert sunlight into glucose.",
    ]

    print("Tokenizing inputs ...")
    inputs = build_inputs(tokenizer, queries, documents, instruction, max_length=args.max_length)
    input_ids = inputs["input_ids"]
    print(f"  input_ids shape: {input_ids.shape}")

    # ------------------------------------------------------------------
    # 3. PyTorch baseline
    # ------------------------------------------------------------------
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).to(DEVICE)
    model_wrapper = RerankerForExport(base_model)

    with torch.inference_mode():
        pyt_logits = model_wrapper(input_ids, position_ids)
    pyt_scores = compute_scores(pyt_logits, token_true_id, token_false_id)
    print("\n--- PyTorch baseline ---")
    for q, d, s in zip(queries, documents, pyt_scores):
        print(f"  Q: {q!r}  D: {d!r}  score={s:.4f}")

    # ------------------------------------------------------------------
    # 4. Export & compile
    # ------------------------------------------------------------------
    print("\nExporting model ...")
    with torch.inference_mode():
        ep = export_reranker(model_wrapper, input_ids)

    print("Compiling with Torch-TensorRT ...")
    with torch.inference_mode():
        trt_model = compile_torchtrt(
            ep, input_ids, args.precision, args.debug, args.min_block_size
        )

    # ------------------------------------------------------------------
    # 5. TRT inference & score comparison
    # ------------------------------------------------------------------
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        trt_logits = trt_model(input_ids, position_ids)
    trt_scores = compute_scores(trt_logits, token_true_id, token_false_id)

    print("\n--- TensorRT ---")
    for q, d, s in zip(queries, documents, trt_scores):
        print(f"  Q: {q!r}  D: {d!r}  score={s:.4f}")

    print("\n--- Score comparison ---")
    for i, (ps, ts) in enumerate(zip(pyt_scores, trt_scores)):
        diff = abs(ps - ts)
        print(f"  pair {i}: PyTorch={ps:.6f}  TRT={ts:.6f}  |diff|={diff:.2e}")

    last_pyt = pyt_logits[:, -1, :].float()
    last_trt = trt_logits[:, -1, :].float()
    max_diff = (last_pyt - last_trt).abs().max().item()
    print(f"\nMax absolute difference in last-token logits: {max_diff:.4e}")

    # ------------------------------------------------------------------
    # 6. (Optional) benchmark
    # ------------------------------------------------------------------
    if args.benchmark:
        print("\n--- Benchmarking ---")

        def pyt_fwd():
            return model_wrapper(input_ids, position_ids)

        def trt_fwd():
            return trt_model(input_ids, position_ids)

        with torch.inference_mode():
            benchmark(pyt_fwd, iterations=args.iterations, label="PyTorch")
            benchmark(trt_fwd, iterations=args.iterations, label="TensorRT")


if __name__ == "__main__":
    main()
