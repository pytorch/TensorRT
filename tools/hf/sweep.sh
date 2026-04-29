#!/usr/bin/env bash
# Accuracy + smoke sweep across every model class supported by run_hf.py.
# Writes machine-readable results to sweep_results.log.
#
# Each entry runs `run_hf.py --accuracy` with FP16 + a slightly relaxed
# atol of 0.05 so allclose matches the cosine-similarity verdict on FP16.
# A run is recorded as PASS/FAIL/ERROR and the full stdout/stderr captured.
#
# Usage: bash sweep.sh
set -u

LOG="$(pwd)/sweep_results.log"
: > "$LOG"

ATOL="${ATOL:-0.05}"
RTOL="${RTOL:-0.01}"
COS_MIN="${COS_MIN:-0.99}"

run_one() {
    local family="$1"
    local model="$2"
    shift 2

    echo "================================================================" | tee -a "$LOG"
    echo "[$family] $model" | tee -a "$LOG"
    echo "  args: $*" | tee -a "$LOG"
    echo "----------------------------------------------------------------" | tee -a "$LOG"

    local start_ts=$(date +%s)
    out=$(uv run python run_hf.py --model "$model" \
            --accuracy --accuracy-atol "$ATOL" --accuracy-rtol "$RTOL" \
            --accuracy-cos-sim-min "$COS_MIN" "$@" 2>&1)
    local rc=$?
    local end_ts=$(date +%s)
    local elapsed=$((end_ts - start_ts))

    # Distill the result line into a single status.
    if [ $rc -ne 0 ]; then
        verdict="ERROR (rc=$rc)"
    elif echo "$out" | grep -q "Overall: PASS"; then
        verdict="PASS"
    elif echo "$out" | grep -q "Overall: FAIL"; then
        verdict="FAIL"
    else
        verdict="UNKNOWN"
    fi

    echo "VERDICT: $verdict   ($elapsed s)" | tee -a "$LOG"

    # Capture cos_sim / max_abs lines if present (the rows of the table).
    echo "$out" | grep -E "out\[[0-9]+\]" | tee -a "$LOG"

    # On error, dump the last 15 lines for debugging.
    if [ "$verdict" != "PASS" ]; then
        echo "--- tail of output ---" | tee -a "$LOG"
        echo "$out" | tail -15 | tee -a "$LOG"
    fi
    echo "" | tee -a "$LOG"
}

# ---- Encoders (text) ----
run_one encoder bert-base-uncased
run_one encoder distilbert-base-uncased
run_one encoder roberta-base
run_one encoder albert-base-v2
run_one encoder google/electra-small-discriminator
run_one encoder sentence-transformers/all-MiniLM-L6-v2

# ---- Encoders (vision) ----
run_one encoder google/vit-base-patch16-224
run_one encoder microsoft/resnet-50
run_one encoder microsoft/swin-tiny-patch4-window7-224
run_one encoder facebook/convnext-tiny-224

# ---- LLMs (prefill, no cache) ----
run_one llm gpt2 --isl 64
run_one llm meta-llama/Llama-3.2-1B-Instruct --isl 128
run_one llm Qwen/Qwen2.5-0.5B-Instruct --isl 128
run_one llm microsoft/Phi-3-mini-4k-instruct --isl 128
run_one llm google/gemma-3-1b-it --isl 64
run_one llm TinyLlama/TinyLlama-1.1B-Chat-v1.0 --isl 128
run_one llm facebook/opt-125m --isl 64
run_one llm EleutherAI/pythia-160m --isl 64

# ---- LLMs (hf_static KV cache, export mode) ----
run_one llm gpt2 --isl 64 --num-tokens 32 --cache hf_static
run_one llm meta-llama/Llama-3.2-1B-Instruct --isl 128 --num-tokens 64 --cache hf_static

# ---- Seq2seq ----
run_one seq2seq t5-small --isl 64

# ---- Audio ----
run_one audio openai/whisper-tiny

# ---- Multimodal ----
run_one multimodal openai/clip-vit-base-patch32
run_one multimodal google/siglip-base-patch16-224

# ---- Diffusion ----
run_one diffusion OFA-Sys/small-stable-diffusion-v0 --num-inference-steps 5

# ---- Video Diffusion ----
run_one video_diffusion THUDM/CogVideoX-2b --num-frames 9 --num-inference-steps 1
run_one video_diffusion guoyww/animatediff-motion-adapter-v1-5-2 --num-frames 16 --num-inference-steps 1

echo "================================================================" | tee -a "$LOG"
echo "SUMMARY" | tee -a "$LOG"
echo "================================================================" | tee -a "$LOG"
grep -E "^\[(encoder|llm|seq2seq|audio|multimodal|diffusion|video_diffusion)\]|^VERDICT" "$LOG" \
    | paste - - \
    | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Log written to: $LOG"
