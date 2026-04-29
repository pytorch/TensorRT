#!/usr/bin/env bash
# Performance benchmark sweep across every model class supported by run_hf.py.
# Writes per-model JSON results to bench_results/<family>__<model>.json and
# prints a consolidated speedup summary at the end.
#
# Usage:
#   bash bench_sweep.sh                    # default FP16, 20 iterations
#   ITERS=5 bash bench_sweep.sh            # fewer iterations (faster)
#   PRECISION=BF16 bash bench_sweep.sh
set -u

ITERS="${ITERS:-20}"
PRECISION="${PRECISION:-FP16}"
OPT_LEVEL="${OPT_LEVEL:-}"          # e.g. OPT_LEVEL=4 or OPT_LEVEL=5
INDUCTOR="${INDUCTOR:-}"            # set to 1 to also benchmark torch.compile(inductor)
OUT_DIR="$(pwd)/bench_results"
SUMMARY_LOG="$(pwd)/bench_summary.log"

mkdir -p "$OUT_DIR"
: > "$SUMMARY_LOG"

# Sanitise a model name into a filesystem-safe slug (replace / and : with __)
slug() { echo "$1" | tr '/: ' '__'; }

bench_one() {
    local family="$1"
    local model="$2"
    shift 2

    local name
    name="$(slug "${family}__${model}")"
    local json_out="${OUT_DIR}/${name}.json"

    echo "================================================================" | tee -a "$SUMMARY_LOG"
    echo "[$family] $model" | tee -a "$SUMMARY_LOG"
    local opt_args=()
    [ -n "$OPT_LEVEL" ] && opt_args+=(--optimization-level "$OPT_LEVEL")
    [ -n "$INDUCTOR" ]  && opt_args+=(--inductor)

    echo "  args: --precision $PRECISION --iterations $ITERS${OPT_LEVEL:+ --optimization-level $OPT_LEVEL}${INDUCTOR:+ --inductor} $*" | tee -a "$SUMMARY_LOG"
    echo "----------------------------------------------------------------" | tee -a "$SUMMARY_LOG"

    local start_ts end_ts elapsed rc out
    start_ts=$(date +%s)
    out=$(uv run python run_hf.py \
            --model "$model" \
            --precision "$PRECISION" \
            --iterations "$ITERS" \
            --benchmark \
            --json-out "$json_out" \
            "${opt_args[@]}" \
            "$@" 2>&1)
    rc=$?
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))

    if [ $rc -ne 0 ]; then
        echo "STATUS: ERROR (rc=$rc, ${elapsed}s)" | tee -a "$SUMMARY_LOG"
        echo "--- tail of output ---" | tee -a "$SUMMARY_LOG"
        echo "$out" | tail -15 | tee -a "$SUMMARY_LOG"
    else
        echo "STATUS: OK (${elapsed}s)" | tee -a "$SUMMARY_LOG"
        # Echo the benchmark table rows from stdout.
        echo "$out" | grep -E "pytorch|torch_tensorrt" | tee -a "$SUMMARY_LOG"
    fi
    echo "" | tee -a "$SUMMARY_LOG"
}

# ---- Encoders (text) ----
bench_one encoder bert-base-uncased
bench_one encoder distilbert-base-uncased
bench_one encoder roberta-base
bench_one encoder albert-base-v2
bench_one encoder google/electra-small-discriminator
bench_one encoder sentence-transformers/all-MiniLM-L6-v2

# ---- Encoders (vision) ----
bench_one encoder google/vit-base-patch16-224
bench_one encoder microsoft/resnet-50
bench_one encoder microsoft/swin-tiny-patch4-window7-224
bench_one encoder facebook/convnext-tiny-224

# ---- LLMs (prefill, no cache) ----
bench_one llm gpt2 --isl 64
bench_one llm meta-llama/Llama-3.2-1B-Instruct --isl 128
bench_one llm Qwen/Qwen2.5-0.5B-Instruct --isl 128
bench_one llm microsoft/Phi-3-mini-4k-instruct --isl 128
bench_one llm google/gemma-3-1b-it --isl 64
bench_one llm TinyLlama/TinyLlama-1.1B-Chat-v1.0 --isl 128
bench_one llm facebook/opt-125m --isl 64
bench_one llm EleutherAI/pythia-160m --isl 64

# ---- LLMs (hf_static KV cache) ----
bench_one llm gpt2 --isl 64 --num-tokens 32 --cache hf_static
bench_one llm meta-llama/Llama-3.2-1B-Instruct --isl 128 --num-tokens 64 --cache hf_static

# ---- Seq2seq ----
bench_one seq2seq t5-small --isl 64

# ---- Audio ----
bench_one audio openai/whisper-tiny

# ---- Multimodal ----
bench_one multimodal openai/clip-vit-base-patch32
bench_one multimodal google/siglip-base-patch16-224

# ---- Diffusion ----
bench_one diffusion OFA-Sys/small-stable-diffusion-v0 --num-inference-steps 5

# ---- Video Diffusion ----
bench_one video_diffusion THUDM/CogVideoX-2b --num-frames 9 --num-inference-steps 1
bench_one video_diffusion guoyww/animatediff-motion-adapter-v1-5-2 --num-frames 16 --num-inference-steps 1

# ---- VLMs ----
bench_one vlm HuggingFaceTB/SmolVLM-256M-Instruct
bench_one vlm HuggingFaceTB/SmolVLM-500M-Instruct
bench_one vlm google/paligemma-3b-pt-224
bench_one vlm llava-hf/llava-1.5-7b-hf

# ---- VLAs (Vision-Language-Action / robotics, trust_remote_code) ----
bench_one vla openvla/openvla-7b

# ---- Object Detection / Segmentation ----
bench_one detection facebook/detr-resnet-50
bench_one detection PekingU/rtdetr_r50vd
bench_one detection facebook/sam-vit-base

# ---- Encoders (newly supported types) ----
bench_one encoder facebook/dinov2-base
bench_one encoder nvidia/mit-b0

# ---- Consolidated speedup summary ----
echo "================================================================" | tee -a "$SUMMARY_LOG"
echo "SPEEDUP SUMMARY  (TRT speedup vs PyTorch)" | tee -a "$SUMMARY_LOG"
echo "================================================================" | tee -a "$SUMMARY_LOG"

uv run python - "$OUT_DIR" 2>/dev/null <<'PYEOF' | tee -a "$SUMMARY_LOG"
import json, sys, pathlib

out_dir = pathlib.Path(sys.argv[1])

# Metric key precedence: pick the first one present in the row dict.
_METRIC_KEYS = ["tokens_per_sec", "images_per_sec", "videos_per_sec", "frames_per_sec", "throughput", "rtf"]

_UNIT_LABELS = {
    "tokens_per_sec": "tok/s",
    "images_per_sec": "img/s",
    "videos_per_sec": "vid/s",
    "frames_per_sec": "frames/s",
    "throughput":     "samples/s",
    "rtf":            "RTF↓",
}

# Keys where lower is better (speedup = pt_val / cmp_val).
_LOWER_IS_BETTER = {"rtf"}

def get_metric(row: dict) -> tuple:
    for k in _METRIC_KEYS:
        v = row.get(k)
        if v is not None:
            return k, float(v)
    return None, None

def fmt_val(v: float) -> str:
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.4f}"

def calc_speedup(pt_val: float, cmp_val: float, key: str) -> float:
    if key in _LOWER_IS_BETTER:
        return pt_val / cmp_val   # lower cmp RTF = faster = speedup > 1
    return cmp_val / pt_val       # higher cmp throughput = faster = speedup > 1

def fmt_cell(val: float | None, pt_val: float | None, key: str | None) -> str:
    """Format a value cell; append (Nx) speedup vs PT when both are present."""
    if val is None:
        return "—"
    s = fmt_val(val)
    if pt_val and key:
        spd = calc_speedup(pt_val, val, key)
        s += f" ({spd:.2f}x)"
    return s

def is_pt(backend: str) -> bool:
    b = backend.lower()
    return "pytorch" in b and "inductor" not in b and "torch_tensorrt" not in b

def is_inductor(backend: str) -> bool:
    b = backend.lower()
    return "inductor" in b and "torch_tensorrt" not in b

def is_trt(backend: str) -> bool:
    return "torch_tensorrt" in backend.lower()

rows_by_file: list = []
for f in sorted(out_dir.glob("*.json")):
    try:
        data = json.loads(f.read_text())
        if data:
            rows_by_file.append((f.stem, data))
    except Exception:
        pass

if not rows_by_file:
    print("No JSON result files found in", out_dir)
    sys.exit(0)

# Detect whether any file has inductor rows to decide column layout.
has_inductor = any(
    is_inductor(r.get("backend", ""))
    for _, data in rows_by_file
    for r in data
)

# Column widths: name | PT | [Inductor] | TRT | unit
NW = 44
if has_inductor:
    col_fmt = f"{{:<{NW}}}  {{:>10}}  {{:>16}}  {{:>16}}  {{}}"
    header  = col_fmt.format("model/variant", "PT", "Inductor (spd)", "TRT (spd)", "unit")
    sep     = "-" * (NW + 10 + 16 + 16 + 3 * 2 + 2 + 10)
else:
    col_fmt = f"{{:<{NW}}}  {{:>10}}  {{:>16}}  {{}}"
    header  = col_fmt.format("model/variant", "PT", "TRT (spd)", "unit")
    sep     = "-" * (NW + 10 + 16 + 2 * 2 + 2 + 10)

print(header)
print(sep)

for stem, data in rows_by_file:
    pt_rows  = [r for r in data if is_pt(r.get("backend", ""))]
    ind_rows = [r for r in data if is_inductor(r.get("backend", ""))]
    trt_rows = [r for r in data if is_trt(r.get("backend", ""))]
    if not pt_rows:
        continue

    metric_key, pt_val  = get_metric(pt_rows[0])
    _,          ind_val = get_metric(ind_rows[0]) if ind_rows else (None, None)
    _,          trt_val = get_metric(trt_rows[0]) if trt_rows else (None, None)

    unit    = _UNIT_LABELS.get(metric_key, metric_key or "?")
    pt_str  = fmt_val(pt_val) if pt_val is not None else "—"
    ind_str = fmt_cell(ind_val, pt_val, metric_key)
    trt_str = fmt_cell(trt_val, pt_val, metric_key)

    name = stem[:NW]
    if has_inductor:
        print(col_fmt.format(name, pt_str, ind_str, trt_str, unit))
    else:
        print(col_fmt.format(name, pt_str, trt_str, unit))

print()
PYEOF

echo "Results written to: $OUT_DIR/"
echo "Summary log:        $SUMMARY_LOG"
