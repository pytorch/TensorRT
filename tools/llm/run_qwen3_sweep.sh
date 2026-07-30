#!/usr/bin/env bash
# Qwen3 MoE Expert-Parallel: correctness + benchmark sweep.
#
# Run from the HOST (passes the TRT-11.3 lib into the container):
#
#   docker exec \
#       -e LD_PRELOAD=/opt/new-trt/libnvinfer.so.11.3.0 \
#       -e LD_LIBRARY_PATH=/opt/new-trt \
#       trtparallel \
#       bash -c "cd /code/trtparallel/torchTRT_llama_export/TensorRT/tools/llm && bash run_qwen3_ep.sh"
#
# Knobs (env vars, optional):
#   E, K, H, INTER   model dims           (default Qwen3-30B-A3B: 128 / 8 / 2048 / 768)
#   BENCH_TOKENS     token counts to sweep (default "64 512 2048 8192")
#   NPROC            GPU counts to sweep   (default "1 2 4 8"; auto-skips N > visible GPUs)
#   CF               benchmark cap factor  (default 1.25)          -- realistic serving
#   RUN_HF=1         also run REAL-weights correctness (needs ~61GB download)
set -u
cd "$(dirname "$0")"

# --- self-log: tee ALL output into a file IN THIS FOLDER (tools/llm) ---
LOG="${LOG:-run_qwen3_ep.log}"
exec > >(tee "$LOG") 2>&1
echo "[run_qwen3_ep] logging to $(pwd)/$LOG"

E=${E:-128}; K=${K:-8}; H=${H:-2048}; INTER=${INTER:-768}
BENCH_TOKENS=${BENCH_TOKENS:-"64 512 2048 8192"}   # token sweep (overhead-bound -> compute-bound)
NPROC=${NPROC:-"1 2 4 8"}                            # GPU sweep (auto-skips N > available GPUs)
CF=${CF:-1.25}                                       # benchmark capacity factor (realistic)
NODROP=$(( E / K ))                                  # dropless factor = E/k (correctness only)
DIMS="--experts $E --topk $K --hidden $H --inter $INTER"

# how many GPUs are actually visible -> skip larger N (needs multinode)
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "${NGPU:-0}" -lt 1 ] && NGPU=1

echo "======================================================================"
echo " Qwen3 MoE EP sweep  (E=$E k=$K H=$H inter=$INTER)"
echo "   correctness f = E/k = $NODROP (dropless)"
echo "   benchmark   f = $CF ,  T sweep = [$BENCH_TOKENS] ,  N sweep = [$NPROC]"
echo "======================================================================"

# ---- 1. CORRECTNESS: bit-exact vs the dropless reference (f = E/k) --------
# Random weights at real Qwen3 dims -> no download; proves TRT == reference.
echo; echo ">>> [correctness] f=$NODROP (dropless), 2 GPUs  -- expect TRT==reference 5/5"
torchtrtrun --nproc_per_node=2 qwen3_moe_ep_export.py $DIMS \
    --tokens 64 --capacity-factor "$NODROP"

# Optional: REAL pretrained weights (the R5 datapoint). Needs local-disk HF cache.
if [ "${RUN_HF:-0}" = "1" ]; then
  export HF_HOME=${HF_HOME:-/tmp/hf-$USER}
  echo; echo ">>> [correctness] REAL Qwen/Qwen3-30B-A3B weights, f=$NODROP  (HF_HOME=$HF_HOME)"
  torchtrtrun --nproc_per_node=2 qwen3_moe_ep_export.py \
      --hf-model Qwen/Qwen3-30B-A3B --capacity-factor "$NODROP"
fi

# ---- 2. BENCHMARK: realistic f=CF, sweep tokens x GPU count --------------
# fix TOTAL tokens per config so per-rank work = T/N (shows the sharding win).
for T in $BENCH_TOKENS; do
  for NP in $NPROC; do
    if [ "$NP" -gt "$NGPU" ]; then
      echo; echo ">>> [benchmark] SKIP N=$NP, T=$T -- only $NGPU GPU(s) here (needs multinode)"
      continue
    fi
    if [ $(( T % NP )) -ne 0 ]; then
      echo; echo ">>> [benchmark] SKIP N=$NP, T=$T -- T not divisible by N"
      continue
    fi
    echo; echo ">>> [benchmark] f=$CF, ${NP} GPU(s), T=$T  -- read the [5] line"
    torchtrtrun --nproc_per_node="$NP" qwen3_moe_ep_export.py $DIMS \
        --tokens "$T" --capacity-factor "$CF" --benchmark
  done
done