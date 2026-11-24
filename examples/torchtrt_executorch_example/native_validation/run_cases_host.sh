#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Run independent FP32 fixtures outside Docker with a clean loader environment.
# SDK/library locations are caller-supplied, not baked into the executable.
set -euo pipefail
if [[ $# != 6 ]]; then
  echo "usage: $0 runner bundle-dir fixtures-dir new-results-dir runtime-lib-dir cuda-root" >&2
  exit 2
fi
runner=$(realpath "$1")
bundle=$(realpath "$2")
fixtures=$(realpath "$3")
results=$(realpath -m "$4")
runtime_lib=$(realpath "$5")
cuda_root=$(realpath "$6")
test -x "$runner"
test -f "$bundle/model.pte"
test -d "$cuda_root/targets/aarch64-linux/lib"
mkdir "$results"  # Refuse to overwrite earlier evidence.
library_path="$runtime_lib:$cuda_root/targets/aarch64-linux/lib:$cuda_root/thor/targets/aarch64-linux/lib"
clean=(env -i PATH=/usr/bin:/bin "LD_LIBRARY_PATH=$library_path")
"${clean[@]}" ldd "$runner" | tee "$results/ldd.log"
if grep -E 'not found|libtorch|libc10|libpython' "$results/ldd.log"; then
  echo "FAIL: missing or forbidden dependency" >&2
  exit 1
fi
data=()
if [[ -f "$bundle/aoti_cuda_blob.ptd" ]]; then
  data=("--data=$bundle/aoti_cuda_blob.ptd")
fi
count=0
shopt -s nullglob
for input in "$fixtures"/*/input_0.bin; do
  case_dir=$(dirname "$input")
  name=$(basename "$case_dir")
  "${clean[@]}" "$runner" --model="$bundle/model.pte" "${data[@]}" \
    --case="$case_dir" --output="$results/$name" 2>&1 | tee "$results/$name.log"
  test -s "$results/$name/loaded-maps.txt"
  if grep -E 'libtorch|libc10|libpython|/site-packages/|/venvs/' "$results/$name/loaded-maps.txt"; then
    echo "FAIL: Torch/Python library loaded during inference" >&2
    exit 1
  fi
  count=$((count + 1))
done
test "$count" -gt 0
sha256sum "$runner" "$bundle/model.pte" | tee "$results/artifact-sha256.txt"
echo "PASS: $count independent native fixture cases; no Torch/Python libraries mapped"
