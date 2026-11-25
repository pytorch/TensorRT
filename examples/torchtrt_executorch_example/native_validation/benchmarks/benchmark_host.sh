#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Independent process repetitions with identical fixed inputs. Run on DRIVE host.
set -euo pipefail
if [[ $# -lt 6 || $# -gt 9 ]]; then
  echo "usage: $0 runner bundle fixture new-results runtime-lib cuda-root [warmup=20] [iterations=100] [repeats=3]" >&2
  exit 2
fi
[[ ! -e /.dockerenv ]] || { echo "Run on the DRIVE host, outside Docker" >&2; exit 1; }
runner=$(realpath "$1")
bundle=$(realpath "$2")
fixture=$(realpath "$3")
results=$(realpath -m "$4")
runtime_lib=$(realpath "$5")
cuda_root=$(realpath "$6")
warmup=${7:-20}
iterations=${8:-100}
repeats=${9:-3}
for count in "$warmup" "$iterations" "$repeats"; do
  [[ $count =~ ^[1-9][0-9]*$ && ${#count} -le 6 && $count -le 100000 ]] || exit 2
done
test -x "$runner"
test -f "$bundle/model.pte"
test -f "$fixture/input_0.bin"
test -d "$runtime_lib"
test -d "$cuda_root/targets/aarch64-linux/lib"
mkdir "$results"  # A failed mkdir prevents overwriting earlier evidence.
clean=(env -i PATH=/usr/bin:/bin
  "LD_LIBRARY_PATH=$runtime_lib:$cuda_root/targets/aarch64-linux/lib:$cuda_root/thor/targets/aarch64-linux/lib")
"${clean[@]}" ldd "$runner" | tee "$results/ldd.log"
if grep -E 'not found|libtorch|libc10|libpython' "$results/ldd.log"; then exit 1; fi
data=()
if [[ -f "$bundle/aoti_cuda_blob.ptd" ]]; then data=("--data=$bundle/aoti_cuda_blob.ptd"); fi
{
  date -u
  uname -a
  printf 'runner=%s\nbundle=%s\nfixture=%s\ncuda_root=%s\n' "$runner" "$bundle" "$fixture" "$cuda_root"
  printf 'warmup=%s\niterations=%s\nrepeats=%s\n' "$warmup" "$iterations" "$repeats"
  printf 'No clocks, power settings or global page caches changed by this script.\n'
} > "$results/environment.txt"
sha256sum "$runner" "$bundle/model.pte" "$fixture"/*.bin "$fixture"/*.shape > "$results/sha256.txt"
if [[ ${#data[@]} -ne 0 ]]; then sha256sum "$bundle/aoti_cuda_blob.ptd" >> "$results/sha256.txt"; fi
for ((rep=0; rep<repeats; ++rep)); do
  date -u >> "$results/process-times.txt"
  "${clean[@]}" "$runner" --model="$bundle/model.pte" "${data[@]}" \
    --case="$fixture" --output="$results/repeat-$rep" \
    --warmup="$warmup" --iterations="$iterations" 2>&1 | tee "$results/repeat-$rep.log"
  test -s "$results/repeat-$rep/PASS"
done
date -u >> "$results/process-times.txt"
echo "PASS: $repeats independent processes, fixed-input native timing with boundary correctness checks"
