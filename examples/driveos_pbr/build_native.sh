#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Run inside the ARM64 DRIVE authoring container after preparing /opt/pbr-venv.
set -euo pipefail
root=$(realpath "${1:?Usage: bash build_native.sh PBR_ROOT}")
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cuda_root=${CUDAToolkit_ROOT:-/usr/local/cuda-13.3}
export PATH="/opt/pbr-venv/bin:$PATH"
export LD_LIBRARY_PATH="$root/bundle/authoring/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ttrt_commit=202fba495341f83c4d25b6b2dece95e0242c5369
et_commit=a9665e25097b324a7116efa2bd6602287c9c064f
mkdir -p "$root/deps"
checkout() {
  local directory=$1 url=$2 commit=$3
  if [[ ! -d "$directory/.git" ]]; then
    git init "$directory"
    git -C "$directory" remote add origin "$url"
    git -C "$directory" fetch --depth=1 origin "$commit"
    git -C "$directory" checkout --detach FETCH_HEAD
  fi
  [[ $(git -C "$directory" rev-parse HEAD) == "$commit" ]] || {
    echo "Unexpected revision in $directory; use a fresh PBR_ROOT" >&2; exit 1;
  }
  [[ -z $(git -C "$directory" status --porcelain --untracked-files=no) ]] || {
    echo "Modified source in $directory; use a fresh PBR_ROOT" >&2; exit 1;
  }
}
checkout "$root/deps/TensorRT-runtime" https://github.com/pytorch/TensorRT.git "$ttrt_commit"
checkout "$root/deps/executorch" https://github.com/pytorch/executorch.git "$et_commit"
git -C "$root/deps/executorch" submodule update --init --recursive --depth=1 -- \
  third-party/flatbuffers third-party/flatcc third-party/gflags \
  third-party/json third-party/pocketfft backends/xnnpack/third-party/FXdiv \
  backends/xnnpack/third-party/cpuinfo backends/xnnpack/third-party/pthreadpool
cmake -S "$source_dir" -B "$root/build-native" \
  -DPBR_BUILD_RUNTIME_FROM_SOURCE=ON \
  -DTORCHTRT_SOURCE_DIR="$root/deps/TensorRT-runtime" \
  -DEXECUTORCH_SOURCE_DIR="$root/deps/executorch" \
  -DCUDAToolkit_ROOT="$cuda_root" \
  -DCMAKE_CUDA_COMPILER="$cuda_root/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=110 -DCMAKE_BUILD_TYPE=Release
cmake --build "$root/build-native" --target benchmark_trt benchmark_et -j2
cmake --install "$root/build-native" --prefix "$root/native"
printf 'Torch-TensorRT %s\nExecuTorch %s\n' "$ttrt_commit" "$et_commit" \
  > "$root/native/source-revisions.txt"
echo "Built native runtime, TensorRT delegate and applications in $root/native"
