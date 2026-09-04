#!/usr/bin/env bash
# Build the ExecuTorch runtime wheel with CUDA architectures its CUDA shims support.

set -euo pipefail

# ExecuTorch's CUDA shims require sm_61 or later. Keep this immediately before the
# wheel build so every caller—CI and both release architectures—uses the same list.
TORCH_CUDA_ARCH_LIST="$(python .github/scripts/filter-executorch-cuda-arches.py "${TORCH_CUDA_ARCH_LIST:-}")"
export TORCH_CUDA_ARCH_LIST
echo "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

python -m pip wheel --no-build-isolation --no-deps --wheel-dir dist \
  py/torch-tensorrt-executorch-runtime
