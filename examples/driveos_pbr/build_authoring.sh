#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Run after build_native.sh in the configured DRIVE container.
set -euo pipefail
root=$(realpath "${1:?Usage: bash build_authoring.sh PBR_ROOT}")
ttrt="$root/deps/TensorRT-runtime"
[[ $(git -C "$ttrt" rev-parse HEAD) == 202fba495341f83c4d25b6b2dece95e0242c5369 ]] || {
  echo "Run build_native.sh first to obtain the matched compiler source" >&2; exit 1;
}
[[ -z $(git -C "$ttrt" status --porcelain --untracked-files=no) ]] || {
  echo "Compiler source contains modifications" >&2; exit 1;
}
unset PIP_CONSTRAINT
export PATH="/opt/pbr-venv/bin:$PATH"
export LD_LIBRARY_PATH="$root/bundle/authoring/lib"
mkdir -p "$root/authoring-wheels" "$root/cache/tmp"
export TMPDIR="$root/cache/tmp"
# Build the Python compiler from source. The native delegate was built by CMake.
PYTHON_ONLY=1 python -m pip wheel --no-build-isolation --no-deps \
  --wheel-dir "$root/authoring-wheels" "$ttrt"
python -m pip install --no-deps \
  "$root"/authoring-wheels/torch_tensorrt-2.16.0.dev0+202fba4-*.whl
python -c 'import torch, torch_tensorrt, executorch.exir; print(torch_tensorrt.__version__); print(torch.cuda.get_device_properties(0)); print(torch.ones(1, device="cuda"))'
