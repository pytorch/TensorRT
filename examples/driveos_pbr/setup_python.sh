#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause
# Run inside the ARM64 NGC 26.06 DRIVE container.
set -euo pipefail
root=$(realpath "${1:?Usage: bash setup_python.sh PBR_ROOT}")
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ $(uname -m) == aarch64 ]] || { echo "This environment requires Linux ARM64" >&2; exit 1; }
[[ ! -L /opt/pbr-venv/lib/python3.12/site-packages/nvidia ]] || {
  echo "Use a fresh container: this venv links to an older bundle" >&2; exit 1;
}
unset PIP_CONSTRAINT
mkdir -p "$root/cache/tmp"
export TMPDIR="$root/cache/tmp"
export PIP_NO_CACHE_DIR=1
# Keep the large environment on the PBR filesystem, not Docker's writable layer.
if [[ -e /opt/pbr-venv && $(realpath /opt/pbr-venv) != "$root/venv" ]]; then
  echo "Use a fresh container: /opt/pbr-venv already points elsewhere" >&2; exit 1
fi
python3 -m venv --system-site-packages "$root/venv"
ln -sfn "$root/venv" /opt/pbr-venv
python=/opt/pbr-venv/bin/python
# Download the exact tested Torch and CUDA packages; verify every wheel hash.
# Keep the remaining build tools (including Triton) from NGC 26.06.
"$python" -m pip install --no-deps --require-hashes \
  --index-url https://pypi.org/simple -r "$source_dir/authoring-requirements.txt"
"$python" -m pip install --no-index --no-deps \
  --find-links="$root/bundle/authoring/wheels" \
  'torchvision==0.30.0.dev20261007+cu132' \
  'executorch==1.6.0.dev20261009+cu132' \
  'ruamel.yaml==0.19.1' 'flatbuffers==25.12.19'
"$python" -m pip install --no-index \
  --find-links="$root/bundle/authoring/wheels" -r "$source_dir/requirements.txt"
export LD_LIBRARY_PATH="$root/bundle/authoring/lib"
"$python" -m pip freeze > "$root/python-environment.txt"
"$python" -c 'import torch, torchvision, executorch.exir; print(torch.__version__); print(torch.cuda.get_device_properties(0)); print(torch.ones(1, device="cuda"))'
