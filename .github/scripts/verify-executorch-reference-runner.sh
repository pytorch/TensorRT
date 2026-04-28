#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

: "${EXECUTORCH_SOURCE_DIR:?Set EXECUTORCH_SOURCE_DIR to an ExecuTorch source checkout}"

if [[ ! -f "${EXECUTORCH_SOURCE_DIR}/CMakeLists.txt" ]]; then
  echo "EXECUTORCH_SOURCE_DIR must point to an ExecuTorch source checkout: ${EXECUTORCH_SOURCE_DIR}" >&2
  exit 1
fi

tarball="${repo_root}/bazel-bin/libtorchtrt.tar.gz"
if [[ ! -f "${tarball}" ]]; then
  echo "Missing ${tarball}; build //:libtorchtrt before running this check" >&2
  exit 1
fi

verify_root="${TORCHTRT_EXECUTORCH_README_VERIFY_DIR:-${RUNNER_TEMP:-/tmp}/torchtrt_executorch_readme_verify}"
rm -rf "${verify_root}"
mkdir -p "${verify_root}"

python - <<'PY'
import importlib
import importlib.util

missing = [
    name
    for name in ("torch", "torch_tensorrt", "executorch.exir")
    if importlib.util.find_spec(name) is None
]
if missing:
    raise SystemExit(
        "Missing Python package(s) required to export the .pte: "
        + ", ".join(missing)
    )

for name in ("torch", "torch_tensorrt", "executorch.exir"):
    importlib.import_module(name)
PY

model_path="${verify_root}/model.pte"
python examples/torchtrt_executorch_example/export_static_shape.py --model_path="${model_path}"
test -f "${model_path}"

tar -xzf "${tarball}" -C "${verify_root}"

tar_entries="${verify_root}/libtorchtrt_tar_entries.txt"
tar -tf "${tarball}" > "${tar_entries}"
grep -qx "libtorchtrt_executorch/CMakeLists.txt" "${tar_entries}"
grep -qx "libtorchtrt_executorch/examples/executorch_reference_runner/CMakeLists.txt" "${tar_entries}"
grep -qx "torch_tensorrt/BUILD" "${tar_entries}"

if grep -q "^torch_tensorrt/libtorchtrt_executorch/" "${tar_entries}"; then
  echo "libtorchtrt_executorch must be a top-level tar entry, not nested under torch_tensorrt/" >&2
  exit 1
fi

export TORCHTRT_EXECUTORCH_SOURCE_DIR="${verify_root}/libtorchtrt_executorch"

if [[ -z "${CMAKE_PREFIX_PATH:-}" ]]; then
  CMAKE_PREFIX_PATH="$(python -c "import torch; print(torch.utils.cmake_prefix_path)")"
  export CMAKE_PREFIX_PATH
fi

torch_lib_dir="$(python - <<'PY'
import os
import torch

print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"
export LD_LIBRARY_PATH="${torch_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cmake -S "${TORCHTRT_EXECUTORCH_SOURCE_DIR}/examples/executorch_reference_runner" \
  -B "${verify_root}/build-executorch-reference-runner" \
  -DEXECUTORCH_SOURCE_DIR="${EXECUTORCH_SOURCE_DIR}" \
  -DTORCHTRT_EXECUTORCH_SOURCE_DIR="${TORCHTRT_EXECUTORCH_SOURCE_DIR}" \
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}"

cmake --build "${verify_root}/build-executorch-reference-runner" \
  --target my_runner \
  -j"${MAX_JOBS:-$(nproc)}"

runner_log="${verify_root}/my_runner.log"
"${verify_root}/build-executorch-reference-runner/my_runner" \
  --model_path="${model_path}" \
  --num_runs=1 2>&1 | tee "${runner_log}"

grep -q "Inference completed" "${runner_log}"
grep -q "output\\[0\\] shape=" "${runner_log}"
