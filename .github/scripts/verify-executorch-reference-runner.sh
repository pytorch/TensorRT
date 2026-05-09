#!/usr/bin/env bash
set -euo pipefail
set +x

# Verifies the documented end-user flow for the ExecuTorch reference runner:
#
#   1. Build //:libtorchtrt first so bazel-bin/libtorchtrt.tar.gz exists.
#   2. Provide an ExecuTorch source checkout with EXECUTORCH_SOURCE_DIR.
#   3. This script exports a small Torch-TensorRT ExecuTorch .pte model,
#      unpacks libtorchtrt.tar.gz, configures the packaged CMake runner,
#      builds my_runner, and runs one inference.
#
# Required:
#   EXECUTORCH_SOURCE_DIR=/path/to/executorch
#
# Optional:
#   TensorRT_ROOT=/path/to/extracted/TensorRT
#     If unset, the script reuses Bazel's fetched TensorRT SDK when available
#     and otherwise downloads the archive pinned in MODULE.bazel.
#   RUNNER_TEMP=/path/to/temp-root
#     Parent directory for the temporary verification workspace.
#   MAX_JOBS=N
#     Parallelism passed to cmake --build.
#   TORCHTRT_TENSORRT_DISTDIR=/path/to/cache
#   TORCHTRT_TENSORRT_EXTRACT_DIR=/path/to/extracted-sdk
#     Override locations used only by the TensorRT download fallback.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_executable="${PYTHON_EXECUTABLE:-}"
if [[ -z "${python_executable}" ]]; then
  python_executable="$(command -v python || true)"
fi
if [[ -z "${python_executable}" ]]; then
  echo "Could not find python on PATH" >&2
  exit 1
fi
export PYTHON_EXECUTABLE="${python_executable}"

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

verify_parent="${RUNNER_TEMP:-/tmp}"
mkdir -p "${verify_parent}"
verify_root="$(mktemp -d "${verify_parent%/}/torchtrt_executorch_readme_verify.XXXXXX")"

# Prefer the TensorRT SDK that Bazel already fetched for //:libtorchtrt. This
# keeps CI from downloading the same SDK twice and keeps CMake linked against
# the same TensorRT version used to build the release tarball.
find_bazel_tensorrt_root() {
  local repo_name="$1"
  local output_base
  local trt_header
  local tensorrt_root

  if ! command -v bazel >/dev/null 2>&1; then
    return 1
  fi

  output_base="$(bazel info output_base 2>/dev/null)" || return 1
  trt_header="$(
    find -L "${output_base}/external" \
      \( -path "*/+*${repo_name}/include/NvInfer.h" -o -path "*/${repo_name}/include/NvInfer.h" \) \
      -print -quit 2>/dev/null || true
  )"

  if [[ -z "${trt_header}" ]]; then
    return 1
  fi

  tensorrt_root="$(dirname "$(dirname "${trt_header}")")"
  echo "Using Bazel TensorRT SDK: ${tensorrt_root}" >&2
  printf '%s\n' "${tensorrt_root}"
}

# The archive repo is architecture-specific in MODULE.bazel.
select_tensorrt_archive_repo() {
  case "$(uname -m)" in
    aarch64|arm64)
      echo "tensorrt_sbsa"
      ;;
    x86_64|amd64)
      echo "tensorrt"
      ;;
    *)
      return 1
      ;;
  esac
}

# Extract the pinned TensorRT URL and strip_prefix from MODULE.bazel so the
# verifier follows the same dependency pin as Bazel without duplicating it here.
read_tensorrt_archive_metadata() {
  local repo_name="$1"

  "${python_executable}" - "${repo_name}" <<'PY'
import re
import sys
from pathlib import Path

repo_name = sys.argv[1]
module_bazel = Path("MODULE.bazel").read_text()

for match in re.finditer(r"http_archive\((?P<body>.*?)\n\)", module_bazel, re.DOTALL):
    body = match.group("body")
    name = re.search(r'name\s*=\s*"([^"]+)"', body)
    if name is None or name.group(1) != repo_name:
        continue

    url = re.search(r'urls\s*=\s*\[\s*"([^"]+)"', body, re.DOTALL)
    strip_prefix = re.search(r'strip_prefix\s*=\s*"([^"]+)"', body)
    if url is None:
        raise SystemExit(f"Could not find urls[] for {repo_name} in MODULE.bazel")

    print(url.group(1), strip_prefix.group(1) if strip_prefix else "")
    break
else:
    raise SystemExit(f'Could not find http_archive(name = "{repo_name}") in MODULE.bazel')
PY
}

# Download TensorRT only when it cannot be found in Bazel's external repo cache
# and TensorRT_ROOT was not provided by the caller.
download_tensorrt_root() {
  local repo_name="$1"
  local tensorrt_url
  local tensorrt_strip_prefix
  local tensorrt_distdir
  local tensorrt_extract_dir
  local tensorrt_archive
  local tensorrt_root

  echo "Downloading TensorRT SDK for ${repo_name}" >&2

  read -r tensorrt_url tensorrt_strip_prefix < <(read_tensorrt_archive_metadata "${repo_name}") || return 1

  tensorrt_distdir="${TORCHTRT_TENSORRT_DISTDIR:-${verify_root}/tensorrt-distdir}"
  tensorrt_extract_dir="${TORCHTRT_TENSORRT_EXTRACT_DIR:-${verify_root}/tensorrt-sdk}"
  tensorrt_archive="${tensorrt_distdir}/$(basename "${tensorrt_url}")"

  mkdir -p "${tensorrt_distdir}" "${tensorrt_extract_dir}"
  if [[ ! -f "${tensorrt_archive}" ]]; then
    curl -fL "${tensorrt_url}" -o "${tensorrt_archive}" || return 1
  fi
  tar -xzf "${tensorrt_archive}" -C "${tensorrt_extract_dir}" || return 1

  if [[ -n "${tensorrt_strip_prefix}" ]]; then
    tensorrt_root="${tensorrt_extract_dir}/${tensorrt_strip_prefix}"
  else
    tensorrt_root="$(find "${tensorrt_extract_dir}" -mindepth 1 -maxdepth 1 -type d -print -quit)"
  fi

  if [[ ! -f "${tensorrt_root}/include/NvInfer.h" ]]; then
    echo "TensorRT_ROOT does not contain include/NvInfer.h: ${tensorrt_root}" >&2
    return 1
  fi

  echo "Using downloaded TensorRT SDK: ${tensorrt_root}" >&2
  printf '%s\n' "${tensorrt_root}"
}

if [[ -z "${TensorRT_ROOT:-}" ]]; then
  tensorrt_repo_name="$(select_tensorrt_archive_repo)" || {
    echo "Unsupported TensorRT archive platform: $(uname -m)" >&2
    exit 1
  }
  TensorRT_ROOT="$(find_bazel_tensorrt_root "${tensorrt_repo_name}" || download_tensorrt_root "${tensorrt_repo_name}" || true)"
  if [[ -n "${TensorRT_ROOT}" ]]; then
    export TensorRT_ROOT
  fi
elif [[ ! -f "${TensorRT_ROOT}/include/NvInfer.h" ]]; then
  echo "TensorRT_ROOT must point to an extracted TensorRT SDK with include/NvInfer.h: ${TensorRT_ROOT}" >&2
  exit 1
fi

# torch_tensorrt and the native runner both need TensorRT/PyTorch shared
# libraries on the runtime path. Set this before importing torch_tensorrt or
# exporting the .pte model.
torch_lib_dir="$("${python_executable}" - <<'PY'
import os
import torch

print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"

if [[ -n "${TensorRT_ROOT:-}" && -d "${TensorRT_ROOT}/lib" ]]; then
  export LD_LIBRARY_PATH="${TensorRT_ROOT}/lib:${torch_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
  export LD_LIBRARY_PATH="${torch_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Fail early if the Python environment cannot export a Torch-TensorRT
# ExecuTorch model or run ExecuTorch's CMake codegen.
if ! "${python_executable}" - <<'PY'
import importlib
import importlib.util

missing = [
    name
    for name in ("yaml", "torch", "torch_tensorrt", "executorch.exir")
    if importlib.util.find_spec(name) is None
]
if missing:
    raise SystemExit(
        "Missing Python package(s) required to export the .pte and build the runner: "
        + ", ".join(missing)
    )

for name in ("yaml", "torch", "torch_tensorrt", "executorch.exir"):
    importlib.import_module(name)
PY
then
  exit 1
fi

export_script="examples/torchtrt_executorch_example/export_static_shape.py"
if [[ ! -f "${export_script}" ]]; then
  echo "Missing ${export_script}; restore the ExecuTorch export example before running this check" >&2
  exit 1
fi

model_path="${verify_root}/model.pte"
"${python_executable}" "${export_script}" --model_path="${model_path}"
if [[ ! -f "${model_path}" ]]; then
  echo "Export did not produce ${model_path}" >&2
  exit 1
fi

tar -xzf "${tarball}" -C "${verify_root}"

# Check the release tarball contract used by the README: libtorchtrt_executorch
# must be a top-level source package, separate from torch_tensorrt/.
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
  CMAKE_PREFIX_PATH="$("${python_executable}" -c "import torch; print(torch.utils.cmake_prefix_path)")"
  export CMAKE_PREFIX_PATH
fi

# Configure the example exactly as an end user would after unpacking
# libtorchtrt.tar.gz.
cmake_args=(
  -S "${TORCHTRT_EXECUTORCH_SOURCE_DIR}/examples/executorch_reference_runner"
  -B "${verify_root}/build-executorch-reference-runner"
  -DEXECUTORCH_SOURCE_DIR="${EXECUTORCH_SOURCE_DIR}"
  -DTORCHTRT_EXECUTORCH_SOURCE_DIR="${TORCHTRT_EXECUTORCH_SOURCE_DIR}"
  -DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}"
  -DPYTHON_EXECUTABLE="${python_executable}"
)

if [[ -n "${TensorRT_ROOT:-}" ]]; then
  cmake_args+=(-DTensorRT_ROOT="${TensorRT_ROOT}")
fi

cmake "${cmake_args[@]}"

cmake --build "${verify_root}/build-executorch-reference-runner" \
  --target my_runner \
  -j"${MAX_JOBS:-$(nproc)}"

runner_log="${verify_root}/my_runner.log"
"${verify_root}/build-executorch-reference-runner/my_runner" \
  --model_path="${model_path}" \
  --num_runs=1 2>&1 | tee "${runner_log}"

# The sample model is x + 1, and my_runner fills inputs with 1.0f, so the
# output sample should contain 2.0000.
grep -q "Inference completed" "${runner_log}"
grep -q "output\\[0\\] shape=" "${runner_log}"
grep -Eq "first [0-9]+ values:.* 2\\.0000" "${runner_log}"
