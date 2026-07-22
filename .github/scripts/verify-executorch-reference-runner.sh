#!/usr/bin/env bash
set -euo pipefail
set +x

# Verifies the documented end-user flow for the ExecuTorch reference runner:
#
#   1. Build //:libtorchtrt first so bazel-bin/libtorchtrt.tar.gz exists.
#   2. Provide an ExecuTorch source checkout with EXECUTORCH_SOURCE_DIR.
#   3. This script exports a small Torch-TensorRT ExecuTorch .pte model,
#      unpacks libtorchtrt.tar.gz, configures the packaged CMake runner,
#      builds example_executorch_runner, and runs one inference.
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

# torch_tensorrt needs TensorRT/PyTorch shared libraries while exporting the
# .pte model. The native C++ runner below must not rely on libtorch.
original_ld_library_path="${LD_LIBRARY_PATH:-}"
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

missing = []
for name in ("yaml", "torch", "torch_tensorrt", "executorch.exir"):
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        spec = None
    if spec is None:
        missing.append(name)
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

model_path="${verify_root}/model.pte"
"${python_executable}" - "${model_path}" <<'PY'
import importlib.util
import runpy
import sys
from pathlib import Path

model_path = sys.argv[1]
repo_root = Path.cwd()

# Use the installed package for native extensions and the in-tree ExecuTorch
# route for the serializer/backend under test.
import torch_tensorrt  # noqa: F401


def overlay_module(name: str, path: Path) -> None:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


overlay_module(
    "torch_tensorrt.executorch.serialization",
    repo_root / "py/torch_tensorrt/executorch/serialization.py",
)
overlay_module(
    "torch_tensorrt.executorch.backend",
    repo_root / "py/torch_tensorrt/executorch/backend.py",
)

export_script = repo_root / "examples/torchtrt_executorch_example/export_static_shape.py"
sys.argv = [str(export_script), "--model_path", model_path]
runpy.run_path(str(export_script), run_name="__main__")
PY
test -f "${model_path}"

if [[ -n "${TensorRT_ROOT:-}" && -d "${TensorRT_ROOT}/lib" ]]; then
  export LD_LIBRARY_PATH="${TensorRT_ROOT}/lib${original_ld_library_path:+:${original_ld_library_path}}"
else
  export LD_LIBRARY_PATH="${original_ld_library_path}"
fi

tar -xzf "${tarball}" -C "${verify_root}"

# Check the release tarball contract used by the README.
tar_entries="${verify_root}/libtorchtrt_tar_entries.txt"
tar -tf "${tarball}" > "${tar_entries}"

require_tar_entry() {
  local entry="$1"

  if ! grep -qx "${entry}" "${tar_entries}"; then
    echo "libtorchtrt.tar.gz is missing expected entry: ${entry}" >&2
    exit 1
  fi
}

require_tar_entry "torch_tensorrt/src/torch_tensorrt/executorch/CMakeLists.txt"
require_tar_entry "torch_tensorrt/examples/executorch_reference_runner/CMakeLists.txt"
require_tar_entry "torch_tensorrt/lib/libextension_cuda.so"
require_tar_entry "torch_tensorrt/BUILD"

export TORCH_TENSORRT_ROOT="${verify_root}/torch_tensorrt"
export TORCHTRT_EXECUTORCH_SOURCE_DIR="${TORCH_TENSORRT_ROOT}/src/torch_tensorrt/executorch"

# Configure the example exactly as an end user would after unpacking
# libtorchtrt.tar.gz.
cmake_args=(
  -S "${TORCH_TENSORRT_ROOT}/examples/executorch_reference_runner"
  -B "${verify_root}/build-executorch-reference-runner"
  -DEXECUTORCH_SOURCE_DIR="${EXECUTORCH_SOURCE_DIR}"
  -DTORCHTRT_EXECUTORCH_SOURCE_DIR="${TORCHTRT_EXECUTORCH_SOURCE_DIR}"
  -DPYTHON_EXECUTABLE="${python_executable}"
)

if [[ -n "${TensorRT_ROOT:-}" ]]; then
  cmake_args+=(-DTensorRT_ROOT="${TensorRT_ROOT}")
fi

cmake "${cmake_args[@]}"

cmake --build "${verify_root}/build-executorch-reference-runner" \
  --target example_executorch_runner \
  -j"${MAX_JOBS:-$(nproc)}"

runner_log="${verify_root}/my_runner.log"
runner_path="${verify_root}/build-executorch-reference-runner/example_executorch_runner"
if command -v ldd >/dev/null 2>&1 &&
  ldd "${runner_path}" |
    grep -E "libtorch|libtorch_cpu|libtorch_cuda|libc10" >&2; then
  echo "example_executorch_runner links PyTorch/libtorch shared libraries" >&2
  exit 1
fi
if command -v ldd >/dev/null 2>&1 &&
  ! ldd "${runner_path}" | grep -q "libextension_cuda.so"; then
  echo "example_executorch_runner does not link libextension_cuda.so" >&2
  exit 1
fi
if command -v nm >/dev/null 2>&1 &&
  nm --defined-only "${runner_path}" |
    grep -E "getCallerStream|CallerStreamGuard" >&2; then
  echo "example_executorch_runner contains a private caller-stream definition" >&2
  exit 1
fi

"${runner_path}" \
  --model_path="${model_path}" \
  --num_runs=1 2>&1 | tee "${runner_log}"

# The sample model is x + 1, and the reference runner fills inputs with 1.0f,
# so the output sample should contain 2.0000.
grep -q "Inference completed" "${runner_log}"
grep -q "output\\[0\\] shape=" "${runner_log}"
grep -Eq "first [0-9]+ values:.* 2\\.0000" "${runner_log}"
