#!/usr/bin/env bash
set -euo pipefail
set +x

# Verifies the documented end-user flow for the ExecuTorch reference runner:
#
#   1. Build //:libtorchtrt first so bazel-bin/libtorchtrt.tar.gz exists.
#   2. Provide an ExecuTorch source checkout with EXECUTORCH_SOURCE_DIR.
#   3. Provide a Torch-TensorRT ExecuTorch .pte model.
#   4. This script unpacks libtorchtrt.tar.gz, configures and builds the
#      packaged CMake runner, and runs one inference.
#
# Required:
#   First argument: path to an existing .pte model.
#   EXECUTORCH_SOURCE_DIR=/path/to/executorch
#
# Optional second argument: path to a caller-owned KV-cache decode .pte (see
#   examples/torchtrt_executorch_example/export_kv_cache_decode.py). When given,
#   kv_cache_decode_check is built and run against it as well.
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

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 PATH_TO_MODEL.pte [PATH_TO_KV_CACHE_DECODE.pte]" >&2
  exit 1
fi
model_path="$1"
if [[ ! -f "${model_path}" ]]; then
  echo "ExecuTorch model not found: ${model_path}" >&2
  exit 1
fi
kv_model_path="${2:-}"
if [[ -n "${kv_model_path}" && ! -f "${kv_model_path}" ]]; then
  echo "KV-cache decode model not found: ${kv_model_path}" >&2
  exit 1
fi

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
  case "${tensorrt_archive}" in
    *.tar.zst)
      tar --zstd -xf "${tensorrt_archive}" -C "${tensorrt_extract_dir}" || return 1
      ;;
    *.tar.gz | *.tgz)
      tar -xzf "${tensorrt_archive}" -C "${tensorrt_extract_dir}" || return 1
      ;;
    *)
      echo "Unsupported TensorRT archive format: ${tensorrt_archive}" >&2
      return 1
      ;;
  esac

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

# Fail early if the Python environment cannot run ExecuTorch's CMake codegen.
if ! "${python_executable}" - <<'PY'
import importlib
import importlib.util

missing = []
for name in ("yaml", "torch", "executorch.exir"):
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        spec = None
    if spec is None:
        missing.append(name)
if missing:
    raise SystemExit(
        "Missing Python package(s) required to build the runner: "
        + ", ".join(missing)
    )

for name in ("yaml", "torch", "executorch.exir"):
    importlib.import_module(name)
PY
then
  exit 1
fi

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
require_tar_entry "torch_tensorrt/bin/example_executorch_runner"
require_tar_entry "torch_tensorrt/lib/libextension_cuda.so"
require_tar_entry "torch_tensorrt/examples/executorch_reference_runner/kv_cache_decode_check.cpp"
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

build_targets=(example_executorch_runner)
if [[ -n "${kv_model_path}" ]]; then
  build_targets+=(kv_cache_decode_check)
fi

cmake --build "${verify_root}/build-executorch-reference-runner" \
  --target "${build_targets[@]}" \
  -j"${MAX_JOBS:-$(nproc)}"

runner_log="${verify_root}/my_runner.log"
runner_path="${verify_root}/build-executorch-reference-runner/example_executorch_runner"

# Symbol/linkage inspection tools are mandatory on this Linux gate: silently
# skipping them would let a broken single-TLS layout pass unnoticed.
for _tool in ldd readelf nm; do
  if ! command -v "${_tool}" >/dev/null 2>&1; then
    echo "Required tool '${_tool}' not found; cannot verify caller-stream linkage" >&2
    exit 1
  fi
done

# Capture nm output instead of piping it into `grep -q`. grep exits at its first
# match and closes the pipe, so nm dies of SIGPIPE and pipefail reports 141, which
# reads as a false verdict in either direction.
nm_matches() {
  local _pattern="$1"
  shift
  local _symbols
  _symbols="$(nm "$@" 2>/dev/null)"
  grep -qE "${_pattern}" <<<"${_symbols}"
}

# The runner must not pull in libtorch: this native path is libtorch-free.
if ldd "${runner_path}" |
    grep -E "libtorch|libtorch_cpu|libtorch_cuda|libc10" >&2; then
  echo "example_executorch_runner links PyTorch/libtorch shared libraries" >&2
  exit 1
fi

# The runner must declare a real DT_NEEDED dependency on libextension_cuda.so
# (an ldd filename match alone would also accept a "=> not found" line).
if ! readelf -d "${runner_path}" |
    grep -E "\(NEEDED\).*libextension_cuda\.so" >&2; then
  echo "example_executorch_runner has no DT_NEEDED entry for libextension_cuda.so" >&2
  exit 1
fi

# ...and that dependency must actually resolve at load time.
if ldd "${runner_path}" | grep -E "libextension_cuda\.so.*=>.*not found" >&2; then
  echo "example_executorch_runner cannot resolve libextension_cuda.so at runtime" >&2
  exit 1
fi

# The runner must import the caller-stream API from the shared library rather
# than define it privately. A private definition means a second copy of the
# thread-local, which silently breaks the cross-backend handshake. Assert the
# import (in .dynsym) rather than the absence of a definition: absence-of-symbol
# checks read .symtab, which is stripped from release binaries and would make
# the assertion pass vacuously. A private definition would satisfy the reference
# at link time and leave no import here.
for _symbol in getCallerStream CallerStreamGuard; do
  if ! nm_matches "${_symbol}" -D --undefined-only "${runner_path}"; then
    echo "example_executorch_runner does not import ${_symbol} from libextension_cuda.so" >&2
    exit 1
  fi
done

# Validate the .so the runner ACTUALLY loads (resolved via ldd), not just a
# packaged copy. The runner is CMake-built and may link the CMake-built
# extension_cuda; whichever .so the loader binds to must export the accessor and
# must be the sole definer the runner sees.
loaded_extension_cuda="$(
  ldd "${runner_path}" 2>/dev/null |
    sed -n 's/.*libextension_cuda\.so[^ ]* => \([^ ]*\).*/\1/p'
)"
loaded_extension_cuda="${loaded_extension_cuda%%$'\n'*}"
if [[ -z "${loaded_extension_cuda}" || ! -f "${loaded_extension_cuda}" ]]; then
  echo "Could not resolve the libextension_cuda.so the runner loads" >&2
  exit 1
fi
if ! nm_matches "getCallerStream" --defined-only --dynamic "${loaded_extension_cuda}"; then
  echo "Loaded ${loaded_extension_cuda} does not export getCallerStream" >&2
  exit 1
fi

# Packaging integrity (independent of the CMake runner): the Bazel-packaged .so
# must exist and export the accessor, and no other packaged ELF may define the
# caller-stream symbols -- a second definition would reintroduce a duplicate
# thread-local in the shipped artifact.
packaged_runner="${TORCH_TENSORRT_ROOT}/bin/example_executorch_runner"
packaged_extension_cuda="${TORCH_TENSORRT_ROOT}/lib/libextension_cuda.so"
if [[ ! -x "${packaged_runner}" ]]; then
  echo "Packaged example_executorch_runner missing or not executable: ${packaged_runner}" >&2
  exit 1
fi
if [[ ! -f "${packaged_extension_cuda}" ]]; then
  echo "Packaged libextension_cuda.so missing at ${packaged_extension_cuda}" >&2
  exit 1
fi
if ! nm_matches "getCallerStream" --defined-only --dynamic "${packaged_extension_cuda}"; then
  echo "Packaged libextension_cuda.so does not export getCallerStream" >&2
  exit 1
fi
if ! readelf -d "${packaged_runner}" |
    grep -E "\(NEEDED\).*libextension_cuda\.so" >&2; then
  echo "Packaged runner has no DT_NEEDED entry for libextension_cuda.so" >&2
  exit 1
fi
if ldd "${packaged_runner}" | grep -E "libextension_cuda\.so.*=>.*not found" >&2; then
  echo "Packaged runner cannot resolve libextension_cuda.so" >&2
  exit 1
fi
for _symbol in getCallerStream CallerStreamGuard; do
  if ! nm_matches "${_symbol}" -D --undefined-only "${packaged_runner}"; then
    echo "Packaged runner does not import ${_symbol} from libextension_cuda.so" >&2
    exit 1
  fi
done

# No other packaged ELF may define the caller-stream symbols: a second
# definition would reintroduce a duplicate thread-local.
extra_defs="$(
  find "${TORCH_TENSORRT_ROOT}/lib" -maxdepth 1 -type f -name '*.so*' \
    ! -name 'libextension_cuda.so' -print0 2>/dev/null |
    while IFS= read -r -d '' _so; do
      if nm_matches "getCallerStream|CallerStreamGuard" --defined-only --dynamic "${_so}"; then
        echo "${_so}"
      fi
    done
)"
if [[ -n "${extra_defs}" ]]; then
  echo "Unexpected caller-stream definitions outside libextension_cuda.so:" >&2
  echo "${extra_defs}" >&2
  exit 1
fi

# ET_CHECK_MSG and ET_LOG hand their text to the core archive's vlogf, so an
# archive compiled with logging off turns every runtime failure in the packaged
# runner into a bare exit code with no output at all. Prove the runner can still
# say why it failed, on a path that needs neither a GPU nor a valid model.
diagnostic_log="${verify_root}/packaged_runner_diagnostic.log"
if "${packaged_runner}" \
  --model_path="${verify_root}/no-such-model.pte" >"${diagnostic_log}" 2>&1; then
  echo "Packaged runner exited 0 on a missing model file" >&2
  exit 1
fi
if ! grep -q "FileDataLoader::from" "${diagnostic_log}"; then
  echo "Packaged runner gave no diagnostic for a missing model file." >&2
  echo "ET_LOG_ENABLED likely disagrees between libexecutorch_core.a and the" >&2
  echo "Bazel targets that call into it. Output was:" >&2
  cat "${diagnostic_log}" >&2
  exit 1
fi

"${runner_path}" \
  --model_path="${model_path}" \
  --num_runs=1 2>&1 | tee "${runner_log}"
packaged_runner_log="${verify_root}/packaged_runner.log"
"${packaged_runner}" \
  --model_path="${model_path}" \
  --num_runs=1 2>&1 | tee "${packaged_runner_log}"

# The sample model is x + 1 on a (2,3,4,4) input and both runners fill inputs with
# 1.0f, so the shape is exactly [2,3,4,4] and every printed value is exactly 2.0000.
# Assert both precisely. Matching only "shape=" accepts any shape, and matching one
# 2.0000 anywhere on the values line accepts a line of wrong numbers that happens to
# contain one right one, so neither catches a stream-ordering regression returning
# stale or partial output. Both lines come from fprintf in the runner, so these
# assertions hold whatever ET_LOG_ENABLED is set to.
for _log in "${runner_log}" "${packaged_runner_log}"; do
  if ! grep -q 'output\[0\] shape=\[2,3,4,4\]' "${_log}"; then
    echo "Unexpected output shape in ${_log}:" >&2
    grep 'output\[0\] shape=' "${_log}" >&2 || echo "  no shape line at all" >&2
    exit 1
  fi

  _values="$(sed -n 's/.*first [0-9]* values://p' "${_log}")"
  _values="${_values%%$'\n'*}"
  if [[ -z "${_values}" ]]; then
    echo "No output values line in ${_log}" >&2
    exit 1
  fi
  for _value in ${_values}; do
    if [[ "${_value}" != "2.0000" ]]; then
      echo "Unexpected output value '${_value}' in ${_log}: ${_values}" >&2
      exit 1
    fi
  done
done

if [[ -n "${kv_model_path}" ]]; then
  # kv_cache_decode_check exits non-zero when a decode step does not observe the KV
  # the previous step wrote; the grep additionally pins the assertion itself, so
  # weakening the check inside the binary cannot quietly turn this into a no-op.
  kv_check_log="${verify_root}/kv_cache_decode_check.log"
  kv_check_path="${verify_root}/build-executorch-reference-runner/kv_cache_decode_check"
  if command -v ldd >/dev/null 2>&1 &&
    ldd "${kv_check_path}" |
      grep -E "libtorch|libtorch_cpu|libtorch_cuda|libc10" >&2; then
    echo "kv_cache_decode_check links PyTorch/libtorch shared libraries" >&2
    exit 1
  fi

  "${kv_check_path}" --model_path="${kv_model_path}" 2>&1 | tee "${kv_check_log}"
  grep -q "PASS: decode at pos=1 observed the KV written at pos=0" "${kv_check_log}"
fi
