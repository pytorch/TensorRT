#!/usr/bin/env bash

# Prepare an ExecuTorch source checkout for CI builds that need Bazel @executorch.
#
# This script is intended to be sourced from CI steps so the exported
# EXECUTORCH_ROOT / EXECUTORCH_SOURCE_DIR values are available to commands in
# the same shell. It also writes the variables to GitHub and build env files
# when those files are available.

_torchtrt_executorch_normalize_ref() {
  case "$1" in
    ""|"latest"|"latest-main"|"latest_main"|"latest main")
      printf '%s\n' "main"
      ;;
    *)
      printf '%s\n' "$1"
      ;;
  esac
}

_torchtrt_executorch_python() {
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
  elif command -v python >/dev/null 2>&1; then
    command -v python
  else
    return 1
  fi
}

_torchtrt_executorch_ensure_cmake() {
  if command -v cmake >/dev/null 2>&1; then
    return 0
  fi

  local python_executable
  python_executable="$(_torchtrt_executorch_python)" || {
    echo "cmake is not installed and no Python interpreter is available to install it" >&2
    return 1
  }

  "${python_executable}" -m pip install --progress-bar=off cmake ninja || return 1
  export PATH="$("${python_executable}" -m site --user-base)/bin:${PATH}"

  if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake is still unavailable after pip install" >&2
    return 1
  fi
}

_torchtrt_executorch_find_core_archive() {
  local executorch_root="$1"
  local candidate
  for candidate in \
    "${executorch_root}/cmake-out/libexecutorch_core.a" \
    "${executorch_root}/cmake-out/lib/libexecutorch_core.a" \
    "${executorch_root}/cmake-out/executorch/libexecutorch_core.a" \
    "${executorch_root}/lib/libexecutorch_core.a" \
    "${executorch_root}/lib64/libexecutorch_core.a" \
    "${executorch_root}/libexecutorch_core.a"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  find "${executorch_root}/cmake-out" -name libexecutorch_core.a -type f -print -quit 2>/dev/null || true
}

_torchtrt_executorch_export_env() {
  local executorch_root="$1"
  export EXECUTORCH_ROOT="${executorch_root}"
  export EXECUTORCH_SOURCE_DIR="${executorch_root}"

  if [[ -n "${GITHUB_ENV:-}" ]]; then
    {
      echo "EXECUTORCH_ROOT=${EXECUTORCH_ROOT}"
      echo "EXECUTORCH_SOURCE_DIR=${EXECUTORCH_SOURCE_DIR}"
    } >> "${GITHUB_ENV}"
  fi

  if [[ -n "${BUILD_ENV_FILE:-}" ]]; then
    {
      echo "export EXECUTORCH_ROOT=\"${EXECUTORCH_ROOT}\""
      echo "export EXECUTORCH_SOURCE_DIR=\"${EXECUTORCH_SOURCE_DIR}\""
    } >> "${BUILD_ENV_FILE}"
  fi
}

_torchtrt_setup_executorch_ci() {
  local executorch_ref
  local executorch_root
  local executorch_parent
  local core_archive
  local max_jobs

  executorch_ref="$(_torchtrt_executorch_normalize_ref "${EXECUTORCH_REF:-main}")"
  executorch_root="${EXECUTORCH_ROOT:-${EXECUTORCH_SOURCE_DIR:-}}"

  if [[ -z "${executorch_root}" ]]; then
    executorch_parent="${RUNNER_TEMP:-/tmp}"
    executorch_root="${executorch_parent%/}/executorch"
  fi

  if [[ ! -f "${executorch_root}/CMakeLists.txt" ]]; then
    if [[ -e "${executorch_root}" ]]; then
      echo "EXECUTORCH_ROOT exists but is not an ExecuTorch source checkout: ${executorch_root}" >&2
      return 1
    fi

    mkdir -p "$(dirname "${executorch_root}")" || return 1
    git clone --depth 1 --branch "${executorch_ref}" --recurse-submodules --shallow-submodules \
      https://github.com/pytorch/executorch.git "${executorch_root}" || return 1
  fi

  core_archive="$(_torchtrt_executorch_find_core_archive "${executorch_root}")"
  if [[ -z "${core_archive}" ]]; then
    _torchtrt_executorch_ensure_cmake || return 1

    cmake -S "${executorch_root}" -B "${executorch_root}/cmake-out" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DBUILD_TESTING=OFF \
      -DEXECUTORCH_BUILD_PYBIND=OFF \
      -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
      -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
      -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
      -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON || return 1

    max_jobs="${MAX_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)}"
    cmake --build "${executorch_root}/cmake-out" --target executorch_core -j "${max_jobs}" || return 1

    core_archive="$(_torchtrt_executorch_find_core_archive "${executorch_root}")"
    if [[ -z "${core_archive}" ]]; then
      echo "ExecuTorch build completed but libexecutorch_core.a was not found under ${executorch_root}" >&2
      return 1
    fi
  fi

  if [[ "${core_archive}" != "${executorch_root}/libexecutorch_core.a" ]]; then
    ln -sf "${core_archive}" "${executorch_root}/libexecutorch_core.a" || return 1
  fi

  _torchtrt_executorch_export_env "${executorch_root}"
  echo "Using EXECUTORCH_ROOT=${EXECUTORCH_ROOT}"
}

_torchtrt_setup_executorch_ci "$@"
