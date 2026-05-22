#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${CONDA_ENV:-}" ]]; then
  echo "::error::CONDA_ENV is not set"
  exit 1
fi

if [[ -z "${GITHUB_ENV:-}" ]]; then
  echo "::error::GITHUB_ENV is not set"
  exit 1
fi

if ! command -v cygpath >/dev/null 2>&1; then
  echo "::error::cygpath is required to shorten Windows paths"
  exit 1
fi

conda_env="${CONDA_ENV%/}"
conda_env_name="$(basename "${conda_env}")"
conda_env_parent="$(dirname "${conda_env}")"
conda_env_parent_win="$(cygpath -w "${conda_env_parent}")"

if [[ -n "${SHORT_CONDA_DRIVE:-}" ]]; then
  short_drive="${SHORT_CONDA_DRIVE%:}:"
else
  short_drive=""
  for drive in T S R Q P O N M L K J I H G F E D; do
    if [[ ! -e "/${drive,,}" ]]; then
      short_drive="${drive}:"
      break
    fi
  done
fi

if [[ -z "${short_drive}" ]]; then
  echo "::error::Could not find an unused drive letter for the Conda env path"
  exit 1
fi

if [[ ! "${short_drive}" =~ ^[A-Za-z]:$ ]]; then
  echo "::error::SHORT_CONDA_DRIVE must be a Windows drive letter like T:"
  exit 1
fi

cmd //c "subst ${short_drive} /D >NUL 2>NUL || exit /B 0"
cmd //c "subst ${short_drive} \"${conda_env_parent_win}\""

short_conda_env="${short_drive}/${conda_env_name}"

{
  echo "CONDA_ENV=${short_conda_env}"
  echo "CONDA_RUN=conda run --no-capture-output -p ${short_conda_env}"
} >> "${GITHUB_ENV}"

if [[ -n "${BUILD_ENV_FILE:-}" && -f "${BUILD_ENV_FILE}" ]]; then
  {
    echo "CONDA_ENV=${short_conda_env}"
    echo "CONDA_RUN=conda run --no-capture-output -p ${short_conda_env}"
  } >> "${BUILD_ENV_FILE}"
fi

echo "Mapped ${conda_env_parent_win} to ${short_drive}"
echo "Using ${short_conda_env} as the Conda environment prefix"
