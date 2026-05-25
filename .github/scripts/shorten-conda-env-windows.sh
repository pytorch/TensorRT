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

to_windows_path() {
  local path="${1//\\//}"

  if [[ "${path}" =~ ^[A-Za-z]:/ ]]; then
    printf "%s\n" "${path//\//\\}"
  else
    cygpath -w "${path}"
  fi
}

to_bash_path() {
  local path="${1//\\//}"

  if [[ "${path}" =~ ^([A-Za-z]):(/.*)?$ ]]; then
    printf "/%s%s\n" "${BASH_REMATCH[1],,}" "${BASH_REMATCH[2]}"
  else
    printf "%s\n" "${path}"
  fi
}

find_unused_drive() {
  local drive

  for drive in T S R Q P O N M L K J I H G F E D; do
    if [[ ! -e "/${drive,,}" ]]; then
      printf "%s:\n" "${drive}"
      return 0
    fi
  done

  return 1
}

conda_env="${CONDA_ENV%/}"
conda_env_bash="$(to_bash_path "${conda_env}")"
conda_env_name="$(basename "${conda_env_bash}")"
conda_env_parent="$(dirname "${conda_env_bash}")"
conda_env_parent_win="$(to_windows_path "${conda_env_parent}")"

if [[ -n "${SHORT_CONDA_DRIVE:-}" ]]; then
  short_drive="${SHORT_CONDA_DRIVE%:}:"
elif ! short_drive="$(find_unused_drive)"; then
  echo "::error::Could not find an unused drive letter for the Conda env path"
  exit 1
fi

if [[ ! "${short_drive}" =~ ^[A-Za-z]:$ ]]; then
  echo "::error::SHORT_CONDA_DRIVE must be a Windows drive letter like T:"
  exit 1
fi

MSYS2_ARG_CONV_EXCL="*" MSYS2_ENV_CONV_EXCL="CONDA_ENV_PARENT_WIN" SHORT_CONDA_DRIVE="${short_drive}" CONDA_ENV_PARENT_WIN="${conda_env_parent_win}" powershell.exe -NoProfile -ExecutionPolicy Bypass -Command '
$ErrorActionPreference = "Stop"
$drive = $env:SHORT_CONDA_DRIVE
$target = $env:CONDA_ENV_PARENT_WIN
$target = $target -replace "^[\\/]+(?=[A-Za-z]:[\\/])", ""
Write-Host "Mapping Conda env parent path: $target"
if (-not (Test-Path -LiteralPath $target -PathType Container)) {
  throw "Conda env parent path not found: $target"
}
& subst.exe $drive /D 2>$null
& subst.exe $drive $target
if ($LASTEXITCODE -ne 0) {
  throw "subst failed to map $drive to $target"
}
'

short_conda_env="${short_drive}/${conda_env_name}"
short_conda_run="conda run --no-capture-output -p ${short_conda_env}"

{
  echo "CONDA_ENV=${short_conda_env}"
  echo "CONDA_RUN=${short_conda_run}"
} >> "${GITHUB_ENV}"

build_env_file="${BUILD_ENV_FILE:-}"
if [[ -n "${build_env_file}" ]]; then
  build_env_file="$(to_bash_path "${build_env_file}")"
fi

if [[ -n "${build_env_file}" && -f "${build_env_file}" ]]; then
  {
    printf "export CONDA_ENV=%q\n" "${short_conda_env}"
    printf "export CONDA_RUN=%q\n" "${short_conda_run}"
  } >> "${build_env_file}"
fi

echo "Mapped ${conda_env_parent_win} to ${short_drive}"
echo "Using ${short_conda_env} as the Conda environment prefix"
