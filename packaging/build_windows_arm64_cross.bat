@echo off
setlocal enabledelayedexpansion
if not "%VSCMD_ARG_HOST_ARCH%"=="x64" (
  echo Expected the x64-hosted MSVC toolchain, got %VSCMD_ARG_HOST_ARCH%
  exit /b 1
)
if not "%VSCMD_ARG_TGT_ARCH%"=="arm64" (
  echo Expected the ARM64 MSVC target, got %VSCMD_ARG_TGT_ARCH%
  exit /b 1
)
if not "%TORCHTRT_TARGET_PLATFORM%"=="windows-arm64" (
  echo TORCHTRT_TARGET_PLATFORM is not windows-arm64
  exit /b 1
)
python -m pip install --upgrade setuptools==72.1.0 wheel
if errorlevel 1 exit /b %errorlevel%
python setup.py bdist_wheel --use-rtx
exit /b %errorlevel%
