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
if "%VCToolsInstallDir%"=="" (
  echo VCToolsInstallDir is unset; initialize MSVC with vcvarsall.bat amd64_arm64 first
  exit /b 1
)
rem Bazel's VS discovery can select a different, incomplete installation. Pin it
rem to the VC root selected by the active amd64_arm64 vcvarsall environment.
for %%I in ("%VCToolsInstallDir%\..\..\..") do set "BAZEL_VC=%%~fI"
echo Using BAZEL_VC=%BAZEL_VC%
if not "%TORCHTRT_TARGET_PLATFORM%"=="windows-arm64" (
  echo TORCHTRT_TARGET_PLATFORM is not windows-arm64
  exit /b 1
)
rem Reuse the amd64_arm64 environment initialized by vcvarsall.bat. Without
rem this, torch.utils.cpp_extension rejects the already-active VC environment.
set DISTUTILS_USE_SDK=1
python -m pip install --upgrade setuptools==72.1.0 wheel
if errorlevel 1 exit /b %errorlevel%
python setup.py bdist_wheel --use-rtx --windows-on-arm
exit /b %errorlevel%
