@echo off
setlocal enabledelayedexpansion

set TARGET_DRIVER_VN=580.88
set DRIVER_FILENAME=%TARGET_DRIVER_VN%-data-center-tesla-desktop-win10-win11-64bit-dch-international.exe
set "DRIVER_DOWNLOAD_LINK=https://ossci-windows.s3.amazonaws.com/%DRIVER_FILENAME%"

echo Checking current NVIDIA driver version...

:: Parse driver version from nvidia-smi output
:: The line looks like: "| NVIDIA-SMI 580.88   Driver Version: 580.88   CUDA Version: 12.8  |"
:: With space delimiters, token 6 is the version after "Driver Version:"
for /f "tokens=6" %%a in ('nvidia-smi 2^>nul ^| findstr /C:"Driver Version"') do (
    set CURRENT_DRIVER_VN=%%a
)

if not defined CURRENT_DRIVER_VN (
    echo WARNING: Could not detect current NVIDIA driver version, proceeding with update
    goto :do_update
)

echo Current driver version: %CURRENT_DRIVER_VN%
echo Target driver version:  %TARGET_DRIVER_VN%

if "%CURRENT_DRIVER_VN%"=="%TARGET_DRIVER_VN%" (
    echo Driver is already at target version %TARGET_DRIVER_VN%, skipping update.
    exit /b 0
)

echo Driver update needed: %CURRENT_DRIVER_VN% -^> %TARGET_DRIVER_VN%

:do_update
echo Downloading driver from %DRIVER_DOWNLOAD_LINK%...
curl --retry 3 -kL %DRIVER_DOWNLOAD_LINK% --output %DRIVER_FILENAME%
if errorlevel 1 (
    echo ERROR: Failed to download driver
    exit /b 1
)

echo Installing driver %TARGET_DRIVER_VN%...
start /wait %DRIVER_FILENAME% -s -noreboot
if errorlevel 1 (
    echo ERROR: Driver installation failed
    del %DRIVER_FILENAME% 2>nul
    exit /b 1
)

echo Driver update to %TARGET_DRIVER_VN% completed successfully.
del %DRIVER_FILENAME% 2>nul
exit /b 0
