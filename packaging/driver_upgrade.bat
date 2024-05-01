REM Source: https://github.com/pytorch/builder/blob/4e109742d88ff3c85e77d60bc4d90d229d7f6afe/windows/internal/driver_update.bat

set "DRIVER_DOWNLOAD_LINK=https://ossci-windows.s3.amazonaws.com/528.89-data-center-tesla-desktop-winserver-2016-2019-2022-dch-international.exe"
curl --retry 3 -kL %DRIVER_DOWNLOAD_LINK% --output 528.89-data-center-tesla-desktop-winserver-2016-2019-2022-dch-international.exe
if errorlevel 1 exit /b 1

start /wait 528.89-data-center-tesla-desktop-winserver-2016-2019-2022-dch-international.exe -s -noreboot
if errorlevel 1 exit /b 1

del 528.89-data-center-tesla-desktop-winserver-2016-2019-2022-dch-international.exe || ver > NUL
