param([Parameter(Mandatory = $true)] [string] $DistDirectory)

$ErrorActionPreference = "Stop"
$wheel = Get-ChildItem -LiteralPath $DistDirectory -Filter "*-win_arm64.whl" | Select-Object -First 1
if (-not $wheel) {
    throw "No win_arm64 wheel found in $DistDirectory"
}
if (Get-ChildItem -LiteralPath $DistDirectory -Filter "*-win_amd64.whl") {
    throw "An unexpected win_amd64 wheel was produced"
}

$extractRoot = Join-Path $env:RUNNER_TEMP "torchtrt-win-arm64-validation"
if (Test-Path -LiteralPath $extractRoot) {
    Remove-Item -LiteralPath $extractRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $extractRoot | Out-Null
python -m zipfile -e $wheel.FullName $extractRoot

$nativeFiles = @(Get-ChildItem -LiteralPath $extractRoot -Recurse -File | Where-Object { $_.Extension -in @('.dll', '.pyd') })
$targetLibraries = @(
    Join-Path $env:TORCHTRT_TARGET_TORCH_ROOT "lib\c10.lib"
    Join-Path $env:TORCHTRT_TARGET_TORCH_ROOT "lib\torch_cpu.lib"
    Join-Path $env:TORCHTRT_TARGET_CUDA_ROOT "lib\arm64\cudart.lib"
    Join-Path $env:TORCHTRT_TARGET_PYTHON_ROOT "libs\python313.lib"
)
$nativeFiles += @($targetLibraries | Get-Item)
$nativeFiles += @(Get-ChildItem -LiteralPath (Join-Path $env:TORCHTRT_TARGET_TRT_ROOT "lib") -Filter "tensorrt_rtx*.lib" | Select-Object -First 1)

if (-not $nativeFiles) {
    throw "No native outputs or target import libraries were found"
}
foreach ($file in $nativeFiles) {
    $headers = & dumpbin.exe /headers $file.FullName 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0 -or $headers -notmatch 'AA64 machine \(ARM64\)') {
        throw "Native file is not ARM64: $($file.FullName)"
    }
}
Write-Host "Validated $($nativeFiles.Count) ARM64 native files in $($wheel.Name) and its target sysroot"
