param(
    [Parameter(Mandatory = $true)] [string] $PyTorchArtifact,
    [Parameter(Mandatory = $true)] [string] $TensorRTArtifact,
    [Parameter(Mandatory = $true)] [string] $TargetRoot,
    [Parameter(Mandatory = $true)] [string] $CudaRoot,
    [Parameter(Mandatory = $true)] [string] $PythonRoot
)

$ErrorActionPreference = "Stop"

function Expand-ZipArtifact {
    param([string] $Artifact, [string] $Destination, [string] $Name)
    $source = $Artifact
    if ($Artifact -match '^https?://') {
        $source = Join-Path $env:RUNNER_TEMP "$Name.zip"
        Invoke-WebRequest -Uri $Artifact -OutFile $source
    }
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "$Name artifact was not found: $source"
    }
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    python -m zipfile -e $source $Destination
}

function Find-Root {
    param([string] $SearchRoot, [string] $IncludePath, [string] $LibraryPath, [string] $Name)
    $candidates = @($SearchRoot) + @(Get-ChildItem -LiteralPath $SearchRoot -Directory -Recurse | Select-Object -ExpandProperty FullName)
    foreach ($candidate in $candidates) {
        if ((Test-Path (Join-Path $candidate $IncludePath)) -and (Test-Path (Join-Path $candidate $LibraryPath))) {
            return $candidate
        }
    }
    throw "Could not locate the normalized $Name root under $SearchRoot"
}

if ($env:PYTHON_VERSION -ne '3.13') {
    throw "Windows ARM64 RTX supports only Python 3.13; got '$env:PYTHON_VERSION'"
}
if (Test-Path -LiteralPath $TargetRoot) {
    Remove-Item -LiteralPath $TargetRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null

$torchExtract = Join-Path $TargetRoot "pytorch"
$trtExtract = Join-Path $TargetRoot "tensorrt-rtx"
Expand-ZipArtifact -Artifact $PyTorchArtifact -Destination $torchExtract -Name "pytorch-arm64"
Expand-ZipArtifact -Artifact $TensorRTArtifact -Destination $trtExtract -Name "tensorrt-rtx-arm64"

$torchRoot = Find-Root -SearchRoot $torchExtract -IncludePath "include" -LibraryPath "lib" -Name "PyTorch"
$trtRoot = Find-Root -SearchRoot $trtExtract -IncludePath "include" -LibraryPath "lib" -Name "TensorRT-RTX"
$cudaArm64Lib = Join-Path $CudaRoot "lib\arm64\cudart.lib"
if (-not (Test-Path -LiteralPath $cudaArm64Lib)) {
    throw "CUDA target library is missing: $cudaArm64Lib"
}
$cudaVersionFile = Join-Path $CudaRoot "version.json"
if (-not (Test-Path -LiteralPath $cudaVersionFile -PathType Leaf)) {
    throw "CUDA version file is missing: $cudaVersionFile"
}
$cudaVersion = (Get-Content -LiteralPath $cudaVersionFile -Raw | ConvertFrom-Json).cuda.version
if ($cudaVersion -notlike '13.4*') {
    throw "Windows ARM64 builds require CUDA 13.4 Preview; got '$cudaVersion'"
}
$pythonHeader = Join-Path $PythonRoot "include\Python.h"
$pythonImportLibrary = Join-Path $PythonRoot "libs\python313.lib"
foreach ($pythonFile in @($pythonHeader, $pythonImportLibrary)) {
    if (-not (Test-Path -LiteralPath $pythonFile -PathType Leaf)) {
        throw "ARM64 Python 3.13 development file is missing: $pythonFile"
    }
}
foreach ($library in @('c10.lib', 'torch.lib', 'torch_cpu.lib', 'torch_python.lib')) {
    if (-not (Test-Path -LiteralPath (Join-Path $torchRoot "lib\$library"))) {
        throw "ARM64 PyTorch artifact is missing lib\$library"
    }
}
if (-not (Get-ChildItem -LiteralPath (Join-Path $trtRoot "lib") -Filter "tensorrt_rtx*.lib" | Select-Object -First 1)) {
    throw "ARM64 TensorRT-RTX artifact contains no tensorrt_rtx import library"
}

function Export-GitHubEnvironment([string] $Name, [string] $Value) {
    "$Name=$($Value -replace '\\', '/')" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
}
Export-GitHubEnvironment "TORCHTRT_TARGET_PLATFORM" "windows-arm64"
Export-GitHubEnvironment "TORCHTRT_TARGET_TORCH_ROOT" $torchRoot
Export-GitHubEnvironment "TORCHTRT_TARGET_TRT_ROOT" $trtRoot
Export-GitHubEnvironment "TORCHTRT_TARGET_CUDA_ROOT" $CudaRoot
Export-GitHubEnvironment "TORCHTRT_TARGET_PYTHON_ROOT" $PythonRoot
Write-Host "Prepared Windows ARM64 target sysroot: PyTorch=$torchRoot TensorRT-RTX=$trtRoot CUDA=$CudaRoot Python=$PythonRoot"
