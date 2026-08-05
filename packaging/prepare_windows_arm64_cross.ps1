param(
    [string] $PyTorchArtifact = "https://pypi.nvidia.com/nvtorch_oot_nightly/torch/torch-2.14.0.dev20260728%2Bcu134-cp313-cp313-win_arm64.whl",
    [string] $PythonArtifact = "https://api.nuget.org/v3-flatcontainer/pythonarm64/3.13.0/pythonarm64.3.13.0.nupkg",
    [Parameter(Mandatory = $true)] [string] $TargetRoot,
    [Parameter(Mandatory = $true)] [string] $CudaRoot
)

$ErrorActionPreference = "Stop"

$PyTorchArtifactSha256 = "23862a93476cb038ffd26f7141cd476717d97b69b50074f2ab14036eb6093200"
$PythonArtifactSha256 = "f44428dc94e6f9c72cd69ad6436280784e6f9eed46a149641fee71866d3081f3"

$ArtifactTempRoot = $env:RUNNER_TEMP
if (-not $ArtifactTempRoot) {
    $ArtifactTempRoot = [System.IO.Path]::GetTempPath()
}
function Expand-ZipArtifact {
    param([string] $Artifact, [string] $Destination, [string] $Name, [string] $ExpectedSha256 = "")
    $source = $Artifact
    if ($Artifact -match '^https?://') {
        $source = Join-Path $ArtifactTempRoot "$Name.zip"
        Invoke-WebRequest -Uri $Artifact -OutFile $source
    }
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "$Name artifact was not found: $source"
    }
    if ($ExpectedSha256) {
        $actualSha256 = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualSha256 -ne $ExpectedSha256) {
            throw "$Name artifact SHA256 mismatch: expected $ExpectedSha256, got $actualSha256"
        }
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

New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null

$torchExtract = Join-Path $TargetRoot "pytorch"
if (-not (Test-Path -LiteralPath $torchExtract -PathType Container)) {
    Expand-ZipArtifact -Artifact $PyTorchArtifact -Destination $torchExtract -Name "pytorch-arm64" -ExpectedSha256 $PyTorchArtifactSha256
}

$torchRoot = Find-Root -SearchRoot $torchExtract -IncludePath "include" -LibraryPath "lib" -Name "PyTorch"
$cudaArm64Lib = Join-Path $CudaRoot "lib\arm64\cudart.lib"
$pythonExtract = Join-Path $TargetRoot "pythonarm64"
if (-not (Test-Path -LiteralPath $pythonExtract -PathType Container)) {
    Expand-ZipArtifact -Artifact $PythonArtifact -Destination $pythonExtract -Name "pythonarm64" -ExpectedSha256 $PythonArtifactSha256
}
$pythonRoot = Find-Root -SearchRoot $pythonExtract -IncludePath "include\Python.h" -LibraryPath "libs\python313.lib" -Name "Python ARM64"
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
$pythonHeader = Join-Path $pythonRoot "include\Python.h"
$pythonImportLibrary = Join-Path $pythonRoot "libs\python313.lib"
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
function Export-GitHubEnvironment([string] $Name, [string] $Value) {
    "$Name=$($Value -replace '\\', '/')" | Out-File -FilePath $env:GITHUB_ENV -Encoding utf8 -Append
}
Export-GitHubEnvironment "TORCHTRT_TARGET_PLATFORM" "windows-arm64"
Export-GitHubEnvironment "TORCHTRT_TARGET_TORCH_ROOT" $torchRoot
Export-GitHubEnvironment "TORCHTRT_TARGET_CUDA_ROOT" $CudaRoot
Export-GitHubEnvironment "TORCHTRT_TARGET_PYTHON_ROOT" $pythonRoot
Write-Host "Prepared Windows ARM64 target sysroot: PyTorch=$torchRoot CUDA=$CudaRoot Python=$pythonRoot"
