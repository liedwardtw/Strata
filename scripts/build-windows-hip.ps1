# build-windows-hip.ps1 -- configure + build the ik2 HIP engine target on Windows (gfx1100).
# ASCII only (PowerShell 5.1 compatible).
#
# Prerequisites:
#   - ROCm Windows SDK with hipcc/clang (AMDGPU_TARGETS=gfx1100)
#   - Visual Studio build tools (vcvarsall x64) for mt.exe manifest embedding
#   - CMake + Ninja
#
# Usage:
#   .\build-windows-hip.ps1 -EngineDir <repo>\engine -BuildDir <repo>\build-hip-win `
#       -RocmSdk <path to _rocm_sdk_devel> [-Target strata-ik2] [-Vcvars <path>]

param(
    [Parameter(Mandatory=$true)][string]$EngineDir,
    [Parameter(Mandatory=$true)][string]$BuildDir,
    [Parameter(Mandatory=$true)][string]$RocmSdk,
    [string]$Target = 'strata-ik2',
    [string]$Vcvars = 'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat',
    [string]$ExtraCmake = '',
    [switch]$SkipVcvars
)
$ErrorActionPreference = 'Continue'

if (-not $SkipVcvars) {
    if (-not (Test-Path $Vcvars)) { Write-Host "ERROR: vcvars.bat not found: $Vcvars"; exit 1 }
    $envDump = & cmd /c "`"$Vcvars`" x64 >nul 2>&1 && set"
    foreach ($line in $envDump) {
        if ($line -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:$($matches[1])" -Value $matches[2] }
    }
}

$env:HIP_PATH = $RocmSdk
$env:ROCM_PATH = $RocmSdk
$env:HIP_PLATFORM = 'amd'
$env:HIP_DEVICE_LIB_PATH = "$RocmSdk\lib\llvm\amdgcn\bitcode"
$env:LLVM_PATH = "$RocmSdk\lib\llvm"
$env:PATH = "$RocmSdk\bin;$RocmSdk\lib\llvm\bin;$env:PATH"

# Ninja is required: the Visual Studio generator cannot build HIP on Windows
# ("HIP language not currently supported by the ... generator"), and it leaves a
# build directory that ninja then refuses to read. The compiler names are given
# without the .exe suffix on purpose: cmake splits a "-D" value on the first dot,
# so "clang.exe" arrives as clang plus a stray ".exe" argument.
$cmakeArgs = @(
    '-G', 'Ninja',
    '-S', $EngineDir, '-B', $BuildDir,
    '-DCMAKE_BUILD_TYPE=Release',
    '-DSTRATA_ENABLE_HIP=ON', '-DSTRATA_ENABLE_CUDA=OFF',
    '-DGGML_HIP=ON', '-DAMDGPU_TARGETS=gfx1100',
    '-DBUILD_TESTING=ON',
    '-DCMAKE_C_COMPILER=clang', '-DCMAKE_CXX_COMPILER=clang++',
    '-DCMAKE_HIP_COMPILER=clang'
)
if ($ExtraCmake) { $cmakeArgs += ($ExtraCmake -split ' ') }
& cmake @cmakeArgs 2>&1 | Select-Object -Last 5
& ninja -C $BuildDir $Target
"BUILD_EXIT=$LASTEXITCODE"
