# Build the CUDA core locally. Requires install-toolchain.ps1 to have been run.
#
# This is the fast path for kernel work: seconds instead of the ~4 minute
# commit-push-CI-download round trip. It targets ONLY sm_89 (Ada, the RTX 4090
# in this box) because that halves compile time and nothing else runs here.
# CI still builds sm_86/89/120 for the shipped binary -- do not treat a green
# local build as a substitute for that.
#
#   powershell -ExecutionPolicy Bypass -File earn\native\build-local.ps1

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

# --- locate the toolchain --------------------------------------------------
$cudaRoot = Get-ChildItem "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA" -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending | Select-Object -First 1
if (-not $cudaRoot) { throw "No CUDA toolkit found. Run install-toolchain.ps1 as administrator first." }
$nvcc = Join-Path $cudaRoot.FullName "bin\nvcc.exe"
if (-not (Test-Path $nvcc)) { throw "nvcc not found at $nvcc" }

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) { throw "vswhere not found. Run install-toolchain.ps1 as administrator first." }
$vsPath = & $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -format value -property installationPath
if (-not $vsPath) { throw "No MSVC C++ toolchain found. Run install-toolchain.ps1 as administrator first." }

# nvcc shells out to cl.exe and fails with "Cannot find compiler 'cl.exe' in
# PATH" unless MSVC's environment is sourced first. vcvarsall sets it in a
# child cmd, so capture the resulting variables and apply them here.
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found under $vsPath" }
Write-Host "MSVC: $vsPath" -ForegroundColor DarkGray
Write-Host "CUDA: $($cudaRoot.FullName)" -ForegroundColor DarkGray

& cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path ("Env:" + $matches[1]) -Value $matches[2] }
}

# --- compile ----------------------------------------------------------------
# Deliberately NOT build/: node-gyp's clean phase deletes that directory, which
# is what produced "cannot find -lpearl_cuda" at link time in CI.
New-Item -ItemType Directory -Force -Path cuda-build | Out-Null

$arch = "-gencode arch=compute_89,code=sm_89"
$common = "-O3 -std=c++17 -cudart static $arch"

Write-Host "`nCompiling pearl_kernel.cu ..." -ForegroundColor Cyan
& $nvcc $common.Split(' ') -c src/pearl_kernel.cu -o cuda-build/pearl_kernel.o
if ($LASTEXITCODE -ne 0) { throw "pearl_kernel.cu failed" }

Write-Host "Compiling pearl_host.cu ..." -ForegroundColor Cyan
& $nvcc $common.Split(' ') -c src/pearl_host.cu -o cuda-build/pearl_host.o
if ($LASTEXITCODE -ne 0) { throw "pearl_host.cu failed" }

Write-Host "Archiving ..." -ForegroundColor Cyan
& $nvcc -lib -cudart static cuda-build/pearl_kernel.o cuda-build/pearl_host.o -o cuda-build/pearl_cuda.lib
if ($LASTEXITCODE -ne 0) { throw "archive failed" }

# --- link the addon ---------------------------------------------------------
# This is the step that catches a drift between pearl_core.cc's extern "C"
# declarations and pearl_host.cu's definitions: it compiles clean and fails at
# LINK time, which reading the diff does not find.
if (-not (Test-Path node_modules/node-addon-api)) {
    Write-Host "Installing node-addon-api ..." -ForegroundColor Cyan
    & npm install --no-save node-addon-api node-gyp
}
Write-Host "Building the addon ..." -ForegroundColor Cyan
& npx node-gyp rebuild
if ($LASTEXITCODE -ne 0) { throw "node-gyp failed" }

$out = "build/Release/pearl_core.node"
if (Test-Path $out) {
    Write-Host "`nBuilt: $((Resolve-Path $out).Path)" -ForegroundColor Green
} else {
    throw "addon did not appear at $out"
}
