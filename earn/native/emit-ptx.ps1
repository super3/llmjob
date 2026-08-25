# Register and shared-memory usage for the fold kernel, straight from ptxas.
#
# This answers a question the runtime cannot: whether the accumulators fit. The
# tile accumulator lives across the whole of k, so its area is a register
# budget, and once ptxas starts spilling to local memory the arithmetic still
# happens but every chunk boundary pays a memory round trip. A spilling kernel
# looks exactly like a slow one from the outside.
#
# Nsight Compute would say the same and more, but its counters need a reboot
# after RmProfilingAdminOnly is cleared. This needs nothing.

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

$vsPath = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" `
    -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vsPath) { throw "no Visual Studio with the C++ toolset" }
$vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvars)) { throw "vcvars64.bat not found under $vsPath" }

$cudaRoot = Get-ChildItem "${env:ProgramFiles}\NVIDIA GPU Computing Toolkit\CUDA" -Directory |
    Sort-Object Name -Descending | Select-Object -First 1
if (-not $cudaRoot) { throw "no CUDA toolkit found" }

& cmd /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}

$nvcc = Join-Path $cudaRoot.FullName "bin\nvcc.exe"
$out = Join-Path $env:TEMP "pearl_kernel.ptx"
& $nvcc -arch=sm_89 -std=c++17 -O3 -ptx `
    (Join-Path $here "src\pearl_kernel.cu") -o $out -I (Join-Path $here "src") 2>&1 |
    Out-Null
    Out-Null
