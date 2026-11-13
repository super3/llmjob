# Build the Quantus CUDA core on Windows: earn\native\quantus\build\quantus_cuda.node.
#
#   powershell -ExecutionPolicy Bypass -File earn\native\quantus\cuda\build.ps1 [-Arch 75,80,86,89,90,120]
#       [-Cuda <toolkit dir>] [-Out <file.node>] [-Defs "-DQC_MINB=4 ..."]
#
# Needs CUDA 12.8 or newer (sm_120 for RTX 50), MSVC (vswhere finds it), and Node's headers and node.lib in
# node-gyp's cache (run `npx node-gyp install` once). The addon links the CUDA runtime and the C runtime
# statically, so it needs no DLL beside it, only the NVIDIA driver. -Arch 89 builds for one card (faster).
param(
    [string]$Arch = '75,80,86,89,90,120',
    [string]$Cuda = '',
    [string]$Out = '',
    [string]$Defs = ''
)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Out) { $Out = Join-Path (Split-Path -Parent $here) 'build\quantus_cuda.node' }

if (-not $Cuda) { $Cuda = $env:CUDA_PATH }
if (-not $Cuda) {
    $Cuda = (Get-ChildItem "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA" -ErrorAction SilentlyContinue |
             Sort-Object Name -Descending | Select-Object -First 1).FullName
}
$nvcc = Join-Path $Cuda 'bin\nvcc.exe'
if (-not (Test-Path $nvcc)) { throw "nvcc not found under '$Cuda' (pass -Cuda)" }

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'no MSVC C++ toolchain found' }
& cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "Env:$($matches[1])" -Value $matches[2] }
}

$ver = (& node -p 'process.versions.node').Trim()
$gyp = Join-Path $env:LOCALAPPDATA "node-gyp\Cache\$ver"
$inc = Join-Path $gyp 'include\node'
$lib = Join-Path $gyp 'x64\node.lib'
if (-not (Test-Path (Join-Path $inc 'node_api.h')) -or -not (Test-Path $lib)) {
    throw "Node $ver headers or node.lib missing under $gyp (run: npx node-gyp install)"
}

$gencode = @()
foreach ($a in $Arch.Split(',')) { $gencode += "-gencode=arch=compute_$a,code=sm_$a" }
$last = $Arch.Split(',')[-1]
$gencode += "-gencode=arch=compute_$last,code=compute_$last"  # PTX for cards newer than the list

New-Item -ItemType Directory -Force (Split-Path -Parent $Out) | Out-Null
$obj = Join-Path $env:TEMP "qc-cuda-$PID"
New-Item -ItemType Directory -Force $obj | Out-Null
$d = @(); if ($Defs) { $d = @($Defs.Split(' ') | Where-Object { $_ }) }

Push-Location $obj
try {
    & $nvcc -O3 -std=c++17 -cudart static -Xcompiler /MT @gencode @d -I $here -c (Join-Path $here 'qc_host.cu') -o qc_host.obj
    if ($LASTEXITCODE -ne 0) { throw 'qc_host.cu failed' }
    & cl /nologo /O2 /MT /c /I $inc /I $here (Join-Path $here 'qc_napi.c') /Foqc_napi.obj
    if ($LASTEXITCODE -ne 0) { throw 'qc_napi.c failed' }
    & cl /nologo /O2 /MT /c /EHsc (Join-Path $here 'qc_win_hook.cc') /Foqc_win_hook.obj
    if ($LASTEXITCODE -ne 0) { throw 'qc_win_hook.cc failed' }
    & link /nologo /DLL /OUT:$Out qc_host.obj qc_napi.obj qc_win_hook.obj $lib delayimp.lib `
        /DELAYLOAD:node.exe "/LIBPATH:$Cuda\lib\x64" cudart_static.lib
    if ($LASTEXITCODE -ne 0) { throw 'link failed' }
} finally {
    Pop-Location
    Remove-Item -Recurse -Force $obj -ErrorAction SilentlyContinue
}
Remove-Item -ErrorAction SilentlyContinue ([IO.Path]::ChangeExtension($Out, '.exp')), ([IO.Path]::ChangeExtension($Out, '.lib'))
Write-Host "built $Out ($Arch)"
