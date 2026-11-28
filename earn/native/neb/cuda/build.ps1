# Build the NEB CUDA core on Windows: earn\native\neb\build\neb_cuda.node.
#
#   powershell -ExecutionPolicy Bypass -File earn\native\neb\cuda\build.ps1 [-Arch 75,80,86,89,90,120]
#       [-Cuda <toolkit dir>] [-Out <file.node>] [-Defs "-DNB_PAIR=1 ..."]
#
# Needs CUDA 12.8 or newer (sm_120 for RTX 50), MSVC (vswhere finds it), and Node's headers and node.lib in
# node-gyp's cache (run `npx node-gyp install` once). The addon links the CUDA runtime and the C runtime
# statically, so it needs no DLL beside it, only the NVIDIA driver. node.exe is delay-loaded and answered by
# ..\..\quantus\win_delay_load_hook.cc, so the addon loads in the Electron app too. -Arch 89 builds for one
# card (faster). The same sources as cuda/build.sh builds on Linux.
param(
    [string]$Arch = '75,80,86,89,90,120',
    [string]$Cuda = '',
    [string]$Out = '',
    [string]$Defs = ''
)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$neb = Split-Path -Parent $here
if (-not $Out) { $Out = Join-Path $neb 'build\neb_cuda.node' }

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
$obj = Join-Path $env:TEMP "neb-cuda-$PID"
New-Item -ItemType Directory -Force $obj | Out-Null
$d = @(); if ($Defs) { $d = @($Defs.Split(' ') | Where-Object { $_ }) }
$ref = Join-Path $neb 'ref'
$objs = @('neb_cuda.obj', 'neb_napi.obj', 'win_hook.obj')

Push-Location $obj
try {
    & $nvcc -O3 -std=c++17 -cudart static -Xcompiler /MT @gencode @d -I $neb -c (Join-Path $here 'neb_cuda.cu') -o neb_cuda.obj
    if ($LASTEXITCODE -ne 0) { throw 'neb_cuda.cu failed' }
    # The chain's reference, which the host uses to evaluate each sample once a model.
    foreach ($f in 'nebhash2', 'shares', 'blake3_min', 'chacha8') {
        & cl /nologo /O2 /MT /c /I $ref (Join-Path $ref "$f.c") "/Fo$f.obj"
        if ($LASTEXITCODE -ne 0) { throw "ref\$f.c failed" }
        $objs += "$f.obj"
    }
    & cl /nologo /O2 /MT /c /I $inc /I $neb (Join-Path $neb 'neb_napi.c') /Foneb_napi.obj
    if ($LASTEXITCODE -ne 0) { throw 'neb_napi.c failed' }
    & cl /nologo /O2 /MT /c /EHsc (Join-Path $neb '..\quantus\win_delay_load_hook.cc') /Fowin_hook.obj
    if ($LASTEXITCODE -ne 0) { throw 'win_delay_load_hook.cc failed' }
    & link /nologo /DLL /OUT:$Out @objs $lib delayimp.lib /DELAYLOAD:node.exe "/LIBPATH:$Cuda\lib\x64" cudart_static.lib
    if ($LASTEXITCODE -ne 0) { throw 'link failed' }
} finally {
    Pop-Location
    Remove-Item -Recurse -Force $obj -ErrorAction SilentlyContinue
}
Remove-Item -ErrorAction SilentlyContinue ([IO.Path]::ChangeExtension($Out, '.exp')), ([IO.Path]::ChangeExtension($Out, '.lib'))
Write-Host "built $Out ($Arch)"
