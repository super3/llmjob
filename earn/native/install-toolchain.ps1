# Install the local CUDA build toolchain. RUN THIS AS ADMINISTRATOR.
#
# Why it is needed: nvcc on Windows does not compile anything itself, it drives
# MSVC's cl.exe. This box has an RTX 4090 and a driver but neither the CUDA
# toolkit nor MSVC, so every kernel change currently goes through CI --
# commit, push, wait for GitHub Actions, download the artifact -- which is
# about four minutes a round trip. It also means no Nsight Compute, so kernel
# tuning is done by timing rather than by looking at occupancy and stalls.
#
# Nothing here is required to BUILD RELEASES: CI still produces the shipped
# binary. This is purely to make local iteration and profiling possible.
#
# To run: right-click Start -> Terminal (Admin), then:
#   powershell -ExecutionPolicy Bypass -File "C:\Users\template\Code\llmjob\earn\native\install-toolchain.ps1"

$ErrorActionPreference = 'Stop'

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "This needs to run from an ADMINISTRATOR terminal." -ForegroundColor Red
    Write-Host "Right-click Start, choose 'Terminal (Admin)', and run it again." -ForegroundColor Red
    exit 1
}

Write-Host "`n=== 1/2  Visual Studio 2022 Build Tools (C++ workload) ===" -ForegroundColor Cyan
Write-Host "About 2-3 GB. This is what provides cl.exe, which nvcc requires.`n"

# VCTools is the C++ compiler and libraries; the Windows SDK comes with
# --includeRecommended. --wait makes winget block until the nested VS installer
# actually finishes, which it otherwise does not.
winget install --id Microsoft.VisualStudio.2022.BuildTools --exact --source winget `
    --accept-package-agreements --accept-source-agreements `
    --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

Write-Host "`n=== 2/2  CUDA Toolkit 12.6 ===" -ForegroundColor Cyan
Write-Host "About 3 GB. 12.6 is chosen deliberately: the installed display driver"
Write-Host "is 560.94, which IS the 12.6 driver, so this combination is guaranteed"
Write-Host "to work. A newer toolkit would want a newer driver.`n"

# The component list deliberately EXCLUDES the bundled display driver, so the
# working 560.94 is left alone. nsight_compute is included because profiling is
# half the reason for doing this at all.
$cudaComponents = "-s nvcc_12.6 cudart_12.6 nvrtc_12.6 nvrtc_dev_12.6 " +
                  "visual_studio_integration_12.6 cuda_profiler_api_12.6 nsight_compute_12.6"

try {
    winget install --id Nvidia.CUDA --exact --version 12.6 --source winget `
        --accept-package-agreements --accept-source-agreements `
        --override $cudaComponents
} catch {
    Write-Host "`nComponent-limited install failed. That usually means one of the" -ForegroundColor Yellow
    Write-Host "component names is wrong for this installer build." -ForegroundColor Yellow
    Write-Host "Fallback (installs the whole toolkit, and MAY update the display driver):" -ForegroundColor Yellow
    Write-Host "  winget install --id Nvidia.CUDA --exact --version 12.6 --source winget ``" -ForegroundColor Yellow
    Write-Host "      --accept-package-agreements --accept-source-agreements" -ForegroundColor Yellow
    exit 1
}

Write-Host "`n=== Done. Verifying ===" -ForegroundColor Cyan
$cuda = Get-ChildItem "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA" -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending | Select-Object -First 1
if ($cuda) { Write-Host ("CUDA:  " + $cuda.FullName) -ForegroundColor Green }
else { Write-Host "CUDA:  NOT FOUND" -ForegroundColor Red }

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
    $vs = & $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -format value -property installationPath
    if ($vs) { Write-Host ("MSVC:  " + $vs) -ForegroundColor Green }
    else { Write-Host "MSVC:  build tools present but no C++ compiler component" -ForegroundColor Red }
} else { Write-Host "MSVC:  NOT FOUND" -ForegroundColor Red }

Write-Host "`nClose this admin window. Back in the normal session, I'll take it from here." -ForegroundColor Cyan
