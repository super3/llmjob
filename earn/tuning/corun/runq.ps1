param([string]$Tag, [string]$Mode, [int]$Secs = 90, [int]$Warm = 30)
# One corunq.mjs run on Windows, with power and clocks over its window. Mode: miner | llm | both | pause |
# coldllm | coldminer | two (llm + miner processes). Env passes through (MINER, SOLO, NTOK, REPS, PRIO,
# QUANTUS_CUDA_GRID, PEARL_FREE_SMS). CUDA_DLLS: a folder with cudart64_12.dll and cublas64_12.dll for
# node-llama-cpp's CUDA build.
$dir = $PSScriptRoot
if ($env:CUDA_DLLS) { $env:PATH = $env:CUDA_DLLS + ';' + $env:PATH }
$csv = Join-Path $dir "smi-$Tag.csv"
$smi = Start-Process nvidia-smi -ArgumentList '--query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,utilization.gpu', '--format=csv,noheader,nounits', '-lms', '500' -RedirectStandardOutput $csv -NoNewWindow -PassThru
$lastLine = $null
try {
  if ($Mode -eq 'two') {
    $ready = Join-Path $dir "ready-$Tag"
    Remove-Item $ready -ErrorAction SilentlyContinue
    $env:READY_FILE = $ready
    $m = Start-Process node -ArgumentList "$dir\corunq.mjs", 'miner', $Secs, $Warm, "$Tag-miner" -NoNewWindow -PassThru -RedirectStandardOutput "$dir\out-$Tag-miner.txt" -RedirectStandardError "$dir\err-$Tag-miner.txt"
    $lastLine = node "$dir\corunq.mjs" llm $Secs $Warm "$Tag-llm" 2>"$dir\err-$Tag.txt" | Select-Object -Last 1
    $m.WaitForExit()
    Get-Content "$dir\out-$Tag-miner.txt" | Select-Object -Last 1
    Remove-Item Env:READY_FILE
  } else {
    $lastLine = node "$dir\corunq.mjs" $Mode $Secs $Warm $Tag 2>"$dir\err-$Tag.txt" | Select-Object -Last 1
  }
} finally {
  Stop-Process -Id $smi.Id -ErrorAction SilentlyContinue
}
$lastLine
if ($lastLine) {
  $j = $lastLine | ConvertFrom-Json
  if ($j.windowStartUtc) { node "$dir\power.mjs" $csv $j.windowStartUtc $j.windowEndUtc }
}
