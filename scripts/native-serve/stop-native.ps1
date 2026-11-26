# stop-native.ps1 -- stop the native Strata serve and its ik2 engine cleanly.
# ASCII only (PowerShell 5.1 compatible). gpu_guard is left running.
$ErrorActionPreference = 'Continue'
$procs = Get-Process strata-ik2,strata,llama-server,llama-cli,hipfire*,daemon* -ErrorAction SilentlyContinue
if ($procs) {
    Write-Output "Stopping: $($procs.Name -join ', ')"
    $procs | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
} else {
    Write-Output "No model process running."
}
# Also stop a serve started via python -m serve.server (commandline matches serve.server).
$pyServ = Get-CimInstance Win32_Process -Filter "name like '%python%'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match 'serve\.server' -and $_.CommandLine -match 'strata' }
if ($pyServ) {
    Write-Output "Stopping python serve.server pid(s): $($pyServ.ProcessId -join ', ')"
    $pyServ | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
$left = Get-Process strata*,llama*,hipfire*,daemon* -ErrorAction SilentlyContinue
if ($left) { Write-Output "STILL RUNNING: $($left.Name -join ',')" } else { Write-Output "STOPPED OK" }
