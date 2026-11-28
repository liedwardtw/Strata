# run-native-serve.ps1 -- launch the native Strata serve (server.py) driving the
# optimized ik2 Inkling-2-Mini engine on Windows RX 7900 XTX (HIP gfx1100).
# This gives the full OpenAI/Anthropic API port, monitor + web page, and agent
# tool calling (tool calling is implemented in serve's Python layer; the engine
# itself does not need any tool support).
#
# ASCII only (PowerShell 5.1 compatible). All machine-specific paths come from
# parameters or environment variables -- nothing is hardcoded to one machine.
#
# Usage (PowerShell):
#   $env:IK2_EXE    = 'C:\path\to\strata-ik2.exe'
#   $env:IK2_MODEL  = 'C:\path\to\Inkling-2-Mini-UD-Q4_K_XL.gguf'
#   $env:IK2_TOKENIZER = 'C:\path\to\inkling-pack\tokenizer'
#   $env:ROCM_SDK   = 'C:\path\to\_rocm_sdk_devel'
#   $env:IK2_INT4_DIR       = 'C:\path\to\int4-bin'
#   $env:IK2_INT4_DENSE_DIR = 'C:\path\to\int4-dense-bin'
#   $env:STRATA_CONTROL_PY  = 'C:\path\to\Strata\control\python\python.exe'
#   .\run-native-serve.ps1 -Ctx 32768 -Port 8099

param(
    [string]$Exe          = $env:IK2_EXE,
    [string]$Model        = $env:IK2_MODEL,
    [string]$Tokenizer    = $env:IK2_TOKENIZER,
    [string]$RocmSdk      = $env:ROCM_SDK,
    [string]$Int4Dir      = $env:IK2_INT4_DIR,
    [string]$Int4DenseDir = $env:IK2_INT4_DENSE_DIR,
    [string]$Python       = $env:STRATA_CONTROL_PY,
    [string]$TerminalDir  = '',
    [string]$WorkDir      = '',
    [int]$Ctx             = 32768,
    # Tokens read per prefill pass. Measured on an RX 7900 XTX at a 44,288-token prompt: 4096 -> 2,272 tok/s,
    # 7168 -> 2,518 (+10.5%). Do not raise it without checking the context: at a 640K context chunk 8192
    # collapses to ~1,435 tok/s because the bigger work buffers push the total past the VRAM budget and
    # Windows starts paging; at a 32K context the same chunk 8192 is faster than 7168 (2,463 vs 2,276).
    [int]$Chunk           = 7168,
    [int]$Port            = 8099,
    # The address the API listens on: 127.0.0.1 = this PC only (the default);
    # 0.0.0.0 = also the other devices on this network (no API key is set here,
    # so use it only on a network you trust).
    [string]$HostAddr     = '127.0.0.1',
    [string]$ConfigOut    = ''
)

$ErrorActionPreference = 'Continue'

# Resolve the repo's terminal/ directory (parent of scripts/) if not supplied.
if (-not $TerminalDir) {
    $scripts = Split-Path $PSScriptRoot -Parent
    $repo    = Split-Path $scripts -Parent
    $TerminalDir = Join-Path $repo 'terminal'
}
if (-not $WorkDir) { $WorkDir = $PSScriptRoot }

$missing = @()
foreach ($pair in @(
    ('Exe',$Exe),('Model',$Model),('Tokenizer',$Tokenizer),('RocmSdk',$RocmSdk),
    ('Int4Dir',$Int4Dir),('Int4DenseDir',$Int4DenseDir),('Python',$Python))) {
    if (-not $pair[1]) { $missing += $pair[0] }
}
if ($missing.Count) {
    Write-Output "REFUSE: missing required settings: $($missing -join ', ')"
    Write-Output "Set them as parameters or environment variables (see the header)."
    exit 1
}
foreach ($p in @($Exe,$Model,$Tokenizer,$Python)) {
    if (-not (Test-Path $p)) { Write-Output "REFUSE: path does not exist: $p"; exit 1 }
}
if (-not (Test-Path (Join-Path $TerminalDir 'serve\server.py'))) {
    Write-Output "REFUSE: terminal/serve/server.py not found under $TerminalDir"; exit 1
}

# GPU safety: refuse a second model process or a real listener on the port.
# A stopped serve leaves TIME_WAIT / orphaned Listen sockets with OwningProcess=0
# for a few seconds; those do not block a fresh bind and must not be treated busy.
$procs = Get-Process strata*,llama*,hipfire*,daemon* -ErrorAction SilentlyContinue
if ($procs) { Write-Output "REFUSE: model process already running: $($procs.Name -join ',')"; exit 1 }
$realListener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Where-Object { $_.OwningProcess -gt 4 }
if ($realListener) { Write-Output "REFUSE: port $Port already in use by pid $($realListener.OwningProcess -join ',')"; exit 1 }
$gp = Join-Path $env:USERPROFILE 'models\gpu_guard.pid'
if (Test-Path $gp) {
    $gpid = [int](Get-Content $gp)
    if (-not (Get-Process -Id $gpid -ErrorAction SilentlyContinue)) {
        Write-Output "WARN: gpu_guard pid $gpid not alive"
    }
}

# Build the engine environment: validated fast/correct ik2 settings.
$envMap = [ordered]@{
    PATH                = (Join-Path $RocmSdk 'bin') + ';' + (Join-Path $RocmSdk 'lib\llvm\bin')
    HIP_PATH            = $RocmSdk
    ROCM_PATH           = $RocmSdk
    HIP_PLATFORM        = 'amd'
    GGML_HIP_NO_VMM     = '1'
    IK2_FLASH           = '1'
    IK2_FLASH_WMMA      = '1'
    IK2_FLASH_NW        = '8'
    IK2_FLASH_BN        = '32'
    IK2_ATTN_MMA        = '0'
    IK2_EXPERT_MMA      = '0'
    IK2_EXPERTS_PREFILL_WMMA = '0'
    IK2_WMMA_GEMM       = '1'
    IK2_INT4_LAYERS     = '24'
    IK2_FIT_ACTUAL      = '1'
    IK2_GATE_FUSE       = '1'
    IK2_RMS_FUSE        = '1'
    IK2_GRAPH           = '1'
    IK2_INT4_DIR        = $Int4Dir
    IK2_INT4_DENSE_DIR  = $Int4DenseDir
    IK2_INT4_HEAD       = '1'
    IK2_INT4_HEAD_DECODE= '1'
    IK2_EC_ROOM_MIB     = '96'
    IK2_KV              = 'int4'
    IK2_KV_HOST         = '0'
    IK2_DENSE_HOST      = '0'
    IK2_ATTN_GQA        = '1'
    IK2_ATTN_DPP        = '1'
    IK2_CONV_CACHE_MIB  = '8192'
    # Emit monitor INFO keys (expert_slots, conversation_cache_*, ...) so the native
    # Strata monitor page shows the 8 GiB conversation cache and expert residency.
    IK2_MONITOR_INFO    = '1'
    IK2_INT4_PREFILL    = '1'
    IK2_LG_CTX          = '1024'
}
$cfg = [ordered]@{
    exe        = $Exe
    # --max-context: serve.py reads the engine's max context from this arg (server.py:768);
    # without it the API reports 32768 even when the engine loaded a 640k KV.
    # --conversation-cache-mib: host-RAM parking of conversation snapshots for agents.
    # --prefill: the read's chunk, -Chunk (7168 measured 10.5% faster than 4096 at a 44,288-token prompt on a 7900 XTX)
    # (2,518 vs 2,272 tok/s); 8192 falls off a cliff to ~1,435 and 10240 to ~694. Decode is unaffected.
    args       = @('--model',$Model,'--ctx',"$Ctx",'--max-context',"$Ctx",'--prefill',"$Chunk",'--gpus','1','--threads','8','--cache-gb','60','--conversation-cache-mib','8192')
    cwd        = $WorkDir
    tokenizer  = $Tokenizer
    model_name = 'inkling-2-mini'
    # The engine writes its own log here (stderr). serve reads this file for the
    # Monitor's conversation-cache card and for a start-failure hint; with no log
    # path the engine's stderr goes to the null device and both stay empty.
    log        = (Join-Path $WorkDir 'engine.log')
    env        = $envMap
    # Inkling-2 is a thinking model. Two ways to keep agent/browser answers fast:
    #  - reasoning_effort "none" -> template enable_thinking:false (no thinking pass at all):
    #    this is the clean speed default for AR-direct 141 t/s.
    #  - reasoning_budget_tokens caps thinking if a client explicitly asks to think.
    # A client can still override per-request via reasoning_effort or chat_template_kwargs.
    reasoning_effort        = 'none'
    reasoning_budget_tokens = 256
}
if (-not $ConfigOut) { $ConfigOut = Join-Path $env:TEMP 'ik2-native-serve.json' }
$json = $cfg | ConvertTo-Json -Depth 6
[System.IO.File]::WriteAllText($ConfigOut, $json, (New-Object System.Text.UTF8Encoding($false)))

$log = Join-Path $WorkDir 'native-serve.log'
$err = Join-Path $WorkDir 'native-serve.err'
$old = $env:PYTHONPATH
$env:PYTHONPATH = $TerminalDir
Push-Location $TerminalDir
Write-Output "Starting native serve: http://${HostAddr}:$Port/v1 (config $ConfigOut)"
& $Python -m serve.server --engine strata --config $ConfigOut --host $HostAddr --port $Port --api-monitor 1>$log 2>$err
$code = $LASTEXITCODE
Pop-Location
$env:PYTHONPATH = $old
Write-Output "EXIT=$code"
