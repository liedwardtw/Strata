# bench.ps1 -- one-click Inkling-2-Mini benchmark on Strata ik2 (HIP gfx1100).
# ASCII only (PowerShell 5.1 compatible). All paths come from parameters / environment.
#
# Usage examples:
#   .\bench.ps1 -Exe <exe> -Model <gguf> -PromptFile <ids.txt> -RocmSdk <sdk> -Draft <mtp.safetensors> -Mode dspark
#   .\bench.ps1 -Exe <exe> -Model <gguf> -PromptFile <ids.txt> -RocmSdk <sdk> -Mode ar -PromptKind structured -Gen 256
#   .\bench.ps1 -Exe <exe> -Model <gguf> -PromptFile <ids.txt> -RocmSdk <sdk> -Mode longctx -Ctx 1052672 -Gen 64
#
# Prompt files are comma-separated token-id files (see tools/ + README for how to produce them).

param(
    [Parameter(Mandatory=$true)][string]$Exe,
    [Parameter(Mandatory=$true)][string]$Model,
    [Parameter(Mandatory=$true)][string]$PromptFile,
    [Parameter(Mandatory=$true)][string]$RocmSdk,
    [string]$Draft = '',
    [ValidateSet('ar','dspark','longctx')][string]$Mode = 'ar',
    [ValidateSet('generic','structured')][string]$PromptKind = 'generic',
    [int]$Gen = 128,
    [int]$Ctx = 20000,
    [int]$Chunk = 9216,
    [string]$Kv = 'f16',
    [string]$Int4Dir = 'int4-bin',
    [string]$Int4DenseDir = 'int4-dense-bin',
    [string]$LogDir = '.',
    [string]$Tag = 'bench'
)
$ErrorActionPreference = 'Continue'

$env:HIP_PATH = $RocmSdk; $env:ROCM_PATH = $RocmSdk; $env:HIP_PLATFORM = 'amd'
$env:PATH = "$RocmSdk\bin;$RocmSdk\lib\llvm\bin;$env:PATH"
$env:GGML_HIP_NO_VMM = '1'

# Core validated correct + fast config (WMMA flash attention + INT4 experts/dense, all default ON).
$env:IK2_FLASH='1'; $env:IK2_FLASH_WMMA='1'; $env:IK2_FLASH_NW='8'; $env:IK2_FLASH_BN='32'
$env:IK2_ATTN_MMA='0'; $env:IK2_EXPERT_MMA='0'; $env:IK2_EXPERTS_PREFILL_WMMA='0'; $env:IK2_WMMA_GEMM='1'
$env:IK2_KV=$Kv; $env:IK2_INT4_DIR=$Int4Dir; $env:IK2_INT4_DENSE='1'; $env:IK2_DENSE='q8'
$env:IK2_INT4_DENSE_DIR=$Int4DenseDir
Remove-Item Env:\IK2_PROF,Env:\IK2_FORCE,Env:\IK2_GRAPH,Env:\IK2_VPROF2,Env:\IK2_VPROF3 -ErrorAction SilentlyContinue

if ($Mode -eq 'dspark') {
    if (-not $Draft) { Write-Host 'ERROR: -Mode dspark requires -Draft <mtp.safetensors>'; exit 1 }
    $env:IK2_VERIFY_FAST='1'; $env:IK2_DRAFT_WMMA='1'; $env:IK2_INT4_DENSE_MULTI='1'
    $env:IK2_MV_SLOT_MAJOR='1'; $env:IK2_INT4_HEAD='1'; $env:IK2_INT4_MSG='1'
    $env:IK2_INT4_DENSE_MV='1'; $env:IK2_MV_NW='6'; $env:IK2_DRAFT_HEAD_INT4='1'
    $extraArgs = @('--draft', $Draft)
} elseif ($Mode -eq 'longctx') {
    # 1M-class context: int4 KV kept in VRAM, dense sidecars, int4 head decode.
    $env:IK2_KV='int4'; $env:IK2_INT4_LAYERS='24'; $env:IK2_FIT_ACTUAL='1'
    $env:IK2_DENSE_HOST='1'; $env:IK2_INT4_PREFILL='1'
    $env:IK2_RESERVE_MIB='0'; $env:IK2_EC_ROOM_MIB='96'; $env:IK2_LG_CTX='1024'
    $env:IK2_INT4_HEAD='1'; $env:IK2_INT4_HEAD_DECODE='1'
    $extraArgs = @()
} else {
    if ($PromptKind -eq 'structured') { $env:IK2_VERIFY_FAST='1' }
    $extraArgs = @()
}

$log = Join-Path $LogDir "$Tag.log"; $errlog = Join-Path $LogDir "$Tag.err.log"
$args = @('--model',$Model,'--tokens',"@$PromptFile",'--gen',"$Gen",'--ctx',"$Ctx",'--prefill',"$Chunk",'--gpus','1') + $extraArgs
Write-Host "mode=$Mode model=$Model prompt=$PromptFile gen=$Gen ctx=$Ctx chunk=$Chunk kv=$($env:IK2_KV)"
$p = Start-Process -FilePath $Exe -ArgumentList $args -RedirectStandardOutput $log -RedirectStandardError $errlog -Wait -PassThru -NoNewWindow
Write-Host "EXIT=$($p.ExitCode)"
Get-Content $log | Select-String 'prompt:|generation:|speculative:|resident|generated:'
Write-Host "full log: $log"
