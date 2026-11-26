@echo off
chcp 65001 >nul
REM One-click: Native Strata serve driving ik2 (Inkling-2-Mini) on RX 7900 XTX.
REM Before first use, set the IK2_* / ROCM_SDK / STRATA_CONTROL_PY env vars
REM (see README.md in this folder). API: http://127.0.0.1:8099/v1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run-native-serve.ps1"
pause
