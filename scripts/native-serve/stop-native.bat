@echo off
chcp 65001 >nul
REM One-click stop: native Strata serve + ik2 engine (gpu_guard stays).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop-native.ps1"
pause
