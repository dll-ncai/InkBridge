@echo off
REM FMDX warm-monitor dispatcher (monitor track, <agent>). Reason: <author> decision 3 Nov ~23:50Z, relayed by the lead. Loops over dispatch\ready\*.cmd in name order; exits at EXIT (ready empty) or 2026-11-07T06:00Z.
powershell -NoProfile -ExecutionPolicy Bypass -File $VPS_ROOT\uhwr\stage\exp\FMD\xs\code\fmdx_monitor_v1\fmdx_dispatcher.ps1
exit /b %ERRORLEVEL%
