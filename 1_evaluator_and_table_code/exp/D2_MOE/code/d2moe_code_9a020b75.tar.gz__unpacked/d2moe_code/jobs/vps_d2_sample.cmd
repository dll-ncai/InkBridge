@echo off
REM D2 sampling (DDIM-50, EMA adapters + learned gate); edit RUN / MS / NPZ / OUT; add --force-expert NAME for oracle-expert runs
set ROOT=$VPS_ROOT\uhwr
set PY=C:\ProgramData\miniconda3\envs\uhwr\python.exe
set STAGE=%ROOT%\stage
set CODE=%ROOT%\code\d2moe_code
set PYTHONPATH=%ROOT%\winshim;%ROOT%\repos\nagamatsu\src;%CODE%
set PYTHONIOENCODING=utf-8
set RUN=%ROOT%\runs\D2_MOE\stage2_p10_frozen
set MS=latest
set NPZ=%STAGE%\data\m5_multiscript_v1\m5_val_ouhdl.npz
set OUT=%RUN%\preds\val_ouhdl_ms%MS%.npz
if not exist %RUN%\preds mkdir %RUN%\preds
%PY% %CODE%\d2moe_sample.py --run-dir %RUN% --milestone %MS% --backbone %STAGE%\exp\M5\runs\joint\checkpoints\checkpoint_000009.pt --npz %NPZ% --sampling-timesteps 50 --ddim-eta 0 --batch-size 256 --seed 0 --out %OUT%
exit /b %ERRORLEVEL%
