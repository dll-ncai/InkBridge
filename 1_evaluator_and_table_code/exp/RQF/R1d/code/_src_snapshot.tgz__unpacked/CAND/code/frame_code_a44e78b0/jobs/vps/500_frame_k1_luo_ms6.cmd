@echo off
set ROOT=$VPS_ROOT\uhwr
set PY=C:\ProgramData\miniconda3\envs\uhwr\python.exe
set STAGE=%ROOT%\stage
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set WANDB_MODE=disabled
set PYTORCH_ALLOC_CONF=expandable_segments:True
set TAG=5dd59e88
set CODE=%ROOT%\code\frame_%TAG%
set PYTHONPATH=%ROOT%\winshim;%CODE%\lib;%ROOT%\repos\nagamatsu\src;%CODE%
set OUT=%ROOT%\runs\FRAME\k1
if not exist %OUT% mkdir %OUT%
echo K1_START_UTC %DATE% %TIME% >> %OUT%\job_history.txt
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader >> %OUT%\job_history.txt
rem K1 frame kill check: LUO ms6 zero-shot, OUHD-L val250 (first 250 of e0_val) + test 4,406 in identity/mirror_x/vt (inverse applied),
rem IAM val250 in mirror_x + identity (symmetric control); seeds 0 1 2. Scoring + summary run on CLUSTER-A (frame_k1.sh) after vps_pull.
%PY% %CODE%\frame_sample.py --label LUO_ms6 --run-dir exp/M5/runs/luo --milestone 6 --stage %STAGE% --sets k1 --frames identity mirror_x vt --ltr-frames mirror_x identity --seeds 0 1 2 --out-dir %OUT% --batch-size 256
set RC=%ERRORLEVEL%
echo K1_END_UTC %DATE% %TIME% rc=%RC% >> %OUT%\job_history.txt
exit /b %RC%
