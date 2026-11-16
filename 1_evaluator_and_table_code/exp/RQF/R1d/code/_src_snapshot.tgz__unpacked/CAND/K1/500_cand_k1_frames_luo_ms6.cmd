@echo off
rem CAND K1_frames (queue_candidates_2026-10-28.csv row K1_frames): LUO ms6 zero-shot, DDIM-50 eta 0, EMA, seeds 0 1 2
rem sets: val250_ouhdl (first 250 of e0_val, stored order) + test_ouhdl (4,406) in identity/mirror_x/vt (inverse applied);
rem val250_iam (m5_val_iam, 250) in mirror_x + identity (symmetric control). Code: frame_code_a44e78b0 dir (CODE_TAG 5dd59e88, tarball sha256 4c99012b).
set ROOT=$VPS_ROOT\uhwr
set PY=C:\ProgramData\miniconda3\envs\uhwr\python.exe
set STAGE=%ROOT%\stage
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set WANDB_MODE=disabled
set PYTORCH_ALLOC_CONF=expandable_segments:True
set CODE=%ROOT%\code\frame_code_a44e78b0
set PYTHONPATH=%ROOT%\winshim;%CODE%\lib;%ROOT%\repos\nagamatsu\src;%CODE%
set OUT=%ROOT%\runs\CAND\K1
if not exist %OUT% mkdir %OUT%
for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "(Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')"`) do set T0=%%T
echo K1_START_UTC %T0% code_tag=5dd59e88 >> %OUT%\job_history.txt
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total,utilization.gpu --format=csv,noheader >> %OUT%\job_history.txt
%PY% -c "import torch,sys;print('torch',torch.__version__,'cuda',torch.version.cuda,torch.cuda.get_device_name(0))" >> %OUT%\job_history.txt
%PY% %CODE%\frame_sample.py --label LUO_ms6 --run-dir exp/M5/runs/luo --milestone 6 --stage %STAGE% --sets k1 --frames identity mirror_x vt --ltr-frames mirror_x identity --seeds 0 1 2 --out-dir %OUT% --batch-size 256
set RC=%ERRORLEVEL%
for /f "usebackq delims=" %%T in (`powershell -NoProfile -Command "(Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')"`) do set T1=%%T
echo K1_END_UTC %T1% rc=%RC% >> %OUT%\job_history.txt
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader >> %OUT%\job_history.txt
exit /b %RC%
