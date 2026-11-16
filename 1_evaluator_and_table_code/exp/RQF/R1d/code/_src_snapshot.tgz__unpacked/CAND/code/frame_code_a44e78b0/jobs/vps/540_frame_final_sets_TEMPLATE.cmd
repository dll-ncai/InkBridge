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
rem TEMPLATE (copy, fill ARM / FRAME / MS after CLUSTER-A selection): final sets with the selected milestone, Urdu sets in the arm's
rem frame (inverse applied), UNIPEN-de + IAM val250 in identity; seeds 0 1 2. Scoring on CLUSTER-A (e0_score.py) after vps_pull.
set ARM=__ARM__
set FRAME=__FRAME__
set MS=__MS__
set RUN=%ROOT%\runs\FRAME\%ARM%
echo FRAME_FINAL_START_UTC %DATE% %TIME% arm=%ARM% ms=%MS% >> %RUN%\job_history.txt
%PY% %CODE%\frame_sample.py --label %ARM%_ms%MS% --run-dir %RUN% --milestone %MS% --stage %STAGE% --sets test_ouhdl gold_ouhdl german_unipen_de_test val250_iam --frames %FRAME% --ltr-frames identity --seeds 0 1 2 --out-dir %RUN%\eval --batch-size 256
set RC=%ERRORLEVEL%
echo FRAME_FINAL_END_UTC %DATE% %TIME% rc=%RC% >> %RUN%\job_history.txt
exit /b %RC%
