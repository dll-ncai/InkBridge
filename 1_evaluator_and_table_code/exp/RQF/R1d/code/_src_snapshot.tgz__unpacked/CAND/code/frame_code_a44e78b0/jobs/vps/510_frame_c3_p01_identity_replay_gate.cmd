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
set D=%STAGE%\data\m5_multiscript_v1
set OU=%STAGE%\data\ouhdl_e0_v3
set DE=%STAGE%\data\d2_v2\unipen_de
set INIT=%STAGE%\exp\M5\runs\luo\checkpoints\checkpoint_000006.pt
set FRAME=identity
set RUN=%ROOT%\runs\FRAME\c3_p01_identity_replay
if not exist %INIT% (echo MISSING LUO checkpoint %INIT% & exit /b 7)
if not exist %RUN% mkdir %RUN%
echo FRAME_ARM_START_UTC %DATE% %TIME% arm=c3_p01_identity_replay frac=p01 frame=identity replay=1 >> %RUN%\job_history.txt
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader >> %RUN%\job_history.txt
rem Route A LUO-FT recipe (init LUO ms6, batch 128 x accum 4 = 512, lr 8e-5, 10k steps, ckpt + val loss every 1k) + frame/replay/gate.
rem The job auto-resumes from %RUN%\checkpoints; on GATE_FAIL e0_train.py exits 0 and the sampling below is skipped.
%PY% %CODE%\e0_train.py --e0-train-npz %D%\m5_ft_ouhdl_p01.npz --condition rendered --run-dir %RUN% --init-from %INIT% ^
 --frame %FRAME% --frame-scripts ouhdl --replay-npz %D%\m5_luo_train.npz --urdu-share 0.25 ^
 --gate-step 4000 --gate-gold-npz %OU%\e0_gold.npz --gate-unipen-npz %DE%\e0_test.npz --gate-label c3_p01_identity_replay_ms4 ^
 --batch-size 128 --gradient-accumulate-every 4 --learning-rate 8e-5 --train-steps 10000 --save-every 1000 --validation-every 1000 ^
 --sequence-length 320 --image-size 64 --sampling-timesteps 50 --train-fraction 0.99 --num-workers 0 ^
 --wandb-mode disabled --random-seed 20261016 --sample-count 25 --classes 0 --notes frame_c3_p01_identity_replay
set RC=%ERRORLEVEL%
echo FRAME_ARM_TRAIN_END_UTC %DATE% %TIME% rc=%RC% >> %RUN%\job_history.txt
if not "%RC%"=="0" exit /b %RC%
if exist %RUN%\GATE_FAIL (echo GATE_FAIL >> %RUN%\job_history.txt & exit /b 0)
rem val1000 sampling of every milestone in the arm's frame (selection = lowest native LDTW, scored on CLUSTER-A after vps_pull)
for /L %%m in (1,1,9) do if exist %RUN%\checkpoints\checkpoint_00000%%m.pt %PY% %CODE%\frame_sample.py --label c3_p01_identity_replay_ms%%m --run-dir %RUN% --milestone %%m --stage %STAGE% --sets val1000_ouhdl --frames %FRAME% --seeds 0 --out-dir %RUN%\eval --batch-size 256
if exist %RUN%\checkpoints\checkpoint_000010.pt %PY% %CODE%\frame_sample.py --label c3_p01_identity_replay_ms10 --run-dir %RUN% --milestone 10 --stage %STAGE% --sets val1000_ouhdl --frames %FRAME% --seeds 0 --out-dir %RUN%\eval --batch-size 256
echo FRAME_ARM_VAL_SAMPLED_UTC %DATE% %TIME% >> %RUN%\job_history.txt
exit /b %ERRORLEVEL%
