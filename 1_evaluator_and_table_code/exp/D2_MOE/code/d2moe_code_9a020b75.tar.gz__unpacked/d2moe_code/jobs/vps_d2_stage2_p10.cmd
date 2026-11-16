@echo off
REM D2 stage 2: add the 'rtl' expert (gate-weighted mix of the stage-1 experts) and train it on the 10 % OUHD-L subset
REM (m5_ft_ouhdl_p10.npz = M5 FT subset; p01 / p30 analogous; p100 = stage\data\ouhdl_e0_v3\e0_train.npz). Old experts frozen.
set ROOT=$VPS_ROOT\uhwr
set PY=C:\ProgramData\miniconda3\envs\uhwr\python.exe
set STAGE=%ROOT%\stage
set CODE=%ROOT%\code\d2moe_code
set PYTHONPATH=%ROOT%\winshim;%ROOT%\repos\nagamatsu\src;%CODE%
set PYTHONIOENCODING=utf-8
set WANDB_MODE=disabled
set PYTORCH_ALLOC_CONF=expandable_segments:True
set D=%STAGE%\data\m5_multiscript_v1
set S1=%ROOT%\runs\D2_MOE\stage1_e3r16\checkpoints\checkpoint_000020.pt
set RUN=%ROOT%\runs\D2_MOE\stage2_p10_frozen
%PY% %CODE%\d2moe_train.py --stage 2 --run-dir %RUN% --init-from-stage1 %S1% ^
  --backbone %STAGE%\exp\M5\runs\joint\checkpoints\checkpoint_000009.pt --backbone-weights ema ^
  --train-npz %D%\m5_ft_ouhdl_p10.npz --script-override ouhdl --scripts ouhdl ^
  --val-npz ouhdl=%D%\m5_val_ouhdl.npz casia_zh=%D%\m5_val_casia_zh.npz casia_en=%D%\m5_val_casia_en.npz hpl_tamil=%D%\m5_val_hpl_tamil.npz iam=%D%\m5_val_iam.npz ^
  --new-expert-name rtl --new-expert-init gate_mix --gate-init-sample 256 --old-experts frozen --train-gate --lb-coef 0.01 ^
  --batch-size 64 --grad-accum 4 --train-steps 10000 --save-every 1000 --val-every 1000 --learning-rate 1e-4 --warmup-steps 200 ^
  --keep-last 5 --keep-every 5000 --num-workers 0 --seed 20261024
exit /b %ERRORLEVEL%
