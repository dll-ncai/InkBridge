@echo off
REM D2 stage 1: 3 LoRA experts + gate on the frozen M5 joint ms9 backbone, non-Urdu scripts (VPS queue runner job; re-queue to resume)
set ROOT=$VPS_ROOT\uhwr
set PY=C:\ProgramData\miniconda3\envs\uhwr\python.exe
set STAGE=%ROOT%\stage
set CODE=%ROOT%\code\d2moe_code
set PYTHONPATH=%ROOT%\winshim;%ROOT%\repos\nagamatsu\src;%CODE%
set PYTHONIOENCODING=utf-8
set WANDB_MODE=disabled
set PYTORCH_ALLOC_CONF=expandable_segments:True
set D=%STAGE%\data\m5_multiscript_v1
set RUN=%ROOT%\runs\D2_MOE\stage1_e3r16
%PY% %CODE%\d2moe_train.py --stage 1 --run-dir %RUN% ^
  --backbone %STAGE%\exp\M5\runs\joint\checkpoints\checkpoint_000009.pt --backbone-weights ema ^
  --train-npz %D%\m5_joint_train.npz --scripts casia_zh casia_en hpl_tamil iam ^
  --val-npz casia_zh=%D%\m5_val_casia_zh.npz casia_en=%D%\m5_val_casia_en.npz hpl_tamil=%D%\m5_val_hpl_tamil.npz iam=%D%\m5_val_iam.npz ouhdl=%D%\m5_val_ouhdl.npz ^
  --n-experts 3 --rank 16 --lora-alpha 16 --lora-targets cross,attn,linear --gate-hidden 128 --gate-feat deepest --lb-coef 0.01 ^
  --batch-size 64 --grad-accum 4 --train-steps 20000 --save-every 1000 --val-every 1000 --learning-rate 1e-4 --warmup-steps 500 ^
  --keep-last 5 --keep-every 5000 --num-workers 0 --seed 20261024
exit /b %ERRORLEVEL%
