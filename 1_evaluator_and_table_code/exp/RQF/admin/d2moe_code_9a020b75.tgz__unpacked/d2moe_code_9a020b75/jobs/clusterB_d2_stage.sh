#!/bin/bash
# D2 on CLUSTER-B <site> (Pyxis). Submit: sbatch --parsable -A <account> -p H100-RP,H100 --gpus=1 --cpus-per-task=8 --mem=64G --time=1-00:00:00 \
#   -J d2moe_s1 -o $U_B/exp/D2_MOE/clusterB/jobs/s1_%j.out --requeue --export=ALL,TAG=<sha8>,STAGE=1 \
#   $U_B/exp/D2_MOE/code/d2moe_code_<sha8>/jobs/clusterB_d2_stage.sh      (STAGE=2,P=p10,S1=... for stage 2)
# 1-day limit: exits 8 when unfinished; resubmit the same command to resume.
# D2 adapter experts + gate, 1 GPU (cap >= 8.0). Env: TAG=<sha8> STAGE=1|2 [RUN=...] [S1=stage-1 ckpt] [TRAIN=npz] [P=p10].
# Requeue-safe (auto-resume from --run-dir).
set -eo pipefail
B=$U_B; IMG=$B/images/pytorch_2.7.1-cuda12.8-cudnn9-runtime.sqsh; VENV=$B/venvs/e0; REPO=$B/repos/nagamatsu
TAG=${TAG:?set TAG=<sha8>}; CODE=$B/exp/D2_MOE/code/d2moe_code_$TAG; D=$B/data/m5_multiscript_v1
STAGE=${STAGE:-1}; BB=$B/exp/M5/runs/joint/checkpoints/checkpoint_000009.pt
mkdir -p $B/exp/D2_MOE/clusterB/jobs
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' '); awk -v c="$cc" 'BEGIN{exit !(c+0 >= 8.0)}' || { echo "compute capability $cc < 8.0: abort"; exit 4; }
(cd $CODE && sha256sum -c --quiet SHA256SUMS.txt) || { echo "code differs from SHA256SUMS.txt"; exit 3; }
CT="srun --ntasks=1 --overlap --container-image=$IMG --container-name=d2moe_${SLURM_JOB_ID:-x} --container-mounts=$SCRATCH_B:$SCRATCH_B,/tmp:/tmp"; export CUDA_DEVICE_ORDER=PCI_BUS_ID
PYG="$CT env PYTHONPATH=$REPO/src:$CODE OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled $VENV/bin/python"
VAL="casia_zh=$D/m5_val_casia_zh.npz casia_en=$D/m5_val_casia_en.npz hpl_tamil=$D/m5_val_hpl_tamil.npz iam=$D/m5_val_iam.npz ouhdl=$D/m5_val_ouhdl.npz"
if [ "$STAGE" = 1 ]; then
  RUN=${RUN:-$B/exp/D2_MOE/clusterB/runs/stage1_e3r16}; mkdir -p $RUN
  echo "$(date -u +%FT%TZ) start job=$SLURM_JOB_ID host=$(hostname) restart=${SLURM_RESTART_COUNT:-0} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" >> $RUN/job_history.txt
  $PYG $CODE/d2moe_train.py --stage 1 --run-dir $RUN --backbone $BB --backbone-weights ema \
    --train-npz $D/m5_joint_train.npz --scripts casia_zh casia_en hpl_tamil iam --val-npz $VAL \
    --n-experts 3 --rank 16 --lora-alpha 16 --lora-targets cross,attn,linear --gate-hidden 128 --gate-feat deepest --lb-coef 0.01 \
    --batch-size 64 --grad-accum 4 --train-steps ${STEPS:-20000} --save-every 1000 --val-every 1000 --learning-rate 1e-4 --warmup-steps 500 \
    --keep-last 5 --keep-every 5000 --num-workers 4 --seed ${SEED:-20261024}
else
  P=${P:-p10}; S1=${S1:-$B/exp/D2_MOE/clusterB/runs/stage1_e3r16/checkpoints/checkpoint_000020.pt}
  if [ "$P" = p100 ]; then TRAIN=${TRAIN:-$B/data/ouhdl_e0_v3/e0_train.npz}; else TRAIN=${TRAIN:-$D/m5_ft_ouhdl_$P.npz}; fi
  RUN=${RUN:-$B/exp/D2_MOE/clusterB/runs/stage2_${P}_${OLD:-frozen}}; mkdir -p $RUN
  echo "$(date -u +%FT%TZ) start job=$SLURM_JOB_ID host=$(hostname) restart=${SLURM_RESTART_COUNT:-0} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" >> $RUN/job_history.txt
  $PYG $CODE/d2moe_train.py --stage 2 --run-dir $RUN --init-from-stage1 $S1 --backbone $BB --backbone-weights ema \
    --train-npz $TRAIN --script-override ouhdl --scripts ouhdl --val-npz $VAL \
    --new-expert-name rtl --new-expert-init gate_mix --gate-init-sample 256 --old-experts ${OLD:-frozen} --train-gate --lb-coef 0.01 \
    --batch-size 64 --grad-accum 4 --train-steps ${STEPS:-10000} --save-every 1000 --val-every 1000 --learning-rate 1e-4 --warmup-steps 200 \
    --keep-last 5 --keep-every 5000 --num-workers 4 --seed ${SEED:-20261024}
fi
[ -f $RUN/training_complete.json ] || { echo "not complete (preempted?)"; exit 8; }
