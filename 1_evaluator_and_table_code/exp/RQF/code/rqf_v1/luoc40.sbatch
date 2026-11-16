#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=10-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o $U/exp/RQF/logs/%x_%j.out
# RQF R2b LUOC40: LUO ms6 (model+EMA, fresh optimiser) + 40,000 steps of the ORIGINAL LUO trainer (m5_train.py --init-from) on m5_luo_train, seed 20261016, save every 10k, val250 x 4 LUO sets selection (= WARM rule).
set -u
export TAG=e33511dd RUNROOT=$U/exp/RQF/runs ARM=LUOC40_bb STEPS=40000 SAVE=10000 VALIDATION_EVERY=1000 EVAL_MAX=0 NGPU=2
echo "$(date -u +%FT%TZ) job $SLURM_JOB_ID host $(hostname) partition $SLURM_JOB_PARTITION restart ${SLURM_RESTART_COUNT:-0}"
bash $U/exp/RQF/code/rqf_v1/luoc40_arm.sh
rc=$?; echo "$(date -u +%FT%TZ) luoc40_arm.sh exit $rc"; exit $rc
