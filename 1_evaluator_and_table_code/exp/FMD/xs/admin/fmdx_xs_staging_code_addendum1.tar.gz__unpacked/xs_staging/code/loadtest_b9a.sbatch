#!/bin/bash
#SBATCH -J xs_lt_b9a
#SBATCH -p cpuidle
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH -o $U/exp/FMD/xs/logs/loadtest_b9a_%j.out
R=$U
export XS_NONE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/xs_probe.py load webssl-mae1b-224 webssl-mae3b-224 aimv2-huge-patch14-224 dinov2-reg-small dinov2-reg-base dinov2-reg-giant qari-ocr-v0.3-vl-2b
