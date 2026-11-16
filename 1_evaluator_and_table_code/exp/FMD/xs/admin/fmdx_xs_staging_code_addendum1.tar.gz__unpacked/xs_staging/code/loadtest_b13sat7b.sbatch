#!/bin/bash
#SBATCH -J xs_lt_b13sat7b
#SBATCH -p cpuidle
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH -o $U/exp/FMD/xs/logs/loadtest_b13sat7b_%j.out
R=$U
export XS_NONE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/xs_probe.py load dinov3-vit7b16-sat493m
