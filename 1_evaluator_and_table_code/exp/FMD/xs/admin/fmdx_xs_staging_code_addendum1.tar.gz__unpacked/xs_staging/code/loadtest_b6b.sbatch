#!/bin/bash
#SBATCH -J xs_lt_b6b
#SBATCH -p cpuidle
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH -o $U/exp/FMD/xs/logs/loadtest_b6b_%j.out
R=$U
export XS_NONE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/xs_probe.py load siglip2-large-patch16-256 siglip2-so400m-patch16-256 dinov3-vitl16-sat493m webssl-dino3b-224 webssl-dino7b-224
