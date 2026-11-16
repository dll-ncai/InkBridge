#!/bin/bash
#SBATCH -J xs_lt_b7flux
#SBATCH -p cpuidle
#SBATCH -c 8
#SBATCH --mem=160G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH -o $U/exp/FMD/xs/logs/loadtest_b7flux_%j.out
R=$U
export XS_LOAD_N=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/xs_probe.py load flux1-schnell flux1-dev
