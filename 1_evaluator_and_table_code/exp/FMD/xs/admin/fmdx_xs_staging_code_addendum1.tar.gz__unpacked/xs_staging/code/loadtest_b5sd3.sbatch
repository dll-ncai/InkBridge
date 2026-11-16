#!/bin/bash
#SBATCH -J xs_lt_b5sd3
#SBATCH -p cpuidle
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH -o $U/exp/FMD/xs/logs/loadtest_b5sd3_%j.out
R=$U
export XS_LOAD_N=2 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/xs_probe.py load sd3-medium-diffusers
