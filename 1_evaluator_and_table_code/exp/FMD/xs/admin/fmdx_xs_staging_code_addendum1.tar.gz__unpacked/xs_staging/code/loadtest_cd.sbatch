#!/bin/bash
#SBATCH -J xs_lt_cd
#SBATCH -p cpuidle
#SBATCH -c 16
#SBATCH --mem=110G
#SBATCH -t 03:00:00
#SBATCH -o $U/exp/FMD/xs/logs/loadtest_cd_%j.out
R=$U
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=16 HF_HUB_OFFLINE=1
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/xs_probe.py load cleandift-sd21 cleandift-sd15
