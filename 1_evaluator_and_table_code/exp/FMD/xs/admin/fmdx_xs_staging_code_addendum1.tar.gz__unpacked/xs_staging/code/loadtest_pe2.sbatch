#!/bin/bash
#SBATCH -J xs_lt_pe2
#SBATCH -p cpuidle
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o $U/exp/FMD/xs/logs/loadtest_pe2_%j.out
R=$U
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/xs_probe.py load pe-spatial-b16-512 pe-spatial-l14-448 pe-spatial-g14-448
