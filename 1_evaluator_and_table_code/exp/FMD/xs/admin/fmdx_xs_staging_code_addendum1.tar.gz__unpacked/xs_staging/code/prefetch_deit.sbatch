#!/bin/bash
#SBATCH -J xs_prefetch
#SBATCH -p cpuidle
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH --requeue
#SBATCH -o $U/exp/FMD/xs/logs/prefetch_%j.out
R=$U
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/prefetch_deit.py
