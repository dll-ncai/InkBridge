#!/bin/bash
#SBATCH -J xs_sam31keys
#SBATCH -p cpuidle
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH -o $U/exp/FMD/xs/logs/sam31keys_%j.out
R=$U
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/sam31_keys.py
