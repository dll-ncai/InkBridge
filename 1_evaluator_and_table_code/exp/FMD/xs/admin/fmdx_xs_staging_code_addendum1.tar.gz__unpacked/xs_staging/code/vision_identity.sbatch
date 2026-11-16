#!/bin/bash
#SBATCH -J xs_visid
#SBATCH -p cpuidle
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o $U/exp/FMD/xs/logs/vision_identity_%j.out
R=$U
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/vision_identity.py
