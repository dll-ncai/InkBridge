#!/bin/bash
#SBATCH -J xs_stage_fix
#SBATCH -p cpuidle
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o $U/exp/FMD/xs/logs/stage_fix_%j.out
R=$U
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/stage_xs_v2b.py extras vit-mae-large trocr-large-handwritten
