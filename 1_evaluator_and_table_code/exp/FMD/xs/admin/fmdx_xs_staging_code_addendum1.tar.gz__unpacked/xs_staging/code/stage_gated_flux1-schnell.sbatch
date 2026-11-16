#!/bin/bash
#SBATCH -J xs_gated_flux1-schnell
#SBATCH -p cpuidle
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --requeue
#SBATCH -o $U/exp/FMD/xs/logs/gated_flux1-schnell_%j.out
R=$U
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/stage_xs.py gated flux1-schnell
