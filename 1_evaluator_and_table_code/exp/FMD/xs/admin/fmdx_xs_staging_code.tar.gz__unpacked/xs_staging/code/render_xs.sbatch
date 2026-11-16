#!/bin/bash
#SBATCH -J xs_render
#SBATCH -p cpuidle
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o $U/exp/FMD/xs/logs/render_xs_%j.out
R=$U
export PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 PYTHONPATH=$R/shared/uhwr_eval-0.3.1
apptainer exec -B $SCRATCH_ROOT -B /tmp $R/images/torch27.sif $R/venvs/uhwr_eval02/bin/python $R/exp/FMD/xs/code/render_xs.py
