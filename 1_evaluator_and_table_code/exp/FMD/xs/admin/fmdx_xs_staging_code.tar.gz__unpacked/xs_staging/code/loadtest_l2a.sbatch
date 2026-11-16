#!/bin/bash
#SBATCH -J xs_lt_l2a
#SBATCH -p cpuidle
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o $U/exp/FMD/xs/logs/loadtest_l2a_%j.out
R=$U
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/xs_probe.py load c-radiov4-h c-radiov4-so400m pe-spatial-g14-448 pe-spatial-s16-512 pe-spatial-t16-512 tipsv2-g14 starvector-1b-im2svg
