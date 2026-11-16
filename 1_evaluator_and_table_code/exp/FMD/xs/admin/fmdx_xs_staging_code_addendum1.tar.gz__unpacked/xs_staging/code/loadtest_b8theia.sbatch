#!/bin/bash
#SBATCH -J xs_lt_b8theia
#SBATCH -p cpuidle
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH -o $U/exp/FMD/xs/logs/loadtest_b8theia_%j.out
R=$U
export XS_DS=1 HF_HOME=$U/exp/FMD/xs/weights/.hf_home PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/xs_probe.py load theia-tiny-cddsv theia-tiny-cdiv theia-small-cdiv theia-base-cdiv theia-base-cddsv
