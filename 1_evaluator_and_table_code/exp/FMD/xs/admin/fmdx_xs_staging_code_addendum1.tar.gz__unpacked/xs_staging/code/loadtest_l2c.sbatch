#!/bin/bash
#SBATCH -J xs_lt_l2c
#SBATCH -p cpuidle
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o $U/exp/FMD/xs/logs/loadtest_l2c_%j.out
R=$U
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/xs_probe.py load dinov2-reg-large webssl-dino1b-224 webssl-mae300m-224 vit-mae-large depth-anything-v2-large aimv2-large-patch14-224 trocr-large-handwritten
