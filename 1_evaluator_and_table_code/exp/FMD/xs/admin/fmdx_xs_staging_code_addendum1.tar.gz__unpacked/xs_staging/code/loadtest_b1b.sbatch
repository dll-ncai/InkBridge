#!/bin/bash
#SBATCH -J xs_lt_b1b
#SBATCH -p cpuidle
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH -o $U/exp/FMD/xs/logs/loadtest_b1b_%j.out
R=$U
export XS_NONE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/xs_probe.py load florence-2-base florence-2-large-ft paddleocr-vl-1.5 paddleocr-vl-1.0 trocr-base-handwritten tipsv1-s14 tipsv1-b14 webssl-dino300m-224
