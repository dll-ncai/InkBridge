#!/bin/bash
#SBATCH -J xs_stage_ocr1b
#SBATCH -p cpuidle
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --requeue
#SBATCH -o $U/exp/FMD/xs/logs/stage_ocr1b_%j.out
R=$U
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/stage_xs_v3.py add_ocr1 dots-ocr dots-ocr-base chandra
