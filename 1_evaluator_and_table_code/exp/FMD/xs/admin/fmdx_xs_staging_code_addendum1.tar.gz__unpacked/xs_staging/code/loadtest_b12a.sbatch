#!/bin/bash
#SBATCH -J xs_lt_b12a
#SBATCH -p cpuidle
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH -o $U/exp/FMD/xs/logs/loadtest_b12a_%j.out
R=$U
export XS_NONE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/xs_probe.py load qari-ocr-0.4.0-vl-4b-vision arabic-english-hw-ocr-v3 arabic-hw-ocr-4bit qwen3-vl-4b-instruct qwen2.5-vl-3b-instruct dinov2-giant dinov2-base aimv2-large-patch14-224-distilled
