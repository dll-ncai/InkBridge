#!/bin/bash
#SBATCH -J xs_lt_vlm
#SBATCH -p cpuidle
#SBATCH -c 16
#SBATCH --mem=110G
#SBATCH -t 03:00:00
#SBATCH -o $U/exp/FMD/xs/logs/loadtest_vlm_%j.out
R=$U
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=16 HF_HUB_OFFLINE=1
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/xs_probe.py load qwen2-vl-2b-instruct deepseek-ocr
