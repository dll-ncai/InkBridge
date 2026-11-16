#!/bin/bash
#SBATCH -J xs_lt_b4a
#SBATCH -p cpuidle
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH -o $U/exp/FMD/xs/logs/loadtest_b4a_%j.out
R=$U
export XS_NONE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/xs_probe.py load pe-lang-l14-448 pe-lang-g14-448 tipsv1-g14 dots-ocr dots-ocr-base depth-anything-3-giant-1.1 depth-anything-3-base
