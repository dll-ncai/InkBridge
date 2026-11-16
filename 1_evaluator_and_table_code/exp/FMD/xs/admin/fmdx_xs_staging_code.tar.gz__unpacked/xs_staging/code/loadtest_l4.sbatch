#!/bin/bash
#SBATCH -J xs_lt_l4
#SBATCH -p cpuidle
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o $U/exp/FMD/xs/logs/loadtest_l4_%j.out
R=$U
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/xs_probe.py load depth-anything-v2-large pe-core-b16-224 pe-core-l14-336 siglip2-giant-opt-patch16-256 siglip2-giant-opt-patch16-384 dinov3-vit7b16 qwen2-vl-2b-instruct tipsv2-b14 tipsv2-l14 tipsv2-so400m14
