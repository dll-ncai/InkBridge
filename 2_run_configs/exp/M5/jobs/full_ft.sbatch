#!/bin/bash
#SBATCH --job-name=m5_ft
#SBATCH --partition=b200idle
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --output=$U/exp/M5/logs/%x_%j.out
export TAG=e33511dd RUNROOT=$U/exp/M5/runs FT_STEPS=10000 FT_SAVE=1000 EVAL_MAX=0 VAL_SUBSET=1000 ARMS="p01 p10 p30 p100"
bash $U/exp/M5/code_e33511dd/m5_ft.sh
