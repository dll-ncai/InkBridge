#!/bin/bash
#SBATCH --job-name=m5_joint
#SBATCH --partition=b200idle
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=5-00:00:00
#SBATCH --requeue
#SBATCH --output=$U/exp/M5/logs/%x_%j.out
export TAG=e33511dd RUNROOT=$U/exp/M5/runs ARM=joint STEPS=100000 SAVE=10000 VALIDATION_EVERY=1000 EVAL_MAX=0 NGPU=2
bash $U/exp/M5/code_e33511dd/m5_arm.sh
