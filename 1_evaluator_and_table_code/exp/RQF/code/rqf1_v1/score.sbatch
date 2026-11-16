#!/bin/bash
#SBATCH --partition=epyc-256
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH -o $U/exp/RQF/R1/logs/%x_%A_%a.out
# usage: TASKS=<file> sbatch -J rqf1_score_<JOB> --array=1-N --dependency=afterok:<id> --export=ALL,TASKS score.sbatch
bash $U/exp/RQF/code/rqf1_v1/r1_score.sh
