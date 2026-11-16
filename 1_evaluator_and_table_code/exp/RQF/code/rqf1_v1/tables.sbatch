#!/bin/bash
#SBATCH --partition=epyc-256
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -o $U/exp/RQF/R1/logs/%x_%j.out
bash $U/exp/RQF/code/rqf1_v1/r1_tables.sh
