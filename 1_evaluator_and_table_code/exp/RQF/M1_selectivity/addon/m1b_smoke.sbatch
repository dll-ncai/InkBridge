#!/bin/bash
#SBATCH -J rqf_m1b_smoke
#SBATCH -p haswell-64s
#SBATCH -N 1 -n 1 --cpus-per-task=8 --mem=32G --time=01:00:00
#SBATCH -o $U/exp/RQF/M1_selectivity/slurm-%j.out
bash $U/exp/RQF/M1_selectivity/addon/run_m1.sh smoke
