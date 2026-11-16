#!/bin/bash
#SBATCH -J xs_reverify
#SBATCH -p cpuidle
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 01:30:00
#SBATCH -o $U/exp/FMD/xs/logs/reverify_%j.out
python3 $U/exp/FMD/xs/code/reverify_5ids.py
