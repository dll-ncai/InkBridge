#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=5-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o $U/exp/RQF/R1/logs/%x_%j.out
# usage: sbatch -J rqf1_gate_<JOB> -p cpuidle gate.sbatch <JOB>
bash $U/exp/RQF/code/rqf1_v1/r1_gate.sh "$@"
