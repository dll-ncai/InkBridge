#!/bin/bash
#SBATCH -J fmdx_controller
#SBATCH -p cpuidle
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=8-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o $U/exp/FMD/xs/logs/controller/%x_%j.out
# FMDX gate-and-priority controller, CLUSTER-A side (prereg A4.3 'fmdx_controller'); stateless between cycles (state file), requeue-safe.
# Usage: sbatch clusterA_controller.sbatch CAP   (CAP = max CLUSTER-A GPUs for backbone arms, from CAPACITY_NOTE.md)
set -u
X=$U/exp/FMD/xs
python3 $X/code/fmdx_backbone_v1/controller/gate_controller.py --site clusterA --root $U --arms $X/admin/arms.tsv \
   --cap ${1:-4} --period 300 --until 2026-11-10T23:59:00Z
