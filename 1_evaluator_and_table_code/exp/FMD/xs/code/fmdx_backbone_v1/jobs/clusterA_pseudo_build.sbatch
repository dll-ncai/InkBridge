#!/bin/bash
#SBATCH -J fmdx_pseudo_build
#SBATCH -p cpuidle
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o $U/exp/FMD/xs/logs/pseudo/%x_%j.out
# Filter (IoU >= 0.80) + E0-format pseudo_train.npz + 224 render array (prereg A4.2). Run after every InkSight shard finished.
set -u
source $U/exp/FMD/xs/code/fmdx_backbone_v1/jobs/common.sh
check_code || exit 3
PD=$X/data/pseudo
$PY $BB/pseudo_label.py build --pool-dir $PD --out-glob "$PD/inksight/out_*.jsonl" --out-dir $PD
