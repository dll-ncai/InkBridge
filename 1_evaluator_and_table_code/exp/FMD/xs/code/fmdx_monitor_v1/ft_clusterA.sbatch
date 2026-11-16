#!/bin/bash
#SBATCH -J fxm_ft
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH -o $U/exp/FMD/xs/runs/fmdx_monitor/logs/ft_%x_%j.out
# usage: sbatch -J fxm_ft_<ARM> -p gpuidle --gres=gpu:<type>:1 ft_clusterA.sbatch ARM SRC_CKPT
exec 2>&1
bash $U/exp/FMD/xs/code/fmdx_monitor_v1/ft_job.sh "$1" clusterA "$2" ${3:-20261034}
