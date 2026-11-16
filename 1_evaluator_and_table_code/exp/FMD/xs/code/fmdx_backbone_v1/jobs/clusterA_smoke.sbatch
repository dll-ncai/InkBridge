#!/bin/bash
#SBATCH -J fmdx_bb_smoke
#SBATCH -p gpuidle
#SBATCH --gres=gpu:2
#SBATCH --exclude=dgx[001-005],gpu[004-005,013-014,017-023,119-148]
#SBATCH --cpus-per-task=8
#SBATCH --mem=46G
#SBATCH --time=02:00:00
#SBATCH -o $U/exp/FMD/xs/logs/build/smoke_%j.out
set -u
export FMDX_SITE=clusterA
bash $U/exp/FMD/xs/code/fmdx_backbone_v1/jobs/smoke.sh $U/exp/FMD/xs/runs/smoke/clusterA_$SLURM_JOB_ID clusterA ${1:-}
