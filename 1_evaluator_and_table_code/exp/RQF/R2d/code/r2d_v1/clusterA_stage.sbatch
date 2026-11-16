#!/bin/bash
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=dgx[001-004],gpu[003-004,013-014,017-023,119-148]
#SBATCH -o $U/exp/RQF/R2d/logs/%x_%j.out
# RQF R2d CLUSTER-A job wrapper (lane copy A). Usage: sbatch -J r2d_<stage>[_<seed>] <resources> [--dependency=afterok:..] clusterA_stage.sbatch STAGE [SEED]
# Stage 'lto' and 'data_tamil+lto' bodies are in stage.sh; START/END lines go to exp/RQF/R2d/admin/r2d_events.log (race evidence).
set -uo pipefail; exec 2>&1
export R2D_SITE=clusterA; RD=$U/exp/RQF/R2d; mkdir -p $RD/admin $RD/logs
ev() { echo "$(date -u +%FT%TZ) [R2d clusterA] $* slurm=${SLURM_JOB_ID:-none} restart=${SLURM_RESTART_COUNT:-0} host=$(hostname) part=${SLURM_JOB_PARTITION:-none}" >> $RD/admin/r2d_events.log; }
[ -e $RD/admin/STOP_clusterA ] && { ev "STOP_clusterA present (race lost / stopped): stage $* not run"; exit 1; }
ev "START stage $*"
if [ "$1" = data ]; then bash $RD/code/r2d_v1/stage.sh data_tamil && bash $RD/code/r2d_v1/stage.sh data_lto; rc=$?
else bash $RD/code/r2d_v1/stage.sh "$@"; rc=$?; fi
ev "END stage $* rc=$rc"; exit $rc
