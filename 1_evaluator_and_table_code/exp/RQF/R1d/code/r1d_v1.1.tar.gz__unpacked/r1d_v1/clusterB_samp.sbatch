#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o $U_B/exp/RQF/R1d/logs/%x_%j.out
# R1d CLUSTER-B race copy: stage (bb) -> VPS lock (owner clusterB; first to START wins) -> run_sampler.sh in the Pyxis container. Loser exits 0.
# On a sampler failure (not preemption) the VPS lock is released so the CLUSTER-A copy can still win.  Usage: clusterB_samp.sbatch UNIT KIND ARGS...
set -u; N=$U_B; IMG=$N/images/pytorch_2.7.1-cuda12.8-cudnn9-runtime.sqsh; RDN=$N/exp/RQF/R1d; C=$RDN/code/r1d_v1
source $C/r1d_vps.sh; U=$1; K=$2; mkdir -p $RDN/race $RDN/logs
ut() { date -u +%FT%TZ; }
if [ $K = bb ]; then flock $RDN/stage.lock bash $C/clusterB_stage.sh || exit 5; fi
L=$(vps_lock $U clusterB); echo "$(ut) $U vps_lock=$L job=$SLURM_JOB_ID host=$(hostname) part=$SLURM_JOB_PARTITION restart=${SLURM_RESTART_COUNT:-0}" | tee -a $RDN/logs/clusterB_history.txt
[ "$L" = WIN ] || exit 0
touch $RDN/race/$U.clusterB_won
srun --ntasks=1 --container-image=$IMG --container-mounts=$SCRATCH_B:$SCRATCH_B,$N:$U --container-workdir=$U/exp/RQF/R1d \
  bash $U/exp/RQF/R1d/code/r1d_v1/run_sampler.sh clusterB "${@:2}"
rc=$?; echo "$(ut) $U end rc=$rc job=$SLURM_JOB_ID gpu=[$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)]" | tee -a $RDN/logs/clusterB_history.txt
if [ $rc -ne 0 ] && [ $rc -ne 143 ] && [ $rc -ne 137 ]; then rm -f $RDN/race/$U.clusterB_won; vps_unlock $U; fi
exit $rc
