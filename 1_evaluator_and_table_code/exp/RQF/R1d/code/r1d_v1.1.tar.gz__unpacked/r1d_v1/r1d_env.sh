#!/bin/bash
# RQF A2.5 R1d shared paths (sourced). Absolute paths only; no sibling env sourced.
R=$U; Q=$R/exp/RQF; RD=$Q/R1d; RC=$RD/code/r1d_v1; X=$R/exp/FMD/xs; BB=$X/code/fmdx_backbone_v1
FCODE=$R/exp/CAND/code/frame_code_a44e78b0; REPO=$R/repos/nagamatsu; HC=$R/exp/FMD/code/fmd_harness_v1; C0=$R/e0/code_4cd40f57
UE=$R/shared/uhwr_eval-0.3.1; TIMMDEPS=$X/code/pydeps_timm; IMG=$R/images/torch27.sif; VE=$R/venvs/e0; VEV=$R/venvs/uhwr_eval02
ENVB="PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID"
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
r1d_log() { echo "$(utc) [rqf-R1d, <agent>] $*" >> $Q/queue/rqf_log.md; }
# v1.1 cross-cluster race hook: runs once inside an CLUSTER-A sampling copy (job name r1d_<UNIT>; Slurm keeps the v1 samp.sh text of the
# already-queued copies, which source this file). First copy to START takes the VPS lock (owner 'clusterA'); if the CLUSTER-B copy holds it, exit 0.
R1D_UNITS=" LUO6 M5u M56 NOFMu_id NOFMu_mir NOFM6 PEINITu_id PEINITu_mir PEINIT6 "
if [ -n "${SLURM_JOB_ID:-}" ] && [ -z "${R1D_VPS_CHECKED:-}" ] && [[ "$R1D_UNITS" == *" ${SLURM_JOB_NAME#r1d_} "* ]] && [[ "${SLURM_JOB_NAME:-}" == r1d_* ]]; then
  export R1D_VPS_CHECKED=1; source $RC/r1d_vps.sh; _u=${SLURM_JOB_NAME#r1d_}; _vl=$(vps_lock $_u clusterA); mkdir -p $RD/logs
  echo "$(utc) $_u vps_lock=$_vl job=$SLURM_JOB_ID host=$(hostname)" >> $RD/logs/samp_history.txt
  case "$_vl" in LOSE*) r1d_log "RACE R1d $_u CLUSTER-A copy $SLURM_JOB_ID lost to ${_vl#LOSE } (VPS lock); exits 0"; exit 0;;
                 ERR) r1d_log "RACE R1d $_u VPS lock unreachable; CLUSTER-A copy $SLURM_JOB_ID proceeds";; esac
fi
