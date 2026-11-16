#!/bin/bash
# RQF R2d CLUSTER-B submit helper (lane copy B; sourced by clusterB_stage.sbatch, or run: bash clusterB_sub.sh STAGE [SEED] [extra sbatch args]).
# CLUSTER-B GPU partitions have 1-day limits, so each stage runs in 1-day segments (continuation afterany:self) and, when its
# done-marker exists, submits its downstream stages (trigger chain = the same DAG as the CLUSTER-A afterok chain).
N=$U_B; RDN=$N/exp/RQF/R2d; SELFS=$RDN/code/r2d_v1/clusterB_stage.sbatch
opts_of() { case $1 in
  lto|ltow) echo "-A <account> -N 1 -p H100-RP,H200-SDS,A100-SDS --gres=gpu:2 --cpus-per-task=16 --mem=200G --time=1-00:00:00";;
  score) echo "-A <account> -N 1 -p V100-32GB,V100-32GB-SDS --cpus-per-task=16 --mem=60G --time=1-00:00:00";;
  *) echo "-A <account> -N 1 -p H100-RP,A100-SDS,A100-RP,H200-SDS --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=1-00:00:00";; esac; }
name_of() { echo "r2d_${1}${2:+_${2: -2}}"; }
sub() {  # sub STAGE SEED [extra sbatch args...] -> prints job id
  local st=$1 sd=$2; shift 2
  sbatch --parsable -J $(name_of $st $sd) $(opts_of $st) "$@" $SELFS $st $sd
}
if [ "${BASH_SOURCE[0]}" = "$0" ]; then st=$1; sd=${2:-}; shift; [ $# -gt 0 ] && shift; sub $st "$sd" "$@"; fi
