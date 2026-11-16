#!/bin/bash
# RQF LUOC40 backbone sampling (selected milestone), ctrl_sample.py as for the comparators; test_ouhdl + 6 non-Urdu sets, seeds 0-2.   Usage: luo_bb_sample.sh TAG(LUOC40) RUNDIR_NAME
set -uo pipefail; exec 2>&1
source $U/exp/RQF/code/rqf_v1/rqf_env.sh
TAG=$1; RN=$2; REPO=$R/repos/nagamatsu; IMGX=$R/images/torch27.sif; VE=$R/venvs/e0
RD=$QR/$RN; BEST=$(cat $RD/eval/best_milestone.txt 2>/dev/null); [[ "$BEST" =~ ^[0-9]+$ ]] || { echo "no selected backbone milestone"; exit 7; }
GPUN=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
PYG="apptainer exec --nv -B $SCRATCH_ROOT $IMGX env CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=$FCODE/lib:$REPO/src:$FCODE OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled $VE/bin/python"
echo "$(utc) bb_sampling_start ms$BEST gpu=[$GPUN] host=$(hostname) slurm=${SLURM_JOB_ID:-x}" >> $RD/job_history.txt
$PYG $CCODE/ctrl_sample.py --label ${TAG}_bb_ms${BEST} --run-dir exp/RQF/runs/$RN --milestone $BEST --stage $R --out-dir $QP/${TAG}_bb --sets test_ouhdl $SIX --frames identity --seeds 0 1 2 --batch-size 256 2>&1 | grep -av "sampling loop" | tail -3
rc=${PIPESTATUS[0]}; echo "$(utc) bb_sampling_end rc=$rc gpu=[$GPUN]" >> $RD/job_history.txt; exit $rc
