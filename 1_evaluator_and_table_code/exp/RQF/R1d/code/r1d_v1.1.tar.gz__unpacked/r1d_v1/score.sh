#!/bin/bash
# R1d scoring task: lane_score.py (uhwr_eval 0.3.1, vendored = rq3_v1 copy) on one (base, set) glob from score_in/BASE/tasks.txt.
set -uo pipefail; exec 2>&1
source $U/exp/RQF/R1d/code/r1d_v1/r1d_env.sh
B=$1; ONLY=$(sed -n ${SLURM_ARRAY_TASK_ID}p $RD/score_in/$B/tasks.txt); [ -z "$ONLY" ] && exit 0
T=/tmp/R1d_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}; mkdir -p $T $RD/score_out
echo "$(utc) start $B task=$SLURM_ARRAY_TASK_ID only=$ONLY node=$(hostname)" >> $RD/score_out/job_history.txt
apptainer exec -B $SCRATCH_ROOT -B /tmp $IMG env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=$T MPLCONFIGDIR=$T CUDA_VISIBLE_DEVICES= PYTHONPATH=$UE $VEV/bin/python $RC/lane_score.py --preds-dir $RD/score_in/$B \
  --out $RD/score_out --e0code $C0 --procs 16 --only "$ONLY"; rc=$?
echo "$(utc) end $B task=$SLURM_ARRAY_TASK_ID rc=$rc" >> $RD/score_out/job_history.txt; rm -rf $T; exit $rc
