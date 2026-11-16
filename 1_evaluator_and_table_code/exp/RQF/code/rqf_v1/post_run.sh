#!/bin/bash
# RQF CPU post-processing for one run: (warm) adapt bb_sample sidecars -> lane_score (uhwr_eval 0.3.1, routeA lane_score.py, same scorer as the monitor), optional composition (A4.4).
# Usage: post_run.sh TAG KIND PREDS_SUBDIR [BBREF_ARM BBREF_MS]   KIND = ft | cmp | luoft (Route A layout already) ; for cmp BBREF_* name the existing backbone preds in the monitor tree.
set -uo pipefail; exec 2>&1
source $U/exp/RQF/code/rqf_v1/rqf_env.sh
TAG=$1; KIND=$2; SUB=$3; BBREF=${4:-}; BBMS=${5:-}
IMGP=$R/images/torch27.sif; PYCPU="apptainer exec -B $SCRATCH_ROOT $IMGP $R/venvs/cpu/bin/python"
GTMAP="test_ouhdl=$R/data/ouhdl_e0_v3/e0_test.npz,val1000_ouhdl=$R/data/ouhdl_e0_v3/e0_val.npz,test_casia_zh=$R/exp/X4a/data/x4a_casia_zh.npz,test_casia_en=$R/exp/X4a/data/x4a_casia_en.npz,test_hpl_tamil=$R/exp/X4a/data/x4a_hpl_tamil.npz,test_iam500=$R/data/m5_multiscript_v1/m5_test_iam500.npz,german_unipen_de_test=$R/data/d2_v2/unipen_de/e0_test.npz,german_onhw_all=$R/data/d2_v2/onhw/e0_all.npz"
SETS="test_ouhdl $SIX"
if [ "$KIND" = cmp ]; then
  mkdir -p $QP/$TAG/bb; for f in $MMR/preds/$BBREF/bb/*__${BBREF}_bb_ms${BBMS}__seed*_preds.*; do b=$(basename $f); ln -sf $f $QP/$TAG/bb/${b/__${BBREF}_bb_ms/__${TAG}_bb_ms}; done
  mkdir -p $QP/$TAG/cmp; $PYCPU $MON/compose_preds.py --arm $TAG --bb-dir $QP/$TAG/bb --ft-dir $QP/$TAG/ft --out $QP/$TAG/cmp | tail -2
fi
SRC=$QP/$TAG/$SUB
if [ "$KIND" = luoft ]; then PRED=$SRC/preds; else PRED=$QS/in/${TAG}_$KIND; rm -rf $PRED; $PYCPU $MON/adapt_sidecars.py --src $SRC --dst $PRED --gt-map "$GTMAP" | tail -1; fi
TASKS=$QS/in/${TAG}_${KIND}.tasks; mkdir -p $QS/in; : > $TASKS
for s in $SETS val1000_ouhdl; do ls $PRED/${s}__*_preds.npz >/dev/null 2>&1 && echo "${s}__*_preds.npz" >> $TASKS; done
U=$R; C=$U/exp/D3_ROUTE/routeA/code; IMG=$U/images/torch27.sif; PYS=$U/venvs/uhwr_eval02/bin/python; UE=$U/shared/uhwr_eval-0.3.1; E0C=$U/e0/code_4cd40f57
export TMPDIR=/tmp/RQS_${SLURM_JOB_ID:-x}; mkdir -p $TMPDIR/numba $TMPDIR/mpl; trap 'rm -rf $TMPDIR' EXIT
ENVV="PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=$TMPDIR/numba MPLCONFIGDIR=$TMPDIR/mpl CUDA_VISIBLE_DEVICES= PYTHONPATH=$UE"
mkdir -p $QS; rc=0
while read -r ONLY; do
  echo "$(utc) score $TAG $KIND $ONLY lane_score_sha256=$(sha256sum $C/lane_score.py | cut -c1-16)" >> $QS/job_history.txt
  timeout 6h apptainer exec -B $SCRATCH_ROOT -B /tmp $IMG env $ENVV $PYS $C/lane_score.py --preds-dir $PRED --out $QS --e0code $E0C --procs ${SLURM_CPUS_PER_TASK:-8} --only "$ONLY" || rc=1
done < $TASKS
echo "$(utc) post_done $TAG $KIND rc=$rc" >> $QS/job_history.txt; exit $rc
