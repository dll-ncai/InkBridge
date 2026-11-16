#!/bin/bash
# R1d sampler body. run_sampler.sh SITE(clusterA|clusterB) KIND ARGS... ; clusterA wraps python in torch27.sif, clusterB runs inside the Pyxis
# container (pytorch 2.7.1-cuda12.8, $U_B bind-mounted at $U). Same vendored samplers on both.
set -uo pipefail
source $U/exp/RQF/R1d/code/r1d_v1/r1d_env.sh
SITE=$1; KIND=$2; shift 2
if [ $SITE = clusterA ]; then W1="apptainer exec --nv -B $SCRATCH_ROOT $IMG env"; W2="apptainer exec --nv -B $SCRATCH_ROOT -B /tmp $IMG env"; else W1=env; W2=env; fi
if [ $KIND = e0 ]; then
  LABEL=$1; RUN=$2; MS=$3; OUT=$4; FR=${5//,/ }; shift 5
  $W1 $ENVB PYTHONPATH=$FCODE/lib:$REPO/src:$FCODE OMP_NUM_THREADS=4 $VE/bin/python $RC/rqf1_sample.py \
    --label $LABEL --run-dir $RUN --milestone $MS --stage $R --out-dir $OUT --sets "$@" --frames $FR --seeds 0 1 2 --batch-size 256 2>&1 | grep -av "sampling loop" | tail -30
  exit ${PIPESTATUS[0]}
fi
LABEL=$1; CK=$2; OUT=$3; FR=${4//,/ }; shift 4; rc=0
for t in "$@"; do IFS=: read S RN GT <<< "$t"
  $W2 $ENVB BB_DIR=$BB PYTHONPATH=$REPO/src:$BB:$BB/vendor:$HC:$C0:$UE:$TIMMDEPS:$FCODE OMP_NUM_THREADS=4 $VE/bin/python \
    $RC/bb_frame_sample.py --ckpt $CK --label $LABEL --set $S --render $RN --render-key img224 --gt-npz $GT --frames $FR --seeds 0 1 2 --batch-size 250 --out-dir $OUT 2>&1 | grep -av "sampling loop" | tail -8
  r=${PIPESTATUS[0]}; [ $r -ne 0 ] && rc=$r
done
exit $rc
