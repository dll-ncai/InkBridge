#!/bin/bash
# R1d unit test (python, CPU) + CPU smoke of both vendored samplers (2 units, seed 0, mirror_x) into $RD/smoke.
set -uo pipefail; exec 2>&1
source $U/exp/RQF/R1d/code/r1d_v1/r1d_env.sh
(cd $RC && sha256sum -c --quiet SHA256SUMS.txt) || { echo "code differs from SHA256SUMS.txt"; exit 3; }
apptainer exec -B $SCRATCH_ROOT -B /tmp $IMG env $ENVB CUDA_VISIBLE_DEVICES= PYTHONPATH=$UE:$FCODE OMP_NUM_THREADS=4 $VEV/bin/python $RC/r1d_selftest.py $RD/plan.json $RD/selftest || exit 1
apptainer exec -B $SCRATCH_ROOT $IMG env $ENVB CUDA_VISIBLE_DEVICES= PYTHONPATH=$FCODE/lib:$REPO/src:$FCODE OMP_NUM_THREADS=8 $VE/bin/python $RC/rqf1_sample.py --label LUO_ms6 \
  --run-dir exp/M5/runs/luo --milestone 6 --stage $R --out-dir $RD/smoke/e0 --sets test_casia_en --frames mirror_x --seeds 0 --max-units 2 2>&1 | grep -av "sampling loop" | tail -2
r1=${PIPESTATUS[0]}
apptainer exec -B $SCRATCH_ROOT -B /tmp $IMG env $ENVB CUDA_VISIBLE_DEVICES= BB_DIR=$BB PYTHONPATH=$REPO/src:$BB:$BB/vendor:$HC:$C0:$UE:$TIMMDEPS:$FCODE OMP_NUM_THREADS=8 $VE/bin/python \
  $RC/bb_frame_sample.py --ckpt $X/runs/NOFM_W_clusterA/bb_ms3.pt --label NOFM_W_bb_ms3 --set test_casia_en --render $X/data/monitor_v1/test_test_casia_en_render224.npz \
  --render-key img224 --gt-npz $R/exp/X4a/data/x4a_casia_en.npz --frames mirror_x --seeds 0 --batch-size 250 --max-units 2 --out-dir $RD/smoke/bb 2>&1 | tail -2
r2=${PIPESTATUS[0]}; echo "smoke e0 rc=$r1 bb rc=$r2"; [ $r1 -eq 0 ] && [ $r2 -eq 0 ]
