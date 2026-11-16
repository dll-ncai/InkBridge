#!/bin/bash
# One CAND control arm on CLUSTER-A: frame_arm.sh (frame_code_a44e78b0, unchanged) for train -> val1000 per milestone ->
# selection -> test/gold/UNIPEN-de/val250_iam sampling+scoring; then the Route A extra sets (EXTRA_SETS, seed 0) with
# ctrl_sample.py; copy all final preds to $CTRL_OUT/preds; stamp sidecars (job id, code shas); one UTC line per event in cand_log.md.
# Caller sets ARM FRAC FRAME REPLAY [QU] TRAIN_SEED SEEDS EXTRA_SETS; GATE_STEP is forced empty (controls have no gate).
set -uo pipefail
B=$U; IMG=$B/images/torch27.sif; VENV=$B/venvs/e0; REPO=$B/repos/nagamatsu
export CODE=${CODE:-$B/exp/CAND/code/frame_code_a44e78b0} STAGE=$B PROCS=${SLURM_CPUS_PER_TASK:-16}
CC=${CC:-$B/exp/CAND/controls/code}; CTRL_OUT=${CTRL_OUT:-$B/exp/CAND/controls}; export RUNROOT=${RUNROOT:-$CTRL_OUT/runs}
CANDLOG=${CANDLOG:-$B/exp/CAND/queue/cand_log.md}
export GATE_STEP=""
: "${ARM:?}" "${FRAC:?}" "${FRAME:?}" "${REPLAY:?}" "${TRAIN_SEED:?}" "${SEEDS:?}" "${EXTRA_SETS:?}"
export ARM FRAC FRAME REPLAY TRAIN_SEED SEEDS
JID=${SLURM_JOB_ID:-none}; RST=${SLURM_RESTART_COUNT:-0}
clog() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) [ctrl $ARM] $* | clusterA slurm $JID restart $RST host $(hostname)" >> $CANDLOG; }
mkdir -p $RUNROOT $CTRL_OUT/preds $CTRL_OUT/logs
(cd $CC && sha256sum -c --quiet SHA256SUMS.txt) || { clog "CTRL_CODE_SHA_MISMATCH"; exit 3; }
export PYG="apptainer exec --nv -B $SCRATCH_ROOT $IMG env PYTHONPATH=$CODE/lib:$REPO/src:$CODE OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled $VENV/bin/python"
export PYC="apptainer exec -B $SCRATCH_ROOT $IMG env PYTHONPATH=$CODE/lib:$REPO/src:$CODE OMP_NUM_THREADS=1 $VENV/bin/python"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ';')
clog "start frac=$FRAC frame=$FRAME replay=$REPLAY qu=${QU:-} train_seed=$TRAIN_SEED seeds=[$SEEDS] extra=[$EXTRA_SETS] gpu=[$GPU] cuda_visible=${CUDA_VISIBLE_DEVICES:-} smoke=${SMOKE_UNITS:-}"
bash $CODE/jobs/frame_arm.sh; rc=$?
if [ $rc -ne 0 ]; then clog "FRAME_ARM_FAILED rc=$rc"; exit $rc; fi
RD=$RUNROOT/$ARM; EV=$RD/eval
BEST=$(cat $EV/best_milestone.txt 2>/dev/null); [[ "$BEST" =~ ^[0-9]+$ ]] || { clog "NO_BEST_MILESTONE"; exit 8; }
clog "frame_arm done; selected milestone $BEST (val1000 OUHD-L native LDTW)"
SMK=(); [ -n "${SMOKE_UNITS:-}" ] && SMK=(--max-units $SMOKE_UNITS)
$PYG $CC/ctrl_sample.py --label ${ARM}_ms${BEST} --run-dir ${RD#$STAGE/} --milestone $BEST --stage $STAGE --sets $EXTRA_SETS \
   --frames identity --seeds 0 --out-dir $CTRL_OUT --batch-size 256 ${SMK[@]+"${SMK[@]}"} 2>&1 | grep -av "sampling loop" | tail -8
rc=${PIPESTATUS[0]}; if [ $rc -ne 0 ]; then clog "EXTRA_SAMPLING_FAILED rc=$rc"; exit 9; fi
n=0
for f in $EV/preds/*__${ARM}_ms${BEST}_${FRAME}__seed*_preds*.npz; do
  [ -f "$f" ] || continue; j=${f%.npz}.json
  cp -p $f $CTRL_OUT/preds/; [ -f $j ] && cp -p $j $CTRL_OUT/preds/; n=$((n+1))
done
clog "copied $n frame_arm phase-3 preds to $CTRL_OUT/preds"
ENVJ="{\"ARM\":\"$ARM\",\"FRAC\":\"$FRAC\",\"FRAME\":\"$FRAME\",\"REPLAY\":\"$REPLAY\",\"QU\":\"${QU:-}\",\"TRAIN_SEED\":\"$TRAIN_SEED\",\"SEEDS\":\"$SEEDS\",\"EXTRA_SETS\":\"$EXTRA_SETS\",\"GATE_STEP\":\"\",\"SMOKE_UNITS\":\"${SMOKE_UNITS:-}\",\"STEPS\":\"${STEPS:-10000}\",\"BATCH\":\"${BATCH:-128}\",\"ACCUM\":\"${ACCUM:-4}\",\"LR\":\"${LR:-8e-5}\"}"
$PYC $CC/ctrl_stamp.py --arm $ARM --preds-dir $CTRL_OUT/preds --label-glob "*__${ARM}_ms${BEST}_*__seed*_preds*.json" --run-dir $RD \
   --out $CTRL_OUT/PROVENANCE_ctrl_${ARM}.json --job $JID --restart $RST --ctrl-code $CC --frame-code $CODE --env-json "$ENVJ" | tail -1
rc=${PIPESTATUS[0]}; if [ $rc -ne 0 ]; then clog "STAMP_FAILED rc=$rc"; exit 10; fi
clog "end ok; preds in $CTRL_OUT/preds; PROVENANCE_ctrl_${ARM}.json"
