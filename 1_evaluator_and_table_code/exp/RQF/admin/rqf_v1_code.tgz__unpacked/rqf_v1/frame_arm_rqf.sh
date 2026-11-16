#!/bin/bash
# RQF copy of frame_code_a44e78b0/jobs/frame_arm.sh phases 1-2 only (INIT overridable by INIT_OVERRIDE; final-set sampling is done by luo_ft.sh with ctrl_sample.py, like the comparators). One C1/C3 fine-tune arm, end to end (train with 4k gate -> val1000 per-milestone sampling + scoring -> selection ->
# test/gold/UNIPEN-de sampling + scoring).  Platform-neutral: the caller (CLUSTER-A sbatch / CLUSTER-B sbatch) sets
#   PYG   python launcher for GPU work   (e.g. "apptainer exec --nv -B $SCRATCH_ROOT $IMG env PYTHONPATH=... $VENV/bin/python")
#   PYC   python launcher for CPU scoring (same without --nv; may equal PYG)
#   CODE  extracted frame_code_<TAG> dir      STAGE  data/run root ($U | $U_B)
#   RUNROOT  where arm run dirs go            ARM    arm name (run dir = $RUNROOT/$ARM)
#   FRAC  p01 | p10 | p30 | p100              FRAME  identity | mirror_x | vt
#   REPLAY 1 | 0   QU (0.25)  QU_END ("" or 0.5)  CUR_START ("" or 7000)   STEPS (10000)  SAVE (1000)  GATE_STEP (4000; "" = no gate)
#   SEEDS ("0 1 2") sampling seeds for the final sets   TRAIN_SEED (20261016)   PROCS (14) scoring processes
#   SMOKE_UNITS ("" or N) caps every sampled set (CPU/GPU smoke only); BATCH (128) ACCUM (4) WORKERS (4) AMP (--amp) are recipe values, override only for smokes
# Recipe = Route A LUO-FT (M5 m5_ft.sh): init LUO ms6 model+EMA, batch 128 x accum 4 = 512, lr 8e-5, 10k steps, ckpt + val loss
# every 1k, --train-fraction 0.99, selection = lowest mean native LDTW on e0_val --subset 1000 --subset-seed 0 (ties -> higher AIoU).
set -uo pipefail
: "${PYG:?}" "${PYC:?}" "${CODE:?}" "${STAGE:?}" "${RUNROOT:?}" "${ARM:?}" "${FRAC:?}" "${FRAME:?}" "${REPLAY:?}"
QU=${QU:-0.25}; QU_END=${QU_END:-}; CUR_START=${CUR_START:-}; STEPS=${STEPS:-10000}; SAVE=${SAVE:-1000}; GATE_STEP=${GATE_STEP-4000}
SEEDS=${SEEDS:-"0 1 2"}; TRAIN_SEED=${TRAIN_SEED:-20261016}; PROCS=${PROCS:-14}; SMOKE_UNITS=${SMOKE_UNITS:-}
BATCH=${BATCH:-128}; ACCUM=${ACCUM:-4}; LR=${LR:-8e-5}; WORKERS=${WORKERS:-4}; AMP=${AMP:---amp}   # recipe values; smoke overrides: BATCH=4 ACCUM=2 WORKERS=0 AMP=--no-amp
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
RD=$RUNROOT/$ARM; EV=$RD/eval; mkdir -p $EV
log() { echo "$(utc) $* host=$(hostname) slurm=${SLURM_JOB_ID:-none} restart=${SLURM_RESTART_COUNT:-0}" | tee -a $RD/job_history.txt; }
D=$STAGE/data/m5_multiscript_v1; OU=$STAGE/data/ouhdl_e0_v3; DE=$STAGE/data/d2_v2/unipen_de
INIT=${INIT_OVERRIDE:-$STAGE/exp/M5/runs/luo/checkpoints/checkpoint_000006.pt}
if [ "$FRAC" = p100 ]; then URDU=$OU/e0_train.npz; else URDU=$D/m5_ft_ouhdl_$FRAC.npz; fi
for f in $INIT $URDU $D/m5_luo_train.npz $OU/e0_val.npz $OU/e0_test.npz $OU/e0_gold.npz $DE/e0_test.npz; do [ -s $f ] || { log "MISSING_INPUT $f"; exit 2; }; done
(cd $CODE && sha256sum -c --quiet SHA256SUMS.txt) || { log "CODE_SHA_MISMATCH"; exit 3; }
log "arm_start frac=$FRAC frame=$FRAME replay=$REPLAY qu=$QU qu_end=$QU_END cur_start=$CUR_START steps=$STEPS gate=$GATE_STEP batch=${BATCH}x${ACCUM} workers=$WORKERS amp=$AMP smoke_units=$SMOKE_UNITS gpu=[$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ';')]"
SMK=(); [ -n "$SMOKE_UNITS" ] && SMK=(--max-units $SMOKE_UNITS)
GSMK=(); [ -n "$SMOKE_UNITS" ] && GSMK=(--gate-max-units $SMOKE_UNITS)
REP=(); [ "$REPLAY" = 1 ] && REP=(--replay-npz $D/m5_luo_train.npz --urdu-share $QU)
[ "$REPLAY" = 1 ] && [ -n "$QU_END" ] && REP+=(--urdu-share-end $QU_END --curriculum-start-step ${CUR_START:-7000})
GATE=(); [ -n "$GATE_STEP" ] && GATE=(--gate-step $GATE_STEP --gate-gold-npz $OU/e0_gold.npz --gate-unipen-npz $DE/e0_test.npz --gate-label ${ARM}_ms$((GATE_STEP / SAVE)) ${GSMK[@]+"${GSMK[@]}"})
# ---------------- phase 1: training (auto-resumes from the run dir's latest checkpoint; exits 0 early on GATE_FAIL)
if [ ! -f $RD/training_complete.json ] && [ ! -f $RD/GATE_FAIL ]; then
  $PYG $CODE/e0_train.py --e0-train-npz $URDU --condition rendered --run-dir $RD --init-from $INIT \
    --frame $FRAME --frame-scripts ouhdl ${REP[@]+"${REP[@]}"} ${GATE[@]+"${GATE[@]}"} \
    --batch-size $BATCH --gradient-accumulate-every $ACCUM --learning-rate ${LR:-8e-5} --train-steps $STEPS --save-every $SAVE --validation-every $SAVE $AMP \
    --sequence-length 320 --image-size 64 --sampling-timesteps 50 --train-fraction 0.99 --num-workers $WORKERS \
    --wandb-mode disabled --random-seed $TRAIN_SEED --sample-count 25 --classes 0 --notes frame_${ARM} \
    > $RD/train_${SLURM_JOB_ID:-x}_${SLURM_RESTART_COUNT:-0}.log 2>&1 || { log "TRAINING_FAILED rc=$?"; exit 7; }
  tr '\r' '\n' < $RD/train_${SLURM_JOB_ID:-x}_${SLURM_RESTART_COUNT:-0}.log | grep -av "sampling loop" | awk 'NR%200==1 || /GATE|gate|Initialised|Replay/' > $RD/train_progress.txt || true
fi
if [ -f $RD/GATE_FAIL ]; then log "GATE_FAIL: $(tr -d '\n' < $RD/GATE_FAIL | cut -c1-400)"; cp $RD/GATE_FAIL $EV/ 2>/dev/null; exit 0; fi
[ -f $RD/training_complete.json ] || { log "TRAINING_INCOMPLETE"; exit 7; }
# ---------------- phase 2: val1000 sampling + scoring per milestone, selection
for f in $RD/checkpoints/checkpoint_*.pt; do
  m=$(basename $f .pt); m=$((10#${m#checkpoint_}))
  P=$EV/preds/val1000_ouhdl__${ARM}_ms${m}_${FRAME}__seed0_preds.npz; R=$EV/val__ouhdl1000__ms${m}_rows.csv
  [ -s $R ] && continue
  $PYG $CODE/frame_sample.py --label ${ARM}_ms${m} --run-dir ${RD#$STAGE/} --milestone $m --stage $STAGE --sets val1000_ouhdl --frames $FRAME --seeds 0 \
     --out-dir $EV --batch-size 256 ${SMK[@]+"${SMK[@]}"} 2>&1 | grep -av "sampling loop" | tail -1
  [ -n "$SMOKE_UNITS" ] && P=${P%.npz}_smoke${SMOKE_UNITS}.npz
  $PYC $CODE/e0_score.py e0 --npz $OU/e0_val.npz --preds $P --method ${ARM}_ms${m} --input rendered --out ${R%.csv}.part.csv --procs $PROCS 2>&1 | tail -1
  mv ${R%.csv}.part.csv $R
done
BEST=$($PYC $CODE/m5_select.py --eval-dir $EV --valsets ouhdl1000 --out $EV/selection.json --criterion-note "frame arm $ARM: E0 rule on e0_val.npz --subset 1000 --subset-seed 0, frame $FRAME (inverse applied before scoring)" | tail -1)
[[ "$BEST" =~ ^[0-9]+$ ]] || { log "SELECTION_FAILED"; exit 8; }
log "selected milestone $BEST"; echo $BEST > $EV/best_milestone.txt
log "arm_phase12_end selected=$BEST"
