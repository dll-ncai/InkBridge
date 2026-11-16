#!/bin/bash
# RQF copy of fmdx_monitor_v1/ft_job.sh (sha256 recorded in PROVENANCE_RQF_R2.json); only RUN, train npz and render npy are parametrised. FMDX warm monitor: Urdu p01 fine-tune (prereg A4.3(e): COMP-p01 recipe exactly) of one backbone milestone + val1000 milestone selection
# (E0 rule: lowest native LDTW on val1000 = e0_val.npz --subset 1000 --subset-seed 0, seed 0, DDIM-50 eta 0, EMA).
# Usage: ft_job.sh ARM LANE SRC_CKPT [SEED]     RUN = $MR/ft/<ARM>_FT_<LANE>
# Race (first copy to start wins): VPS lock dir race\<ARM>_FT (common.sh race_lock). Frozen code fmdx_backbone_v1 is only executed, never edited.
set -u
source $U/exp/FMD/xs/code/fmdx_backbone_v1/jobs/common.sh
ARM=$1; LANE=$2; SRC=$3; SEED=${4:-20261034}
STEPS=${FT_STEPS:-10000}; SAVE=${FT_SAVE:-1000}   # prereg A4.3(e): 10,000 steps, save every 1,000 (env overrides are for the smoke test only)
NMS=$((STEPS / SAVE))
MR=$X/runs/fmdx_monitor; RUN=${RQF_RUN:?}; mkdir -p $RUN
hist $RUN "ft_job_start arm=$ARM lane=$LANE src=$SRC seed=$SEED"
check_code || { hist $RUN CODE_SHA_MISMATCH; exit 3; }
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
awk -v c="$cc" 'BEGIN{exit !(c+0 >= 8.0)}' || { hist $RUN "GPU_CC_LT_8 $cc"; exit 4; }
[ -f $SRC ] || { hist $RUN "SRC_MISSING $SRC"; exit 5; }
# (VPS race lock removed 4 Nov: VPS off, <author> decision; cross-cluster race locks ended)
# local guard (same-lane copies share RUN): only one copy may train in RUN at a time (race_lock treats every clusterA copy as WIN)
if [ ! -f $RUN/training_complete.json ]; then
  ME=${SLURM_JOB_ID:-x}
  if mkdir $RUN/.local_lock 2>/dev/null; then echo $ME > $RUN/.local_owner
  else OWN=$(cat $RUN/.local_owner 2>/dev/null)
    if [ "$OWN" != "$ME" ] && squeue -h -j "$OWN" -t R,CF,CG 2>/dev/null | grep -q .; then
      echo "{\"arm\":\"$ARM\",\"lane\":\"$LANE\",\"slurm\":\"$ME\",\"owner\":\"local copy $OWN\",\"utc\":\"$(utc)\"}" > $RUN/RACE_LOST_$ME.json; hist $RUN "LOCAL_RACE_LOST to $OWN"; exit 0
    fi
    echo $ME > $RUN/.local_owner
  fi
fi
# batch layout: 512 sequences per optimiser step on every lane (A6): 1 GPU >=40 GB 128x4; <40 GB 32 x accum 16
MEMG=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' '); MEMG=$((MEMG / 1024))
if [ $MEMG -ge 40 ]; then GB=128; ACC=4; else GB=32; ACC=16; fi
[ -n "${FT_LAYOUT:-}" ] && { GB=${FT_LAYOUT%x*}; ACC=${FT_LAYOUT#*x}; }   # e.g. FT_LAYOUT=32x16 keeps the micro-batch layout of a resumed run on a bigger GPU (same 512 sequences per step)
hist $RUN "batch_layout ng=1 mem_gb=$MEMG global_micro=$GB accum=$ACC units_per_step=512"
if [ ! -f $RUN/training_complete.json ]; then
  hist $RUN "train_start"
  J=${SLURM_JOB_ID:-x}_${SLURM_RESTART_COUNT:-0}
  $WRAP $VENV/bin/python $BB/bb_train.py --arm-id ${ARM}_FT --run-dir $RUN --regime ft --init-ckpt $SRC --train-npz ${FT_NPZ:-$P01_NPZ} --render-npy ${FT_R224:-$P01_R224} \
     --train-steps $STEPS --save-every $SAVE --batch-size $GB --grad-accum $ACC --learning-rate 8e-5 --enc-lr 8e-5 --enc-warmup 0 --seed $SEED --num-workers 4 --lane $LANE > $RUN/train_${J}.log 2>&1
  rc=$?; hist $RUN "train_exit rc=$rc"
  [ -f $RUN/training_complete.json ] || { tail -20 $RUN/train_${J}.log; exit 7; }
fi
EV=$RUN/eval; mkdir -p $EV
MS=$(seq 1 $NMS)
for m in $MS; do
  o=$EV/val__ouhdl1000__ms${m}_preds.npz
  [ -s $o ] || $PY $BB/bb_sample.py --ckpt $RUN/bb_ms${m}.pt --render $R/exp/FMD/data/fmd_render224_val1000.npz --gt-npz $R/data/ouhdl_e0_v3/e0_val.npz \
      --subset 1000 --subset-seed 0 --seed 0 --sampling-timesteps 50 --ddim-eta 0 --batch-size 250 --out $o >> $EV/sample.log 2>&1 || { hist $RUN "VAL_SAMPLING_FAILED ms$m"; exit 8; }
done
CPUS=${SLURM_CPUS_PER_TASK:-8}
for m in $MS; do r=$EV/val__ouhdl1000__ms${m}_rows.csv
  [ -s $r ] && continue
  $PY $C0/e0_score.py e0 --npz $R/data/ouhdl_e0_v3/e0_val.npz --preds $EV/val__ouhdl1000__ms${m}_preds.npz --method fmdx_${ARM}_FT_ms${m} --input rendered \
     --out ${r%.csv}.part.csv --procs $CPUS >> $EV/score.log 2>&1 && mv ${r%.csv}.part.csv $r || { hist $RUN "VAL_SCORING_FAILED ms$m"; exit 9; }
done
BEST=$($PY $BB/vendor/m5_select.py --eval-dir $EV --valsets ouhdl1000 --out $EV/selection.json \
  --criterion-note "FMDX monitor FT $ARM: E0 rule on val1000 (e0_val.npz --subset 1000 --subset-seed 0), bb_sample.py seed 0, DDIM-50 eta 0, EMA, training lane $LANE" | tail -1)
[[ "$BEST" =~ ^[0-9]+$ ]] || { hist $RUN SELECTION_FAILED; exit 10; }
echo $BEST > $EV/best_milestone.txt; hist $RUN "selected FT milestone $BEST (step $((BEST * SAVE)))"
(cd $RUN && sha256sum bb_ms*.pt eval/selection.json eval/*_preds.npz run_config*.json training_complete.json > SHA256SUMS_outputs.txt)
hist $RUN "ft_job_end"
