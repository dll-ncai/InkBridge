#!/bin/bash
# RQF copy of fmdx_backbone_v1/jobs/arm.sh (arm table, run root and seed parametrised only; BB_SEED default 20261017). FMDX backbone arm job body (both clusters): race lock -> gate pre-check -> train (torchrun, requeue-safe) ->
# non-Urdu val250 milestone selection (LUO rule: lowest mean native LDTW over casia_zh/casia_en/hpl_tamil/iam val250,
# DDIM-50 eta 0 seed 0, EMA weights; m5_select.py vendored unchanged) -> SHA256SUMS of the outputs.
# Usage: arm.sh ARM_ID LANE     (LANE = clusterA | clusterB; arm settings from $BB/arms.tsv)
set -u
source $U/exp/FMD/xs/code/fmdx_backbone_v1/jobs/common.sh
ARMS_TSV=$U/exp/RQF/admin/arms_rqf.tsv; QR=$R/exp/RQF/runs   # RQF: own arm table (the controller's arms.tsv is not touched)
ARM=$1; LANE=$2
RUN=$QR/${ARM}_${LANE}; mkdir -p $RUN
hist $RUN "arm_job_start arm=$ARM lane=$LANE"
# RQF local race guard (same-filesystem copies): the first copy to start owns the run dir; later copies exit 0 while the owner is RUNNING.
RDL=$RUN; mkdir -p $RDL
if [ ! -f $RDL/training_complete.json ]; then
  ME=${SLURM_JOB_ID:-x}
  if mkdir $RDL/.local_lock 2>/dev/null; then echo $ME > $RDL/.local_owner
  else OWN=$(cat $RDL/.local_owner 2>/dev/null)
    if [ "$OWN" != "$ME" ]; then
      if squeue -h -j "$OWN" -t R,CF,CG 2>/dev/null | grep -q .; then echo "$(date -u +%FT%TZ) RACE_LOST $ME owner=$OWN" >> $RDL/job_history.txt; exit 0; fi
      echo $ME > $RDL/.local_owner
    fi
  fi
fi
check_code || { hist $RUN "CODE_SHA_MISMATCH"; exit 3; }
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
awk -v c="$cc" 'BEGIN{exit !(c+0 >= 8.0)}' || { hist $RUN "GPU_CC_LT_8 $cc"; exit 4; }
# race lock (CLUSTER-A: here, on the host; CLUSTER-B: in clusterB_arm.sbatch on the host before srun, FMDX_LOCK_DONE=1)
[ "${FMDX_LOCK_DONE:-0}" = 1 ] || [ -f $RUN/training_complete.json ] || [ -f $RUN/GATE_FAIL ] || {
  L=$(race_lock $ARM $LANE); hist $RUN "race_lock $L"
  case "$L" in LOSE*) echo "{\"arm\":\"$ARM\",\"lane\":\"$LANE\",\"slurm\":\"${SLURM_JOB_ID:-}\",\"owner\":\"${L#LOSE }\",\"utc\":\"$(utc)\"}" > $RUN/RACE_LOST.json; exit 0;; esac
}
REG=$(arm_field $ARM regime); ENC=$(arm_field $ARM encoder); FM=$(arm_field $ARM fm); STEPS=$(arm_field $ARM steps)
SAVE=$(arm_field $ARM save_every); TGT=$(arm_field $ARM repa_targets); TUID=$(arm_field $ARM repa_unit_ids); LAM=$(arm_field $ARM repa_lambda)
PSN=$(arm_field $ARM pseudo_npz); PSR=$(arm_field $ARM pseudo_render); PSF=$(arm_field $ARM pseudo_frac); PEW=$(arm_field $ARM pe_weights)
[ -n "$REG" ] || { hist $RUN "ARM_NOT_IN_TABLE"; exit 5; }
# ---- batch layout (prereg A6: 512 sequences per optimiser step on every lane). Rule fixed in DECISION_RECORD_backbone.md:
#   2 GPUs (each >= 40 GB): 64 per process x accum 4 (= LUO: 128 global split_batches x 4)
#   1 GPU  >= 40 GB       : 128 x accum 4
#   any GPU < 40 GB       : 32 per process x accum 16/NG   (same 512; activation memory of the 224-px encoder)
read -r -a GL <<< "$(echo ${CUDA_VISIBLE_DEVICES:-0} | tr ',' ' ')"; NG=${#GL[@]}
MEMG=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' '); MEMG=$((MEMG / 1024))
if [ $MEMG -ge 40 ]; then PERGPU=$((NG == 1 ? 128 : 64)); else PERGPU=32; fi
# v1.0.2 (lead 4 Nov ~18Z: resume on another GPU type keeps the GPU layout): with a latest.pt, reuse per_gpu of the last
# batch_layout line of this run if it had the same number of GPUs (e.g. A4000 32/proc x accum 8 resumed on 96 GB cards).
if [ -f $RUN/checkpoints/latest.pt ] && [ -f $RUN/job_history.txt ]; then
  PL=$(grep "batch_layout ng=$NG " $RUN/job_history.txt | tail -1 | grep -o "per_gpu=[0-9]*" | cut -d= -f2)
  [ -n "$PL" ] && [ "$PL" != "$PERGPU" ] && { hist $RUN "batch_layout_kept_for_resume per_gpu=$PL (device rule gave $PERGPU)"; PERGPU=$PL; }
fi
GB=$((PERGPU * NG)); ACC=$((512 / GB))
[ $((GB * ACC)) -eq 512 ] || { hist $RUN "BAD_BATCH_LAYOUT ng=$NG pergpu=$PERGPU"; exit 11; }
hist $RUN "batch_layout ng=$NG mem_gb=$MEMG per_gpu=$PERGPU global_micro=$GB accum=$ACC units_per_step=512"
A=(--arm-id $ARM --run-dir $RUN --regime $REG --encoder $ENC --train-npz $LUO_NPZ --render-npy $LUO_R224 --train-steps $STEPS --save-every $SAVE
   --batch-size $GB --grad-accum $ACC --learning-rate 8e-5 --enc-lr 2e-4 --enc-warmup 200 --seed ${BB_SEED:-20261017} --num-workers 6 --lane $LANE)
[ "$REG" = warm ] && A+=(--init-ckpt $LUO_CKPT)
[ "$PEW" != "-" ] && A+=(--pe-weights $PEW)
if [ "$FM" != "-" ]; then A+=(--fm $FM --verdict-file $X/verdicts/verdict_${FM}.json --verdict-every 10000); fi
[ "$TGT" != "-" ] && A+=(--repa-targets $TGT --repa-unit-ids $TUID --repa-lambda $LAM)
[ "$PSN" != "-" ] && A+=(--pseudo-npz $PSN --pseudo-render-npy $PSR --pseudo-frac $PSF)
if [ "$TGT" != "-" ] && [ ! -s "$TGT" ]; then hist $RUN "REPA_TARGETS_MISSING $TGT"; exit 6; fi
if [ ! -f $RUN/training_complete.json ] && [ ! -f $RUN/GATE_FAIL ]; then
  J=${SLURM_JOB_ID:-x}_${SLURM_RESTART_COUNT:-0}; PORT=$((29500 + (${SLURM_JOB_ID:-0} % 3000)))
  hist $RUN "train_start ngpu=$NG"
  $WRAP $VENV/bin/python -m torch.distributed.run --standalone --nproc_per_node $NG --master_port $PORT $BB/bb_train.py "${A[@]}" > $RUN/train_${J}.log 2>&1
  rc=$?; hist $RUN "train_exit rc=$rc"
  [ -f $RUN/GATE_FAIL ] && { hist $RUN "GATE_FAIL"; exit 0; }
  [ -f $RUN/training_complete.json ] || { tail -20 $RUN/train_${J}.log; exit 7; }
fi
[ -f $RUN/GATE_FAIL ] && exit 0
# ---- milestone selection on the non-Urdu val250 sets (in-job, both GPUs)
EV=$RUN/eval; mkdir -p $EV
MS=$(ls $RUN/bb_ms*.pt 2>/dev/null | sed 's/.*bb_ms\([0-9]*\)\.pt/\1/' | sort -n)
jobs_list=(); for m in $MS; do for s in $VALSETS; do jobs_list+=("$m:$s"); done; done
stream() {  # stream GPU STRIDE OFFSET
  local k=0
  for js in "${jobs_list[@]}"; do
    if [ $((k % $2)) -eq $3 ]; then m=${js%%:*}; s=${js#*:}; o=$EV/val__${s}__ms${m}_preds.npz
      if [ ! -s $o ]; then
        CUDA_VISIBLE_DEVICES=$1 $PY $BB/bb_sample.py --ckpt $RUN/bb_ms${m}.pt --render $DB/val250_${s}_render224.npz --gt-npz $D/m5_val_${s}.npz \
          --subset 250 --subset-seed 0 --seed 0 --sampling-timesteps 50 --ddim-eta 0 --batch-size 250 --out $o >> $EV/sample.log 2>&1 || return 5
      fi
    fi; k=$((k + 1))
  done
}
pids=(); for i in $(seq 0 $((NG - 1))); do stream ${GL[$i]} $NG $i & pids+=($!); done
for p in "${pids[@]}"; do wait $p || { hist $RUN "VAL_SAMPLING_FAILED"; exit 8; }; done
CPUS=${SLURM_CPUS_PER_TASK:-8}
for js in "${jobs_list[@]}"; do m=${js%%:*}; s=${js#*:}; r=$EV/val__${s}__ms${m}_rows.csv
  [ -s $r ] && continue
  $PY $C0/e0_score.py e0 --npz $D/m5_val_${s}.npz --preds $EV/val__${s}__ms${m}_preds.npz --method fmdx_${ARM}_ms${m} --input rendered \
     --out ${r%.csv}.part.csv --procs $CPUS >> $EV/score.log 2>&1 && mv ${r%.csv}.part.csv $r || { hist $RUN "VAL_SCORING_FAILED $r"; exit 9; }
done
BEST=$($PY $BB/vendor/m5_select.py --eval-dir $EV --valsets ${VALSETS// /,} --out $EV/selection.json \
       --criterion-note "FMDX $ARM: LUO rule, val250 x {${VALSETS}}, bb_sample.py --subset 250 --subset-seed 0 --seed 0, DDIM-50 eta 0, EMA" | tail -1)
[[ "$BEST" =~ ^[0-9]+$ ]] || { hist $RUN "SELECTION_FAILED"; exit 10; }
echo $BEST > $EV/best_milestone.txt
hist $RUN "selected milestone $BEST (step $((BEST * $SAVE)))"
(cd $RUN && sha256sum bb_ms*.pt eval/selection.json eval/*_preds.npz run_config*.json training_complete.json > SHA256SUMS_outputs.txt)
hist $RUN "arm_job_end"
