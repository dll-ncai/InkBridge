#!/bin/bash
# RQF R2d stage body (both lanes). Usage: stage.sh STAGE [SEED]
# STAGE: data_tamil | data_lto | smoke | lto | ltow | ftp SEED | ftw SEED | smrp SEED | smrw SEED | samp_lto | samp_ltow | samp_ftp SEED | samp_ftw SEED |
#        samp_smrp SEED | samp_smrw SEED | score    (SM-R = amendment A2.3, sha bface42b...)
# Recipes (prereg R2d): LTO = M5 LUO recipe (m5_arm.sh phases 1-2, code_e33511dd unchanged) on m5_lto_train, seed 20261016;
# LTO_W = FMDX A4.2 WARM/NOFM (fmdx_backbone_v1 bb_train.py, as jobs/arm.sh) init LTO selected ckpt, 40k steps on m5_lto_train;
# LTO-FT = COMP-p01 recipe (frame_code_a44e78b0 e0_train.py as fs_arm_p12.sh) on tamil_p01; LTO_W-FT = A4.3(e) FT (as
# fmdx_monitor_v1/ft_job.sh) on tamil_p01; FT selection = lowest native LDTW on tamil_val1000 (seed 0, DDIM-50 eta 0, EMA).
set -uo pipefail; exec 2>&1
source $U/exp/RQF/R2d/code/r2d_v1/r2d_env.sh
ST=$1; SEED=${2:-}
[ "${SMOKE:-0}" = 1 ] || check_code
CPUS=${SLURM_CPUS_PER_TASK:-8}; J=${SLURM_JOB_ID:-x}_${SLURM_RESTART_COUNT:-0}
SETS7=${SETS7:-"test_hpl_tamil test_ouhdl test_casia_zh test_casia_en test_iam500 german_unipen_de_test german_onhw_all"}
SSEEDS=${SSEEDS:-"0 1 2"}; MAXU=${MAXU:-}; MX64=(); MXBB=(); [ -n "$MAXU" ] && { MX64=(--max-units $MAXU); MXBB=(--max-units $MAXU); }
LTO_NPZ=${LTO_NPZ:-$DD/m5_lto_train.npz}; LTO_R224=${LTO_R224:-$DD/lto_train_render224.npy}
P01_NPZ=${P01_NPZ:-$DD/tamil_p01.npz}; P01_R224=${P01_R224:-$DD/tamil_p01_render224.npy}; TV=$DD/tamil_val1000.npz; TVR=$DD/tamil_val1000_render224.npz
BB_STEPS=${BB_STEPS:-100000}; BB_SAVE=${BB_SAVE:-10000}; W_STEPS=${W_STEPS:-40000}; W_SAVE=${W_SAVE:-10000}; F_STEPS=${F_STEPS:-10000}; F_SAVE=${F_SAVE:-1000}
LTO_VALSETS=${LTO_VALSETS:-"casia_zh casia_en iam ouhdl"}; VALN=${VALN:-250}
gt_of() { case $1 in test_hpl_tamil|test_casia_zh|test_casia_en) echo $R/exp/X4a/data/x4a_${1#test_}.npz;; test_ouhdl) echo $OU/e0_test.npz;;
  test_iam500) echo $D/m5_test_iam500.npz;; german_unipen_de_test) echo $R/data/d2_v2/unipen_de/e0_test.npz;; german_onhw_all) echo $R/data/d2_v2/onhw/e0_all.npz;;
  val1000_tamil) echo $TV;; *) echo UNKNOWN_SET_$1;; esac; }
rd_of() { if [ "$SITE" = clusterA ]; then case $1 in test_ouhdl) echo $R/exp/FMD/data/fmd_render224_test.npz;; *) echo $X/data/monitor_v1/test_$1_render224.npz;; esac
  else echo $DD/test_renders/test_$1_render224.npz; fi; }
vr_of() { [ "$1" = ouhdl ] && echo $DD/val250_ouhdl_render224.npz || echo $DB/val250_$1_render224.npz; }
best() { cat $1/eval/best_milestone.txt 2>/dev/null; }
# run_train RUN LOGKIND LOGGLOB MINSTEP CMD... : setsid training + in-job gate watcher; rc 20 on GATE_FAIL
run_train() { local run=$1 kind=$2 lg=$3 ms=$4; shift 4
  setsid "$@" > $run/train_${J}.log 2>&1 & local tp=$!; echo $tp > $run/.train_pgid
  start_watch $kind "$lg" $run $ms $tp; wait $tp; local rc=$?; stop_watch
  hist $run "train_exit rc=$rc"; [ -f $run/GATE_FAIL ] && { hist $run "GATE_FAIL $(tr -d '\n' < $run/GATE_FAIL | cut -c1-300)"; return 20; }
  return $rc; }
sc64() {  # sc64 GT PREDS METHOD ROWS PATH SCORER : e0_score.py e0 (native+standardised rows)
  [ -s "$4" ] && return 0; pyc $5 $6 e0 --npz $1 --preds $2 --method $3 --input rendered --out ${4%.csv}.part.csv --procs $CPUS >> $(dirname $4)/score.log 2>&1; [ -s ${4%.csv}.part.csv ] && mv ${4%.csv}.part.csv $4; }

do_data_tamil() { mkdir -p $DD
  [ -s $DD/tamil_val1000.npz ] || pyc $P_M5 $RC/build_r2d_data.py tamil --out $DD --procs $CPUS || return 2
  [ -s $DD/tamil_p01_render224.npy ] || pyc $P_BB $RC/r2d_render.py tamilp01 --npz $DD/tamil_p01.npz --out $DD/tamil_p01_render224.npy --procs $CPUS || return 2
  [ -s $TVR ] || pyc $P_BB $BB/prep_renders.py gtset --npz $TV --out $TVR --procs $CPUS || return 2; }
do_data_lto() { do_data_lto_build || return 2; do_data_lto_render; }
do_data_lto_build() { mkdir -p $DD; [ -s $DD/m5_lto_train.npz ] || pyc $P_M5 $RC/build_r2d_data.py lto --out $DD || return 2; }
do_data_lto_render() { mkdir -p $DD
  [ -s $DD/lto_train_render224.npy ] || pyc $P_BB $RC/r2d_render.py lto --npz $DD/m5_lto_train.npz --out $DD/lto_train_render224.npy --procs $CPUS || return 2
  [ -s $DD/val250_ouhdl_render224.npz ] || pyc $P_BB $BB/prep_renders.py gtset --npz $D/m5_val_ouhdl.npz --out $DD/val250_ouhdl_render224.npz --procs $CPUS || return 2
  if [ "$SITE" != clusterA ]; then mkdir -p $DD/test_renders
    [ -s $TVR ] || pyc $P_BB $BB/prep_renders.py gtset --npz $TV --out $TVR --procs $CPUS || return 2
    for s in $SETS7; do o=$(rd_of $s); [ -s $o ] || pyc $P_BB $BB/prep_renders.py gtset --npz $(gt_of $s) --out $o --procs $CPUS || return 2; done; fi; }

do_lto() { local RUN=$RUNB/LTO EV=$RUNB/LTO/eval; mkdir -p $EV; hist $RUN "lto_start"; gpu_guard
  [ -f $RUN/GATE_FAIL ] && return 20
  if [ ! -f $RUN/training_complete.json ]; then
    read -r -a GL <<< "$(echo ${CUDA_VISIBLE_DEVICES:-} | tr ',' ' ')"; local NG=${#GL[@]}; [ "${SMOKE:-0}" = 1 ] && NG=1
    local ARGS=(--e0-train-npz $LTO_NPZ --condition rendered --run-dir $RUN --batch-size ${BATCH:-128} --gradient-accumulate-every ${ACCUM:-4}
      --train-steps $BB_STEPS --save-every $BB_SAVE --validation-every ${VAL_EVERY:-1000} --sequence-length 320 --image-size 64 --sampling-timesteps 50
      --train-fraction 0.99 --num-workers ${NW:-6} --wandb-mode disabled --random-seed 20261016 --sample-count ${SAMPLE_COUNT:-25} --classes 0 --notes RQF_R2d_LTO_LUOrecipe ${AMPF:-})
    hist $RUN "train_start ngpu=$NG"
    if [ $NG -ge 2 ]; then PORT=$((29500 + (${SLURM_JOB_ID:-0} % 3000)))
      run_train $RUN tqdm "$RUN/train_${J}.log" ${GATE_MIN:-5000} $W PYTHONPATH=$P_M5 OMP_NUM_THREADS=1 $VENV/bin/python -m accelerate.commands.launch \
        --num_processes 2 --multi_gpu --main_process_port $PORT $MC/m5_train.py "${ARGS[@]}"; rc=$?
    else run_train $RUN tqdm "$RUN/train_${J}.log" ${GATE_MIN:-5000} $W PYTHONPATH=$P_M5 OMP_NUM_THREADS=1 $VENV/bin/python $MC/m5_train.py "${ARGS[@]}"; rc=$?; fi
    [ $rc = 20 ] && return 20; [ -f $RUN/training_complete.json ] || { hist $RUN "TRAINING_INCOMPLETE rc=$rc"; return 7; }
  fi
  [ -s $EV/best_milestone.txt ] && return 0
  local MS; MS=$(for f in $RUN/checkpoints/checkpoint_*.pt; do m=$(basename $f .pt); echo $((10#${m#checkpoint_})); done | sort -n)
  for m in $MS; do for s in $LTO_VALSETS; do o=$EV/val__${s}__ms${m}_preds.npz
    [ -s $o ] || { py $P_M5 $C0/e0_sample.py --run-dir $RUN --milestone $m --npz $D/m5_val_$s.npz --condition rendered --sampling-timesteps 50 --ddim-eta 0 \
      --batch-size 256 --seed 0 --subset $VALN --subset-seed 0 "${MX64[@]}" --out ${o%.npz}.part.npz > /dev/null 2>&1 && mv ${o%.npz}.part.npz $o; } || { hist $RUN "VAL_SAMPLING_FAILED $s ms$m"; return 8; }
    sc64 $D/m5_val_$s.npz $o r2d_LTO_ms$m $EV/val__${s}__ms${m}_rows.csv $P_M5 $C0/e0_score.py || { hist $RUN "VAL_SCORING_FAILED $s ms$m"; return 9; }
  done; done
  B=$(pyc $P_M5 $MC/m5_select.py --eval-dir $EV --valsets ${LTO_VALSETS// /,} --out $EV/selection.json --criterion-note "RQF R2d LTO: LUO rule on val${VALN} x {$LTO_VALSETS} (Tamil excluded), e0_sample --subset $VALN --subset-seed 0 --seed 0, DDIM-50 eta 0, EMA" | tail -1)
  [[ "$B" =~ ^[0-9]+$ ]] || { hist $RUN SELECTION_FAILED; return 10; }
  echo $B > $EV/best_milestone.txt; hist $RUN "selected milestone $B"
  (cd $RUN && sha256sum checkpoints/checkpoint_$(printf %06d $B).pt eval/selection.json run_config*.json training_complete.json > SHA256SUMS_outputs.txt); }

do_ltow() { local RUN=$RUNB/LTO_W EV=$RUNB/LTO_W/eval; mkdir -p $EV; hist $RUN "ltow_start"; gpu_guard
  [ -f $RUN/GATE_FAIL ] && return 20
  local B0; B0=$(best $RUNB/LTO); [ -n "$B0" ] || { hist $RUN "NO_LTO_SELECTION"; return 5; }
  local INIT=$RUNB/LTO/checkpoints/checkpoint_$(printf %06d $B0).pt
  if [ ! -f $RUN/training_complete.json ]; then
    read -r -a GL <<< "$(echo ${CUDA_VISIBLE_DEVICES:-0} | tr ',' ' ')"; local NG=${#GL[@]} PERGPU GB ACC
    if [ "${SMOKE:-0}" = 1 ]; then NG=1; GB=4; ACC=1; else
      local MG; MG=$(memg); if [ $MG -ge 40 ]; then PERGPU=$((NG == 1 ? 128 : 64)); else PERGPU=32; fi
      if [ -f $RUN/checkpoints/latest.pt ]; then PL=$(grep "batch_layout ng=$NG " $RUN/job_history.txt | tail -1 | grep -o "per_gpu=[0-9]*" | cut -d= -f2); [ -n "$PL" ] && PERGPU=$PL; fi
      GB=$((PERGPU * NG)); ACC=$((512 / GB)); [ $((GB * ACC)) -eq 512 ] || { hist $RUN "BAD_BATCH_LAYOUT"; return 11; }; fi
    hist $RUN "batch_layout ng=$NG per_gpu=${PERGPU:-smoke} global_micro=$GB accum=$ACC init=$INIT"
    local A=(--arm-id LTO_W --run-dir $RUN --regime warm --encoder vits14 --init-ckpt $INIT --train-npz $LTO_NPZ --render-npy $LTO_R224 --train-steps $W_STEPS
      --save-every $W_SAVE --batch-size $GB --grad-accum $ACC --learning-rate 8e-5 --enc-lr 2e-4 --enc-warmup 200 --seed 20261016 --num-workers ${NW:-6} --lane $SITE ${AMPF:-} ${BBMAX:-})
    local PORT=$((29500 + (${SLURM_JOB_ID:-0} % 3000)))
    if [ $NG -ge 2 ]; then run_train $RUN jsonl "$RUN/train_log.jsonl" ${GATE_MIN:-2000} $W PYTHONPATH=$P_BB OMP_NUM_THREADS=4 $VENV/bin/python -m torch.distributed.run --standalone --nproc_per_node $NG --master_port $PORT $BB/bb_train.py "${A[@]}"; rc=$?
    else run_train $RUN jsonl "$RUN/train_log.jsonl" ${GATE_MIN:-2000} $W PYTHONPATH=$P_BB OMP_NUM_THREADS=4 $VENV/bin/python $BB/bb_train.py "${A[@]}"; rc=$?; fi
    [ $rc = 20 ] && return 20; [ -f $RUN/training_complete.json ] || { hist $RUN "TRAINING_INCOMPLETE rc=$rc"; tail -5 $RUN/train_${J}.log; return 7; }
  fi
  [ -s $EV/best_milestone.txt ] && return 0
  local MS; MS=$(ls $RUN/bb_ms*.pt | sed 's/.*bb_ms\([0-9]*\)\.pt/\1/' | sort -n)
  for m in $MS; do for s in $LTO_VALSETS; do o=$EV/val__${s}__ms${m}_preds.npz
    [ -s $o ] || py $P_BB $BB/bb_sample.py --ckpt $RUN/bb_ms${m}.pt --render $(vr_of $s) --gt-npz $D/m5_val_${s}.npz --subset $VALN --subset-seed 0 --seed 0 \
      --sampling-timesteps 50 --ddim-eta 0 --batch-size 250 "${MXBB[@]}" --out $o >> $EV/sample.log 2>&1 || { hist $RUN "VAL_SAMPLING_FAILED $s ms$m"; return 8; }
    sc64 $D/m5_val_$s.npz $o r2d_LTO_W_ms$m $EV/val__${s}__ms${m}_rows.csv $P_BB $C0/e0_score.py || { hist $RUN "VAL_SCORING_FAILED $s ms$m"; return 9; }
  done; done
  B=$(pyc $P_BB $BB/vendor/m5_select.py --eval-dir $EV --valsets ${LTO_VALSETS// /,} --out $EV/selection.json --criterion-note "RQF R2d LTO_W: WARM backbone rule (LUO rule) on val${VALN} x {$LTO_VALSETS}, bb_sample --subset $VALN --subset-seed 0 --seed 0, DDIM-50 eta 0, EMA" | tail -1)
  [[ "$B" =~ ^[0-9]+$ ]] || { hist $RUN SELECTION_FAILED; return 10; }
  echo $B > $EV/best_milestone.txt; hist $RUN "selected milestone $B"
  (cd $RUN && sha256sum bb_ms$B.pt eval/selection.json run_config*.json training_complete.json > SHA256SUMS_outputs.txt); }

do_ftp() { local N=${1:-LTO_FT}_s$SEED; local EXTRA=("${@:2}"); local RUN=$RUNB/$N EV=$RUNB/$N/eval; mkdir -p $EV; hist $RUN "ftp_start seed=$SEED"; gpu_guard
  [ -f $RUN/GATE_FAIL ] && return 20
  local B0; B0=$(best $RUNB/LTO); [ -n "$B0" ] || { hist $RUN "NO_LTO_SELECTION"; return 5; }
  local INIT=$RUNB/LTO/checkpoints/checkpoint_$(printf %06d $B0).pt
  if [ "${SMOKE:-0}" != 1 ] && [ $(memg) -lt 23 ]; then hist $RUN "GPU_MEM_LT_24GB"; return 4; fi
  if [ ! -f $RUN/training_complete.json ]; then hist $RUN "train_start init=$INIT"
    run_train $RUN tqdm "$RUN/train_${J}.log" ${GATE_MIN:-1000} $W PYTHONPATH=$P_FC OMP_NUM_THREADS=1 $VENV/bin/python $FC/e0_train.py --e0-train-npz $P01_NPZ --condition rendered \
      --run-dir $RUN --init-from $INIT --frame identity --frame-scripts ouhdl --batch-size ${BATCH:-128} --gradient-accumulate-every ${ACCUM:-4} --learning-rate 8e-5 \
      --train-steps $F_STEPS --save-every $F_SAVE --validation-every $F_SAVE ${AMPF:---amp} --sequence-length 320 --image-size 64 --sampling-timesteps 50 --train-fraction 0.99 \
      --num-workers ${NW:-4} --wandb-mode disabled --random-seed $SEED --sample-count ${SAMPLE_COUNT:-25} --classes 0 --notes RQF_R2d_${N}_p01tamil ${EXTRA[@]+"${EXTRA[@]}"}; rc=$?
    [ $rc = 20 ] && return 20; [ -f $RUN/training_complete.json ] || { hist $RUN "TRAINING_INCOMPLETE rc=$rc"; return 7; }
  fi
  [ -s $EV/best_milestone.txt ] && return 0
  for f in $RUN/checkpoints/checkpoint_*.pt; do m=$(basename $f .pt); m=$((10#${m#checkpoint_}))
    P=$EV/preds/val1000_tamil__${N}_ms${m}_identity__seed0_preds${MAXU:+_smoke$MAXU}.npz
    [ -s $P ] || py $P_FC $RC/r2d_sample64.py --label ${N}_ms$m --run-dir ${RUN#$R/} --milestone $m --stage $R --sets val1000_tamil --frames identity --seeds 0 \
      --out-dir $EV --batch-size 256 "${MX64[@]}" > /dev/null 2>&1 || { hist $RUN "VAL_SAMPLING_FAILED ms$m"; return 8; }
    sc64 $TV $P ${N}_ms$m $EV/val__tamil1000__ms${m}_rows.csv $P_FC $FC/e0_score.py || { hist $RUN "VAL_SCORING_FAILED ms$m"; return 9; }
  done
  B=$(pyc $P_FC $FC/m5_select.py --eval-dir $EV --valsets tamil1000 --out $EV/selection.json --criterion-note "RQF R2d $N: E0 rule on tamil_val1000 (all 1,000 units), seed 0, DDIM-50 eta 0, EMA" | tail -1)
  [[ "$B" =~ ^[0-9]+$ ]] || { hist $RUN SELECTION_FAILED; return 10; }
  echo $B > $EV/best_milestone.txt; hist $RUN "selected milestone $B"
  (cd $RUN && sha256sum checkpoints/checkpoint_$(printf %06d $B).pt eval/selection.json run_config*.json training_complete.json > SHA256SUMS_outputs.txt); }

do_ftw() { local N=${1:-LTO_W_FT}_s$SEED; local EXTRA=("${@:2}"); local RUN=$RUNB/$N EV=$RUNB/$N/eval; mkdir -p $EV; hist $RUN "ftw_start seed=$SEED"; gpu_guard
  [ -f $RUN/GATE_FAIL ] && return 20
  local B0; B0=$(best $RUNB/LTO_W); [ -n "$B0" ] || { hist $RUN "NO_LTO_W_SELECTION"; return 5; }
  local SRC=$RUNB/LTO_W/bb_ms$B0.pt GB ACC
  if [ "${SMOKE:-0}" = 1 ]; then GB=4; ACC=1; elif [ $(memg) -ge 40 ]; then GB=128; ACC=4; else GB=32; ACC=16; fi
  if [ -f $RUN/checkpoints/latest.pt ]; then L0=$(grep -o "global_micro=[0-9]* accum=[0-9]*" $RUN/job_history.txt | tail -1); [ -n "$L0" ] && { GB=$(echo $L0 | sed 's/global_micro=\([0-9]*\).*/\1/'); ACC=${L0##*=}; }; fi
  hist $RUN "batch_layout ng=1 global_micro=$GB accum=$ACC src=$SRC"
  if [ ! -f $RUN/training_complete.json ]; then
    run_train $RUN jsonl "$RUN/train_log.jsonl" ${GATE_MIN:-1000} $W PYTHONPATH=$P_BB OMP_NUM_THREADS=4 $VENV/bin/python $BB/bb_train.py --arm-id ${N} --run-dir $RUN --regime ft \
      --init-ckpt $SRC --train-npz $P01_NPZ --render-npy $P01_R224 --train-steps $F_STEPS --save-every $F_SAVE --batch-size $GB --grad-accum $ACC --learning-rate 8e-5 \
      --enc-lr 8e-5 --enc-warmup 0 --seed $SEED --num-workers ${NW:-4} --lane $SITE ${AMPF:-} ${BBMAX:-} ${EXTRA[@]+"${EXTRA[@]}"}; rc=$?
    [ $rc = 20 ] && return 20; [ -f $RUN/training_complete.json ] || { hist $RUN "TRAINING_INCOMPLETE rc=$rc"; return 7; }
  fi
  [ -s $EV/best_milestone.txt ] && return 0
  local MS; MS=$(ls $RUN/bb_ms*.pt | sed 's/.*bb_ms\([0-9]*\)\.pt/\1/' | sort -n)
  for m in $MS; do o=$EV/val__tamil1000__ms${m}_preds.npz
    [ -s $o ] || py $P_BB $BB/bb_sample.py --ckpt $RUN/bb_ms${m}.pt --render $TVR --gt-npz $TV --seed 0 --sampling-timesteps 50 --ddim-eta 0 --batch-size 250 "${MXBB[@]}" \
      --out $o >> $EV/sample.log 2>&1 || { hist $RUN "VAL_SAMPLING_FAILED ms$m"; return 8; }
    sc64 $TV $o ${N}_ms$m $EV/val__tamil1000__ms${m}_rows.csv $P_BB $C0/e0_score.py || { hist $RUN "VAL_SCORING_FAILED ms$m"; return 9; }
  done
  B=$(pyc $P_BB $BB/vendor/m5_select.py --eval-dir $EV --valsets tamil1000 --out $EV/selection.json --criterion-note "RQF R2d $N: E0 rule on tamil_val1000 (all units), bb_sample seed 0, DDIM-50 eta 0, EMA" | tail -1)
  [[ "$B" =~ ^[0-9]+$ ]] || { hist $RUN SELECTION_FAILED; return 10; }
  echo $B > $EV/best_milestone.txt; hist $RUN "selected FT milestone $B"
  (cd $RUN && sha256sum bb_ms$B.pt eval/selection.json run_config*.json training_complete.json > SHA256SUMS_outputs.txt); }

# ---- test sampling (sealed until the run's own selection exists): 7 sets x sampling seeds 0-2
samp64() {  # samp64 NAME : LTO or LTO_FT_s<seed> (Trainer1D); LTO also test_hpl_tamil in mirror_x (zero-shot, descriptive)
  local N=$1 RUN=$RUNB/$1 OUT=$PREDB/$1; local B; B=$(best $RUN); [ -n "$B" ] || { echo "no selection for $N"; return 5; }; gpu_guard; mkdir -p $OUT
  hist $OUT "samp_start $N ms$B"
  py $P_FC $RC/r2d_sample64.py --label ${N}_ms$B --run-dir ${RUN#$R/} --milestone $B --stage $R --sets $SETS7 --frames identity --seeds $SSEEDS --out-dir $OUT --batch-size 256 "${MX64[@]}" > $OUT/sample_${J}.log 2>&1 || return 8
  if [ "$N" = LTO ]; then py $P_FC $RC/r2d_sample64.py --label ${N}_ms$B --run-dir ${RUN#$R/} --milestone $B --stage $R --sets test_hpl_tamil --frames mirror_x --seeds $SSEEDS --out-dir $OUT --batch-size 256 "${MX64[@]}" >> $OUT/sample_${J}.log 2>&1 || return 8; fi
  echo "$(utc) $N ms$B $(hostname) [$(gpus)] slurm=${SLURM_JOB_ID:-none}" > $OUT/SAMPLING_DONE; hist $OUT "samp_end"; }
sampbb() {  # sampbb NAME CKPT_KIND : LTO_W (bb_ms of the backbone) or LTO_W_FT_s<seed>
  local N=$1 RUN=$RUNB/$1 OUT=$PREDB/$1; local B; B=$(best $RUN); [ -n "$B" ] || { echo "no selection for $N"; return 5; }; gpu_guard; mkdir -p $OUT/preds
  local G; G=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1); hist $OUT "samp_start $N ms$B"
  for s in $SETS7; do for k in $SSEEDS; do o=$OUT/preds/${s}__${N}_ms${B}__seed${k}_preds.npz; [ -f ${o%.npz}.json ] && continue
    py $P_BB $BB/bb_sample.py --ckpt $RUN/bb_ms$B.pt --render $(rd_of $s) --gt-npz $(gt_of $s) --seed $k --sampling-timesteps 50 --ddim-eta 0 --batch-size 250 "${MXBB[@]}" \
      --out $o >> $OUT/sample_${J}.log 2>&1 || return 8
    printf "%s\t%s\t%s\t%s\t%s\n" "$(basename $o)" "$G" "$(hostname)" "${SLURM_JOB_ID:-none}" "$(utc)" >> $OUT/device_log.tsv; done; done
  echo "$(utc) $N ms$B $(hostname) [$(gpus)] slurm=${SLURM_JOB_ID:-none}" > $OUT/SAMPLING_DONE; hist $OUT "samp_end"; }

do_score() { local SYS="LTO LTO_W"; for k in 20261034 20261035 20261036; do SYS="$SYS LTO_FT_s$k LTO_W_FT_s$k SMR_LTO_s$k SMR_LTO_W_s$k"; done; [ -n "${SCORE_SYS:-}" ] && SYS=$SCORE_SYS
  for n in $SYS; do [ -f $PREDB/$n/SAMPLING_DONE ] || { echo "score: $n not sampled yet; nothing to do"; return 0; }; done
  mkdir -p $SCOREB; hist $SCOREB "score_start"
  local GM=""; for s in $SETS7; do GM="$GM,$s=$(gt_of $s)"; done; GM=${GM#,}
  local TMPD=/tmp/r2d_sc_${SLURM_JOB_ID:-$$}; mkdir -p $TMPD/numba $TMPD/mpl
  local EV2="PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=$TMPD/numba MPLCONFIGDIR=$TMPD/mpl CUDA_VISIBLE_DEVICES= PYTHONPATH=$UE"
  local AX; if [ "$SITE" = clusterA ]; then AX="apptainer exec -B $SCRATCH_ROOT -B /tmp $IMG env $EV2 $EVENV/bin/python"; else AX="env $EV2 $EVENV/bin/python"; fi
  for n in $SYS; do local IN=$PREDB/$n/preds
    case $n in LTO_W*|SMR_LTO_W*) IN=$SCOREB/in/$n; mkdir -p $IN; $AX $RC/vendor/adapt_sidecars.py --src $PREDB/$n/preds --dst $IN --gt-map "$GM" > /dev/null || return 2;; esac
    [ -f $SCOREB/rows/$n/.done ] && continue
    $AX $RC/vendor/lane_score.py --preds-dir $IN --out $SCOREB/rows/$n --e0code $C0 --procs $CPUS > $SCOREB/score_$n.log 2>&1 || { hist $SCOREB "SCORE_FAILED $n"; return 9; }
    touch $SCOREB/rows/$n/.done; done
  $AX $RC/r2d_tables.py --rows $SCOREB/rows --preds $PREDB --router $R/exp/D3_ROUTE/router --out $SCOREB/tables ${TABLES_ARGS:-} > $SCOREB/tables.log 2>&1 || { hist $SCOREB "TABLES_FAILED"; tail -5 $SCOREB/tables.log; return 9; }
  rm -rf $TMPD; hist $SCOREB "score_end"; }

do_smoke() {  # CPU smoke of every stage on tiny inputs (same functions, SMOKE overrides); outputs under $RD/smoke
  export SMOKE=1 RUNB=$RD/smoke/runs PREDB=$RD/smoke/preds SCOREB=$RD/smoke/score_out
  local S=$RD/smoke/data; mkdir -p $S; rm -rf $RUNB $PREDB $SCOREB
  pyc $P_M5 - <<PYEOF || return 2
import numpy as np
def cut(src, dst, n, rsrc=None, rdst=None):
    z = dict(np.load(src)); o = z["seq_offsets"]; d = {k: v[:n] for k, v in z.items() if k not in ("seq", "seq_offsets")}
    d["seq"] = z["seq"][:o[n]]; d["seq_offsets"] = o[:n + 1]; np.savez(dst, **d)
    if rsrc: np.save(rdst, np.load(rsrc, mmap_mode="r")[:n])
cut("$DD/m5_lto_train.npz", "$S/lto64.npz", 64, "$DD/lto_train_render224.npy", "$S/lto64_r224.npy")
cut("$DD/tamil_p01.npz", "$S/p01_32.npz", 32, "$DD/tamil_p01_render224.npy", "$S/p01_32_r224.npy")
PYEOF
  export LTO_NPZ=$S/lto64.npz LTO_R224=$S/lto64_r224.npy P01_NPZ=$S/p01_32.npz P01_R224=$S/p01_32_r224.npy
  export BB_STEPS=2 BB_SAVE=2 W_STEPS=2 W_SAVE=2 F_STEPS=2 F_SAVE=2 VAL_EVERY=2 BATCH=4 ACCUM=1 NW=0 AMPF=--no-amp SAMPLE_COUNT=4 MAXU=4 SSEEDS=0 GATE_MIN=999999
  export LTO_VALSETS=ouhdl VALN=4 SEED=20261034 TABLES_ARGS=--smoke
  MX64=(--max-units $MAXU); MXBB=(--max-units $MAXU)   # v1.1: derived caps were computed before the smoke exports
  local rc=0
  for fn_ in do_lto do_ltow do_ftp do_ftw; do $fn_; r_=$?; echo "smoke $fn_ rc=$r_"; [ $r_ = 0 ] || rc=1; done
  do_ftp SMR_LTO --replay-npz $LTO_NPZ --urdu-share 0.5; r_=$?; echo "smoke smrp rc=$r_"; [ $r_ = 0 ] || rc=1
  do_ftw SMR_LTO_W --pseudo-npz $LTO_NPZ --pseudo-render-npy $LTO_R224 --pseudo-frac 0.5; r_=$?; echo "smoke smrw rc=$r_"; [ $r_ = 0 ] || rc=1
  samp64 SMR_LTO_s20261034; echo "smoke samp smrp rc=$?"; sampbb SMR_LTO_W_s20261034; echo "smoke samp smrw rc=$?"
  samp64 LTO; echo "smoke samp64 LTO rc=$?"; samp64 LTO_FT_s20261034; echo "smoke samp64 FT rc=$?"
  sampbb LTO_W; echo "smoke sampbb LTO_W rc=$?"; sampbb LTO_W_FT_s20261034; echo "smoke sampbb FT rc=$?"
  for n in LTO LTO_FT_s20261034 LTO_W LTO_W_FT_s20261034 SMR_LTO_s20261034 SMR_LTO_W_s20261034; do [ -f $PREDB/$n/SAMPLING_DONE ] || rc=1; done
  SCORE_SYS="LTO LTO_W LTO_FT_s20261034 LTO_W_FT_s20261034 SMR_LTO_s20261034 SMR_LTO_W_s20261034" do_score; r_=$?; echo "smoke score rc=$r_"; [ $r_ = 0 ] && [ -s $SCOREB/tables/R2d_tables_long.csv ] || rc=1
  # gate watcher self-test: synthetic tqdm log with a nan loss must produce GATE_FAIL (no kill in --once mode)
  local GT=$RD/smoke/gate_test; rm -rf $GT; mkdir -p $GT; printf 'loss: 0.0100:  1%%|  | 10/100 [00:01<00:10]\rloss: nan:  2%%|  | 20/100 [00:02<00:10]\n' > $GT/t.log
  $PY3 $RC/gate_watch.py --kind tqdm --log $GT/t.log --run-dir $GT --min-step 5 --pgid 1 --once > /dev/null; [ -f $GT/GATE_FAIL ] && echo "smoke gate_watch nan -> GATE_FAIL ok" || { echo "smoke gate_watch FAILED"; rc=1; }
  [ $rc = 0 ] && { echo "$(utc) SMOKE_PASS slurm=${SLURM_JOB_ID:-none} host=$(hostname)" > $RD/admin/SMOKE_PASS; echo SMOKE_PASS; }
  return $rc; }

mkdir -p $RD/admin
case $ST in
  data_tamil) do_data_tamil;; data_lto) do_data_lto;; data_lto_build) do_data_lto_build;; data_lto_render) do_data_lto_render;; smoke) do_smoke;;
  lto) do_lto;; ltow) do_ltow;; ftp) do_ftp LTO_FT;; ftw) do_ftw LTO_W_FT;;
  smrp) do_ftp SMR_LTO --replay-npz $LTO_NPZ --urdu-share 0.5;;   # A2.3 SM-R on LTO: 50 % tamil_p01 / 50 % m5_lto_train per batch
  smrw) do_ftw SMR_LTO_W --pseudo-npz $LTO_NPZ --pseudo-render-npy $LTO_R224 --pseudo-frac 0.5;;   # A2.3 SM-R on LTO_W (fixed 50 % second source)
  samp_smrp) samp64 SMR_LTO_s$SEED;; samp_smrw) sampbb SMR_LTO_W_s$SEED;;
  samp_lto) samp64 LTO;; samp_ftp) samp64 LTO_FT_s$SEED;; samp_ltow) sampbb LTO_W;; samp_ftw) sampbb LTO_W_FT_s$SEED;;
  score) do_score;; *) echo "unknown stage $ST"; exit 2;;
esac
rc=$?; echo "$(utc) stage $ST $SEED exit $rc"; exit $rc
