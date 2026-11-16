# M5 / X6 shared job helpers. Source after setting TAG (code tag) and RUNROOT. Sampling/scoring use the E0
# final-eval code e0/code_4cd40f57 unchanged (e0_sample.py, e0_score.py, e0_adapter.py, e0_dataset.py).
B=$U
M=$B/exp/M5
IMG=$B/images/torch27.sif
VENV=$B/venvs/e0
C0=$B/e0/code_4cd40f57
MC=$M/code_${TAG}
REPO=$B/repos/nagamatsu
D=$B/data/m5_multiscript_v1
OU=$B/data/ouhdl_e0_v3
X4=$B/exp/X4a/data
PYP=$REPO/src:$MC:$C0
RUN="apptainer exec --nv -B $SCRATCH_ROOT $IMG env PYTHONPATH=$PYP OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $VENV/bin/python"
RUNCPU="apptainer exec -B $SCRATCH_ROOT $IMG env PYTHONPATH=$PYP OMP_NUM_THREADS=1 $VENV/bin/python"
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
gpu_info() { local g; g=$(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null || true); printf '%s' "${g%%$'\n'*}" | tr ',' ';'; }
gpu_guard() {   # torch27.sif fails on compute capability < 8.0 (V100 7.0): exit non-zero
  local cc; cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true); cc=${cc%%$'\n'*}; cc=${cc// /}
  if ! awk -v c="$cc" 'BEGIN{exit !(c+0 >= 8.0)}'; then
    echo "$(utc) GPU compute capability '$cc' < 8.0 on $(hostname): abort"; exit 4
  fi
}
check_code() {  # E0 eval code must be byte-identical to the ms8 final eval; M5 code must match its sha list
  (cd $C0 && sha256sum -c --quiet $MC/code_4cd40f57_sha256.txt) || { echo "E0 eval code differs"; exit 3; }
  (cd $MC && sha256sum -c --quiet $MC/m5_code_sha256.txt) || { echo "M5 code differs from its sha list"; exit 3; }
}
log_job() {  # log_job RUNDIR WHAT
  mkdir -p "$1"
  echo "$(utc) $2 host=$(hostname) slurm=${SLURM_JOB_ID:-none} restart=${SLURM_RESTART_COUNT:-0} partition=${SLURM_JOB_PARTITION:-none} cuda_visible=${CUDA_VISIBLE_DEVICES:-unset} gpus=[$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ';')]" >> "$1/job_history.txt"
}
# sample_one RUNDIR NPZ COND MILESTONE OUT [extra e0_sample args]; skips finished outputs (requeue-safe)
sample_one() {
  local rd=$1 npz=$2 cond=$3 ms=$4 out=$5; shift 5
  if [ -s "$out" ]; then return 0; fi
  local tmp="${out%.npz}.part.npz"; rm -f "$tmp"; local t0; t0=$(utc)
  $RUN $C0/e0_sample.py --run-dir $rd --milestone $ms --npz $npz --condition $cond --sampling-timesteps 50 \
     --ddim-eta 0 --batch-size 256 --seed 0 "$@" --out "$tmp" 2>&1 | grep -av "sampling loop" | tail -1
  [ -s "$tmp" ] || { echo "no output for $out"; return 5; }
  mv "$tmp" "$out"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(basename $out)" "$t0" "$(utc)" "${SLURM_JOB_ID:-none}" \
     "${SLURM_RESTART_COUNT:-0}" "$(hostname)" "$(gpu_info)" "$(sha256sum $out | cut -c1-64)" >> "$(dirname $out)/shard_log.tsv"
}
# score_one NPZ PREDS METHOD INPUT ROWS PROCS (e0_score.py e0: native + standardised); skips finished
score_one() {
  [ -s "$5" ] && return 0
  $RUNCPU $C0/e0_score.py e0 --npz $1 --preds $2 --method $3 --input $4 --out "${5%.csv}.part.csv" --procs $6 2>&1 | tail -1
  [ -s "${5%.csv}.part.csv" ] || { echo "no rows for $5"; return 6; }
  mv "${5%.csv}.part.csv" "$5"
}
# train_arm RUNDIR NGPU GPU_LIST TRAIN_ARGS... ; NGPU 2 -> accelerate 2 procs (NCCL_P2P_DISABLE=1, as E0), else 1 proc
train_arm() {
  local rd=$1 ng=$2 gl=$3; shift 3
  local port=$((29500 + (${SLURM_JOB_ID:-0} + ${RANDOM}) % 3000))
  mkdir -p $rd; local J=${SLURM_JOB_ID:-x}_${SLURM_RESTART_COUNT:-0}
  if [ "$ng" = 2 ]; then
    echo "$(utc) train start 2gpu gpus=$gl" >> $rd/job_history.txt
    if apptainer exec --nv -B $SCRATCH_ROOT $IMG env CUDA_VISIBLE_DEVICES=$gl PYTHONPATH=$PYP OMP_NUM_THREADS=1 \
         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 \
         $VENV/bin/python -m accelerate.commands.launch --num_processes 2 --multi_gpu --main_process_port $port \
         $MC/m5_train.py "$@" > $rd/train_${J}_2gpu.log 2>&1; then
      echo "$(utc) train 2gpu finished" >> $rd/job_history.txt
    else
      echo "$(utc) train 2gpu FAILED; falling back to 1 GPU" >> $rd/job_history.txt
      tr '\r' '\n' < $rd/train_${J}_2gpu.log | grep -a -E "Error|error" | tail -3 >> $rd/job_history.txt || true
      apptainer exec --nv -B $SCRATCH_ROOT $IMG env CUDA_VISIBLE_DEVICES=${gl%%,*} PYTHONPATH=$PYP OMP_NUM_THREADS=1 \
         PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $VENV/bin/python $MC/m5_train.py "$@" > $rd/train_${J}_1gpu.log 2>&1 || return 7
      echo "$(utc) train 1gpu (fallback) finished" >> $rd/job_history.txt
    fi
  else
    echo "$(utc) train start 1gpu gpu=$gl" >> $rd/job_history.txt
    apptainer exec --nv -B $SCRATCH_ROOT $IMG env CUDA_VISIBLE_DEVICES=$gl PYTHONPATH=$PYP OMP_NUM_THREADS=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $VENV/bin/python $MC/m5_train.py "$@" > $rd/train_${J}_1gpu.log 2>&1 || return 7
    echo "$(utc) train 1gpu finished" >> $rd/job_history.txt
  fi
  for f in $rd/train_${J}_*.log; do tr '\r' '\n' < $f | grep -av "sampling loop" | awk 'NR%200==1' > ${f%.log}_progress.txt || true; done
  [ -f $rd/training_complete.json ]
}
# ckpt_ms RUNDIR -> list of milestone integers present
ckpt_ms() { for f in $1/checkpoints/checkpoint_*.pt; do [ -e "$f" ] || continue; m=$(basename $f .pt); echo $((10#${m#checkpoint_})); done | sort -n; }
