#SBATCH -p gpuidle
#SBATCH --gres=gpu:a30:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
# E0 smoke 2 (smoke 1 = job 24415666: batch 512 and 256 on one A30 ran out of memory at step 0).
# 2x A30 with HF accelerate, global batch 512 = 2 GPUs x 128 x grad-accum 2 (fallback 2 x 64 x 4),
# 300 steps (checkpoint every 150), resume to 400, then DDIM-50 on 512 random val units + scoring.
set -u
B=$U
IMG=$B/images/torch27.sif
VENV=$B/venvs/e0
D=$B/data/ouhdl_e0_v3
CODE_TAG=1a5ee72e
C=$B/e0/code_$CODE_TAG
REPO=$B/repos/nagamatsu
echo "host $(hostname) job ${SLURM_JOB_ID:-none} start $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee job_info.txt
sha256sum e0_code.tar.gz | tee staged_sha256.txt
rm -rf $C && mkdir -p $C && tar -xzf e0_code.tar.gz -C $C --strip-components=1
(cd $C && sha256sum *.py) | tee code_sha256.txt
(cd $D && sha256sum e0_*.npz) | tee data_sha256.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN="apptainer exec --nv -B $SCRATCH_ROOT $IMG env PYTHONPATH=$REPO/src:$C OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $VENV/bin/python"
$RUN -c "import torch; print('cuda_available', torch.cuda.is_available(), 'n_gpu', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])" | tee env_check.txt
PORT=$((29500 + ${SLURM_JOB_ID:-0} % 1000))
LAUNCH="$RUN -m accelerate.commands.launch --num_processes 2 --multi_gpu --main_process_port $PORT"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 15 > gpumem.log 2>&1 &
SMI=$!
COMMON="--e0-train-npz $D/e0_train.npz --condition rendered --sequence-length 320 --image-size 64 --sampling-timesteps 50 \
 --train-fraction 0.99 --num-workers 6 --wandb-mode disabled --random-seed 20261016 --sample-count 25 --classes 0 \
 --save-every 150 --validation-every 150"
SM=$B/runs/e0/smoke2_${SLURM_JOB_ID:-x}
BS=256; GA=2
T0=$(date +%s)
if $LAUNCH $C/e0_train.py $COMMON --batch-size $BS --gradient-accumulate-every $GA --run-dir $SM --train-steps 300 > train_a.log 2>&1; then
  echo "2gpu bs256 ga2 (128/gpu) ok" | tee smoke_status.txt
else
  echo "2gpu bs256 ga2 FAILED -> 2gpu bs128 ga4 (64/gpu)" | tee smoke_status.txt
  tr '\r' '\n' < train_a.log | grep -a -E "Error|error" | tail -3 | tee -a smoke_status.txt || true
  SM=${SM}_bs128; BS=128; GA=4
  T0=$(date +%s)
  $LAUNCH $C/e0_train.py $COMMON --batch-size $BS --gradient-accumulate-every $GA --run-dir $SM --train-steps 300 > train_b.log 2>&1 || echo "bs128 FAILED" | tee -a smoke_status.txt
fi
T1=$(date +%s); echo "train_300_steps_seconds $((T1-T0)) bs $BS ga $GA" | tee -a smoke_status.txt
$LAUNCH $C/e0_train.py $COMMON --batch-size $BS --gradient-accumulate-every $GA --run-dir $SM --train-steps 400 > train_resume.log 2>&1 || echo "resume FAILED" | tee -a smoke_status.txt
T2=$(date +%s); echo "resume_100_steps_seconds $((T2-T1))" | tee -a smoke_status.txt
grep -a "Resumed from" train_resume.log | tee -a smoke_status.txt || true
ls -la $SM $SM/checkpoints | tee run_listing.txt || true
cp $SM/run_config*.json . || true
cp $SM/training_samples/generated/*000300* . 2>/dev/null || true
T3=$(date +%s)
$RUN $C/e0_sample.py --run-dir $SM --milestone 2 --npz $D/e0_val.npz --condition rendered --sampling-timesteps 50 \
  --ddim-eta 0 --batch-size 256 --seed 0 --subset 512 --subset-seed 0 --out smoke2_val512_preds.npz 2>&1 | grep -av "sampling loop" | tail -3 | tee sample_smoke.log || true
T4=$(date +%s); echo "sample_512_ddim50_seconds $((T4-T3))" | tee -a smoke_status.txt
$RUN $C/e0_score.py e0 --npz $D/e0_val.npz --preds smoke2_val512_preds.npz --method e0_smoke2_step300 --input rendered \
  --out smoke2_val512_rows.csv --procs 12 2>&1 | tail -3 || true
T5=$(date +%s); echo "score_512_seconds $((T5-T4))" | tee -a smoke_status.txt
$RUN $C/e0_score.py summary --rows smoke2_val512_rows.csv --out smoke2_val512_summary.csv 2>&1 | tail -2 || true
kill $SMI 2>/dev/null || true
for f in train_a.log train_b.log train_resume.log; do [ -f $f ] && (tr '\r' '\n' < $f | grep -av "sampling loop" | grep -a -E "it/s|s/it|Resumed|number of|Error" | awk 'NR%20==1' | tail -20 > ${f%.log}_tail.txt) || true; done
echo "end $(date -Is)" | tee -a job_info.txt
ls -lh
