#SBATCH -p gpuidle
#SBATCH --gres=gpu:a30:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
# E0 smoke 3.  Smoke 1 (24415666): batch 512 / 256 on one A30 -> CUDA OOM at step 0.
# Smoke 2 (24416132): 2x A30 via accelerate -> NCCL all-reduce timeout (rank 1 never completed a collective).
# Here: (a) 1 GPU, batch 128 x grad-accum 4 (= 512), 200 steps;  (b) 2 GPUs with NCCL_P2P_DISABLE=1,
# batch 256 x grad-accum 2 (128/GPU), 200 steps (NCCL debug on); resume check (+50) and DDIM-50 on 512 val
# units with whichever run produced a checkpoint.
set -u
B=$U
IMG=$B/images/torch27.sif
VENV=$B/venvs/e0
D=$B/data/ouhdl_e0_v3
CODE_TAG=1a5ee72e
C=$B/e0/code_$CODE_TAG
REPO=$B/repos/nagamatsu
echo "host $(hostname) job ${SLURM_JOB_ID:-none} start $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee job_info.txt
[ -d $C ] || { mkdir -p $C && tar -xzf e0_code.tar.gz -C $C --strip-components=1; }
nvidia-smi topo -m | tee topo.txt || true
RUNB="apptainer exec --nv -B $SCRATCH_ROOT $IMG env PYTHONPATH=$REPO/src:$C OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 10 > gpumem.log 2>&1 &
SMI=$!
COMMON="--e0-train-npz $D/e0_train.npz --condition rendered --sequence-length 320 --image-size 64 --sampling-timesteps 50 \
 --train-fraction 0.99 --num-workers 6 --wandb-mode disabled --random-seed 20261016 --sample-count 25 --classes 0 \
 --save-every 100 --validation-every 100"
# (a) single GPU
SA=$B/runs/e0/smoke3a_${SLURM_JOB_ID:-x}
T0=$(date +%s)
$RUNB CUDA_VISIBLE_DEVICES=0 $VENV/bin/python $C/e0_train.py $COMMON --batch-size 128 --gradient-accumulate-every 4 --run-dir $SA --train-steps 200 > train_a.log 2>&1 && echo "a_ok" | tee smoke_status.txt || echo "a_FAILED" | tee smoke_status.txt
T1=$(date +%s); echo "a_1gpu_bs128_ga4_200_steps_seconds $((T1-T0))" | tee -a smoke_status.txt
# (b) two GPUs, P2P disabled
SBR=$B/runs/e0/smoke3b_${SLURM_JOB_ID:-x}
PORT=$((29500 + ${SLURM_JOB_ID:-0} % 1000))
T2=$(date +%s)
$RUNB NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN $VENV/bin/python -m accelerate.commands.launch --num_processes 2 --multi_gpu --main_process_port $PORT \
  $C/e0_train.py $COMMON --batch-size 256 --gradient-accumulate-every 2 --run-dir $SBR --train-steps 200 > train_b.log 2>&1 && echo "b_ok" | tee -a smoke_status.txt || echo "b_FAILED" | tee -a smoke_status.txt
T3=$(date +%s); echo "b_2gpu_p2poff_bs256_ga2_200_steps_seconds $((T3-T2))" | tee -a smoke_status.txt
# resume + sampling on (a)
if [ -f $SA/checkpoints/checkpoint_000002.pt ]; then
  T4=$(date +%s)
  $RUNB CUDA_VISIBLE_DEVICES=0 $VENV/bin/python $C/e0_train.py $COMMON --batch-size 128 --gradient-accumulate-every 4 --run-dir $SA --train-steps 250 > train_a_resume.log 2>&1 || echo "a_resume_FAILED" | tee -a smoke_status.txt
  T5=$(date +%s); echo "a_resume_50_steps_seconds $((T5-T4))" | tee -a smoke_status.txt
  grep -a "Resumed from" train_a_resume.log | tee -a smoke_status.txt || true
  $RUNB CUDA_VISIBLE_DEVICES=0 $VENV/bin/python $C/e0_sample.py --run-dir $SA --milestone 2 --npz $D/e0_val.npz --condition rendered --sampling-timesteps 50 \
    --ddim-eta 0 --batch-size 256 --seed 0 --subset 512 --subset-seed 0 --out smoke3_val512_preds.npz 2>&1 | grep -av "sampling loop" | tail -2 | tee sample_smoke.log || true
  T6=$(date +%s); echo "sample_512_ddim50_seconds $((T6-T5))" | tee -a smoke_status.txt
  $RUNB $VENV/bin/python $C/e0_score.py e0 --npz $D/e0_val.npz --preds smoke3_val512_preds.npz --method e0_smoke3_step200 --input rendered \
    --out smoke3_val512_rows.csv --procs 14 2>&1 | tail -1 || true
  T7=$(date +%s); echo "score_512_seconds $((T7-T6))" | tee -a smoke_status.txt
  $RUNB $VENV/bin/python $C/e0_score.py summary --rows smoke3_val512_rows.csv --out smoke3_val512_summary.csv 2>&1 | tail -1 || true
  cp $SA/run_config.json run_config_a.json || true
fi
kill $SMI 2>/dev/null || true
for f in train_a.log train_b.log train_a_resume.log; do [ -f $f ] && (tr '\r' '\n' < $f | grep -av "sampling loop" | grep -a -E "it/s|s/it|Resumed|number of|Error|NCCL WARN|out of memory" | awk 'NR%10==1' | tail -30 > ${f%.log}_tail.txt) || true; done
grep -v " 0 MiB" gpumem.log | sort -t, -k3 -n | tail -4 > gpumem_peak.txt || true
echo "end $(date -Is)" | tee -a job_info.txt
ls -lh
