#SBATCH --gres=gpu:a30:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
# E0 main run (REDUCED BUDGET): Nagamatsu diffusion trained on OUHD-L ligature units v3.0, rendered input.
# Phase 1: train 100000 steps, 512 samples per optimiser step, checkpoint every 10k,
#          resumes automatically from the latest checkpoint in RUNDIR (resubmitting this script continues the run).
# Phase 2: checkpoint selection on 1,000 random v3 val units (subset seed 0): DDIM-50, eta 0, sample seed 0;
#          criterion = lowest mean native LDTW (ties -> higher AIoU). Every evaluated checkpoint is reported.
# Phase 3: best checkpoint on all 4,406 v3 test units and the 354 gold units, inputs rendered + dataset img/.
# Every phase skips work whose outputs already exist (idempotent).
set -u
B=$U
IMG=$B/images/torch27.sif
VENV=$B/venvs/e0
D=$B/data/ouhdl_e0_v3
CODE_TAG=1a5ee72e
C=$B/e0/code_$CODE_TAG
REPO=$B/repos/nagamatsu
RUNDIR=$B/runs/e0/e0_v3_rendered_100k_bs512
EV=$RUNDIR/eval
mkdir -p $RUNDIR $EV
echo "host $(hostname) job ${SLURM_JOB_ID:-none} start $(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee -a $RUNDIR/job_history.txt | tee job_info.txt
if [ ! -d $C ]; then mkdir -p $C && tar -xzf e0_code.tar.gz -C $C --strip-components=1; fi
(cd $C && sha256sum *.py) > code_sha256.txt
(cd $D && sha256sum e0_*.npz) > data_sha256.txt
git -C $REPO rev-parse HEAD > repo_commit.txt || true
apptainer exec -B $SCRATCH_ROOT $IMG $VENV/bin/pip freeze > pip_freeze.txt
cp code_sha256.txt data_sha256.txt repo_commit.txt pip_freeze.txt $RUNDIR/ || true
RUN="apptainer exec --nv -B $SCRATCH_ROOT $IMG env PYTHONPATH=$REPO/src:$C OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $VENV/bin/python"
$RUN -c "import torch; print('cuda_available', torch.cuda.is_available(), torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])" | tee env_check.txt
PORT=$((29500 + ${SLURM_JOB_ID:-0} % 1000))
LAUNCH="$RUN -m accelerate.commands.launch --num_processes 2 --multi_gpu --main_process_port $PORT"
# ---------------- phase 1: train
# attempt 1: 2 GPUs (NCCL_P2P_DISABLE=1; P2P hung in smoke 2), global batch 128 = 64/GPU, grad-accum 4 -> 512 per step
# fallback : 1 GPU, batch 128, grad-accum 4 -> 512 per step (verified in smoke 3), resuming from the latest checkpoint
TRAINARGS="--e0-train-npz $D/e0_train.npz --condition rendered --run-dir $RUNDIR \
    --batch-size 128 --gradient-accumulate-every 4 --train-steps 100000 --save-every 10000 --validation-every 1000 \
    --sequence-length 320 --image-size 64 --sampling-timesteps 50 --train-fraction 0.99 --num-workers 6 \
    --wandb-mode disabled --random-seed 20261016 --sample-count 25 --classes 0 --notes E0_reduced_budget_OUHD-L_v3"
J=${SLURM_JOB_ID:-x}
if [ ! -f $RUNDIR/training_complete.json ]; then
  echo "train attempt 2gpu start $(date -Is)" | tee -a $RUNDIR/job_history.txt
  if apptainer exec --nv -B $SCRATCH_ROOT $IMG env PYTHONPATH=$REPO/src:$C OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_P2P_DISABLE=1 \
       $VENV/bin/python -m accelerate.commands.launch --num_processes 2 --multi_gpu --main_process_port $PORT \
       $C/e0_train.py $TRAINARGS > train_${J}_2gpu.log 2>&1; then
    echo "train 2gpu finished $(date -Is)" | tee -a $RUNDIR/job_history.txt
  else
    echo "train 2gpu FAILED $(date -Is); falling back to 1 GPU" | tee -a $RUNDIR/job_history.txt
    tr '\r' '\n' < train_${J}_2gpu.log | grep -a -E "Error|error" | tail -3 | tee -a $RUNDIR/job_history.txt || true
    $RUN $C/e0_train.py $TRAINARGS > train_${J}_1gpu.log 2>&1
    echo "train 1gpu finished $(date -Is)" | tee -a $RUNDIR/job_history.txt
  fi
  for f in train_${J}_*.log; do tr '\r' '\n' < $f | grep -av "sampling loop" | awk 'NR%500==1' > $RUNDIR/${f%.log}_progress.txt || true; done
fi
ls -la $RUNDIR/checkpoints | tee ckpt_listing.txt
# ---------------- phase 2: checkpoint selection on val
for f in $RUNDIR/checkpoints/checkpoint_*.pt; do
  m=$(basename $f .pt); m=$((10#${m#checkpoint_}))
  [ -f $EV/val1000_ms${m}_rows.csv ] && continue
  $RUN $C/e0_sample.py --run-dir $RUNDIR --milestone $m --npz $D/e0_val.npz --condition rendered --sampling-timesteps 50 \
     --ddim-eta 0 --batch-size 256 --seed 0 --subset 1000 --subset-seed 0 --out $EV/val1000_ms${m}_preds.npz 2>&1 | grep -av "sampling loop" | tail -2
  $RUN $C/e0_score.py e0 --npz $D/e0_val.npz --preds $EV/val1000_ms${m}_preds.npz --method e0_ms${m} --input rendered \
     --out $EV/val1000_ms${m}_rows.csv --procs 14 2>&1 | tail -1
done
$RUN $C/e0_score.py summary --rows $EV/val1000_ms*_rows.csv --out $EV/val1000_summary.csv
BEST=$($RUN -c "
import pandas as pd, json
d = pd.read_csv('$EV/val1000_summary.csv'); d = d[d.protocol == 'native']
p = d.pivot_table(index='method', columns='metric', values='mean')
p = p.sort_values(['ldtw', 'aiou'], ascending=[True, False])
best = p.index[0]
json.dump(dict(criterion='lowest mean native LDTW on 1000 random val units (subset seed 0), ties -> higher AIoU; DDIM-50 eta 0 seed 0',
               best=best, table=p[['ldtw', 'aiou', 'dtw', 'strict_order_acc', 'direction_acc']].round(5).reset_index().to_dict('records')),
          open('$EV/selection.json', 'w'), indent=1)
print(int(best.replace('e0_ms', '')))")
echo "best milestone $BEST" | tee -a $RUNDIR/job_history.txt
# ---------------- phase 3: test + gold with the best checkpoint
for S in test gold; do for COND in rendered dataset; do
  P=$EV/${S}_${COND}_ms${BEST}_ddim50_preds.npz; R=$EV/${S}_${COND}_ms${BEST}_ddim50_rows.csv
  [ -f $R ] && continue
  $RUN $C/e0_sample.py --run-dir $RUNDIR --milestone $BEST --npz $D/e0_${S}.npz --condition $COND --sampling-timesteps 50 \
     --ddim-eta 0 --batch-size 256 --seed 0 --out $P 2>&1 | grep -av "sampling loop" | tail -2
  $RUN $C/e0_score.py e0 --npz $D/e0_${S}.npz --preds $P --method e0_v3_ms${BEST}_ddim50 --input $COND --out $R --procs 14 2>&1 | tail -1
done; done
$RUN $C/e0_score.py summary --rows $EV/test_*_ms${BEST}_ddim50_rows.csv $EV/gold_*_ms${BEST}_ddim50_rows.csv --out $EV/summary_e0_ms${BEST}.csv
cp $EV/*.csv $EV/*.json . 2>/dev/null || true
cp $EV/test_*_ms${BEST}_ddim50_preds.npz $EV/gold_*_ms${BEST}_ddim50_preds.npz . 2>/dev/null || true
cp $RUNDIR/run_config*.json $RUNDIR/train_*_progress.txt $RUNDIR/job_history.txt . 2>/dev/null || true
echo "end $(date -Is)" | tee -a $RUNDIR/job_history.txt
ls -lh
