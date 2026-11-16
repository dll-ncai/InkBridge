#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=05:00:00
# E0 evaluation only (phases 2-3 of e0_main.sh, same code) on the checkpoints that exist NOW in RUNDIR.
# Used for interim (PARTIAL-TRAINING) results while the main run (Slurm 24417278) continues; writes to
# RUNDIR/eval_partial_upto_ms<MAXMS>/ so the main run's own eval/ directory is untouched.
set -u
B=$U
IMG=$B/images/torch27.sif
VENV=$B/venvs/e0
D=$B/data/ouhdl_e0_v3
CODE_TAG=4cd40f57
C=$B/e0/code_$CODE_TAG
REPO=$B/repos/nagamatsu
RUNDIR=$B/runs/e0/e0_v3_rendered_100k_bs512
MAXMS=1
EV=$RUNDIR/eval_partial_upto_ms$MAXMS
mkdir -p $EV
echo "host $(hostname) job ${SLURM_JOB_ID:-none} start $(date -Is)" | tee job_info.txt
[ -d $C ] || { mkdir -p $C && tar -xzf e0_code_v2.tar.gz -C $C --strip-components=1; }
(cd $C && sha256sum *.py) > code_sha256.txt
(cd $D && sha256sum e0_*.npz) > data_sha256.txt
[ -f $RUNDIR/training_complete.json ] && echo "training_complete: $(cat $RUNDIR/training_complete.json)" | tee eval_status.txt || echo "training NOT complete: PARTIAL-TRAINING result" | tee eval_status.txt
RUN="apptainer exec --nv -B $SCRATCH_ROOT $IMG env PYTHONPATH=$REPO/src:$C OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $VENV/bin/python"
for k in $(seq 1 180); do [ -f $RUNDIR/checkpoints/checkpoint_000001.pt ] && break; sleep 60; done   # wait (<= 3 h) for the 10k checkpoint
ls -la $RUNDIR/checkpoints | tee ckpt_listing.txt
# ---------------- phase 2: checkpoint selection on val
for f in $RUNDIR/checkpoints/checkpoint_*.pt; do
  m=$(basename $f .pt); m=$((10#${m#checkpoint_}))
  [ $m -le $MAXMS ] || continue
  [ -f $EV/val1000_ms${m}_rows.csv ] && continue
  $RUN $C/e0_sample.py --run-dir $RUNDIR --milestone $m --npz $D/e0_val.npz --condition rendered --sampling-timesteps 50 \
     --ddim-eta 0 --batch-size 256 --seed 0 --subset 1000 --subset-seed 0 --out $EV/val1000_ms${m}_preds.npz 2>&1 | grep -av "sampling loop" | tail -2
  $RUN $C/e0_score.py e0 --npz $D/e0_val.npz --preds $EV/val1000_ms${m}_preds.npz --method e0_ms${m} --input rendered \
     --out $EV/val1000_ms${m}_rows.csv --procs 14 2>&1 | tail -1
done
$RUN $C/e0_score.py summary --rows $EV/val1000_ms*_rows.csv --out $EV/val1000_summary.csv
BEST=$($RUN -c "
import pandas as pd, json
d = pd.read_csv('$EV/val1000_summary.csv'); d = d[d.protocol.astype(str).str.startswith('native')]
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
cp $RUNDIR/run_config*.json $RUNDIR/job_history.txt . 2>/dev/null || true
echo "end $(date -Is)"
ls -lh
