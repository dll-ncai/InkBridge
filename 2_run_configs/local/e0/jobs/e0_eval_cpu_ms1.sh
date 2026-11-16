#SBATCH -p epyc-256
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=08:00:00
# E0 INTERIM evaluation on CPU (PARTIAL TRAINING: checkpoint 1 = step 10,000 of the 100k main run, Slurm 24417278).
# GPU eval jobs were estimated to start only after ~1.7 days (Priority), so the 10k checkpoint is sampled on 8 CPU
# nodes: every task takes units shard::8 of each set.  DDIM-50, eta 0, EMA weights, torch seed 0 + 1000*shard.
# Sets: all 4,406 v3 test units and the 354 gold units, inputs rendered and dataset img/.  No checkpoint selection.
set -u
B=$U
IMG=$B/images/torch27.sif
VENV=$B/venvs/e0
D=$B/data/ouhdl_e0_v3
CODE_TAG=4a031a0d
C=$B/e0/code_$CODE_TAG
REPO=$B/repos/nagamatsu
RUNDIR=$B/runs/e0/e0_v3_rendered_100k_bs512
MS=1
EV=$RUNDIR/eval_partial_cpu_ms$MS
mkdir -p $EV
echo "host $(hostname) job ${SLURM_JOB_ID:-none} nodes ${SLURM_JOB_NODELIST:-?} start $(date -Is)" | tee job_info.txt
[ -d $C ] || { mkdir -p $C && tar -xzf e0_code_v3.tar.gz -C $C --strip-components=1; }
(cd $C && sha256sum *.py) > code_sha256.txt
(cd $D && sha256sum e0_*.npz) > data_sha256.txt
for k in $(seq 1 120); do [ -f $RUNDIR/checkpoints/checkpoint_00000$MS.pt ] && break; sleep 60; done
ls -la $RUNDIR/checkpoints | tee ckpt_listing.txt
sha256sum $RUNDIR/checkpoints/checkpoint_00000$MS.pt | tee ckpt_sha256.txt
cp $RUNDIR/run_config.json . || true
cat > worker.sh <<EOF
set -u
K=\$SLURM_PROCID; N=\$SLURM_NTASKS
RUN="apptainer exec -B $SCRATCH_ROOT $IMG env PYTHONPATH=$REPO/src:$C OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 $VENV/bin/python"
for S in test gold; do for COND in rendered dataset; do
  P=$EV/\${S}_\${COND}_ms${MS}_ddim50_preds_shard\${K}of\${N}.npz; R=$EV/\${S}_\${COND}_ms${MS}_ddim50_rows_shard\${K}of\${N}.csv
  [ -f \$R ] && continue
  \$RUN $C/e0_sample.py --run-dir $RUNDIR --milestone $MS --npz $D/e0_\${S}.npz --condition \$COND --sampling-timesteps 50 \
     --ddim-eta 0 --batch-size 64 --seed 0 --shard \$K --nshards \$N --out \$P 2>&1 | grep -av "sampling loop" | tail -1
  \$RUN $C/e0_score.py e0 --npz $D/e0_\${S}.npz --preds \$P --method e0_v3_ms${MS}_step10k_ddim50 --input \$COND --out \$R --procs 16 2>&1 | tail -1
done; done
echo "worker \$K done \$(date -Is) on \$(hostname)"
EOF
srun --ntasks=$SLURM_NTASKS --ntasks-per-node=1 --cpus-per-task=16 bash worker.sh 2>&1 | tee workers.log
RUN1="apptainer exec -B $SCRATCH_ROOT $IMG env PYTHONPATH=$REPO/src:$C $VENV/bin/python"
$RUN1 - <<EOF
import glob, pandas as pd, numpy as np, json
ev = "$EV"
for S in ("test", "gold"):
    for COND in ("rendered", "dataset"):
        fs = sorted(glob.glob(f"{ev}/{S}_{COND}_ms${MS}_ddim50_rows_shard*.csv"))
        d = pd.concat([pd.read_csv(f) for f in fs])
        d.to_csv(f"{ev}/{S}_{COND}_ms${MS}_ddim50_rows.csv", index=False)
        ps = sorted(glob.glob(f"{ev}/{S}_{COND}_ms${MS}_ddim50_preds_shard*.npz"))
        zs = [np.load(p) for p in ps]
        np.savez_compressed(f"{ev}/{S}_{COND}_ms${MS}_ddim50_preds.npz", preds=np.concatenate([z["preds"] for z in zs]),
                            unit_ids=np.concatenate([z["unit_ids"] for z in zs]),
                            meta=json.dumps([json.loads(str(z["meta"])) for z in zs]))
        print(S, COND, len(fs), "shards", d.unit_id.nunique(), "units", int((d.error.fillna("") != "").sum()), "errors")
EOF
$RUN1 $C/e0_score.py summary --rows $EV/test_rendered_ms${MS}_ddim50_rows.csv $EV/test_dataset_ms${MS}_ddim50_rows.csv \
   $EV/gold_rendered_ms${MS}_ddim50_rows.csv $EV/gold_dataset_ms${MS}_ddim50_rows.csv --out $EV/summary_e0_ms${MS}_partial_cpu.csv
cp $EV/summary_e0_ms${MS}_partial_cpu.csv $EV/*_ddim50_rows.csv . || true
cp $EV/test_rendered_ms${MS}_ddim50_preds.npz $EV/test_dataset_ms${MS}_ddim50_preds.npz $EV/gold_*_ms${MS}_ddim50_preds.npz . || true
rm -f ./*_shard*.csv || true
echo "end $(date -Is)" | tee -a job_info.txt
ls -lh
