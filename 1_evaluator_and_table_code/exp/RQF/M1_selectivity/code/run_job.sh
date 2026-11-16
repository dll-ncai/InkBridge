#!/bin/bash
# E1_v2: job driver (smoke|full). Writes outputs + PROVENANCE_E1_v2.json into $O. Times recorded UTC.
set -eo pipefail
MODE=${1:-smoke}
U=$U; X=E1_v2; R=$U/exp/$X; IMG=$U/images/torch27.sif; VE=$U/venvs/uhwr_eval02
if [ "$MODE" = smoke ]; then O=$R/smoke; LIM="--limit 5"; BOOT=200; else O=$R/out; LIM=""; BOOT=2000; fi
mkdir -p $O
NCPU=${SLURM_CPUS_PER_TASK:-16}
export TMPDIR=/tmp/${X}_${SLURM_JOB_ID:-x}; mkdir -p $TMPDIR
PY="apptainer exec -B $SCRATCH_ROOT -B /tmp $IMG env PYTHONPATH=$R/code/lib:$R/code TORCH_HOME=$U/cache/torch PYTHONNOUSERSITE=1 NUMBA_CACHE_DIR=$TMPDIR/numba MPLCONFIGDIR=$TMPDIR/mpl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 UHWR_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= SLURM_CPUS_PER_TASK=$NCPU $VE/bin/python"
HPY=$(command -v python3 || echo "")
if [ -n "$HPY" ]; then PROV="$HPY $R/code/prov.py"; else PROV="$PY $R/code/prov.py"; fi
PROV="$PROV --exp $X --out $O/PROVENANCE_$X.json --run-dir $O --code-tar $R/code_$X.tar.gz"
echo "[$(date -u +%FT%TZ)] start $X mode=$MODE host=$(hostname) slurm=${SLURM_JOB_ID:-} ncpu=$NCPU" | tee -a $O/steps.log

EV3=$R/code/lib/uhwr_eval; (cd $R/code/lib && tar --sort=name --mtime='2026-10-19 00:00Z' --owner=0 --group=0 --numeric-owner --exclude __pycache__ -cf $O/uhwr_eval-0.3.0.tar uhwr_eval) ; gzip -nf $O/uhwr_eval-0.3.0.tar
cp $R/code/lib/uhwr_eval_0.2.0_to_0.3.0.diff $O/
CMD="python e1_v2_run.py --bundle $U/data/e_bundle_v1 --out $O --datasets ouhdl casia tamil --workers $NCPU --boot $BOOT $LIM"
$PROV --stage start --inputs $U/data/e_bundle_v1/e1_units.pkl $U/data/e_bundle_v1/PROVENANCE_e_inputs.json $O/uhwr_eval-0.3.0.tar.gz $O/uhwr_eval_0.2.0_to_0.3.0.diff \
  --params "{\"mode\":\"$MODE\",\"boot\":$BOOT,\"boot_seed\":0,\"seed\":20261016,\"evaluator\":\"uhwr_eval 0.3.0 (new: penup_f1_v2, timing_profile_tv_v2)\",\"datasets\":[\"ouhdl\",\"casia\",\"tamil\"],\"perturbations\":\"uhwr_exp.e1_run.FACTORS (7 types = 5 factors x 5 strengths)\",\"protocols\":[\"native\",\"standardised\"],\"lpips\":true}" \
  --seeds 20261016 --command "$CMD" --venv $VE --image $IMG \
  --deviations "IAM components arm not run: no IAM component units exist yet (output of X3, not queued here: dependent)" "CASIA units carry no timestamps: timing metrics NaN for CASIA" "HPL Tamil 500 units (from the v1 bundle) run in addition to the plan's datasets"
cd $R/code; $PY e1_v2_run.py --bundle $U/data/e_bundle_v1 --out $O --datasets ouhdl casia tamil --workers $NCPU --boot $BOOT $LIM 2>&1 | tee $O/run.log

echo "[$(date -u +%FT%TZ)] run finished rc=0" | tee -a $O/steps.log
$PROV --stage end --outputs "$O/*.csv" "$O/*.csv.gz" "$O/*.json" "$O/run.log" --notes "end of $MODE run $(date -u +%FT%TZ)"
ls -la $O
