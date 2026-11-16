#!/bin/bash
# M1 (A1.2): E1-v2 selectivity run (exp/E1_v2 code + uhwr_eval 0.3.0 tarball unchanged) on the 7 RQF test sets, then the leak table.
# usage: bash run_m1.sh both   (venv+code copied once to /dev/shm with parallel cp; smoke (5 units/set) then full in the same job; full only if smoke rc=0)
set -eo pipefail
exec 2>&1
U=$U; X=M1_selectivity; R=$U/exp/RQF/$X; SRC=$U/exp/E1_v2; IMG=$U/images/torch27.sif; VE=$U/venvs/uhwr_eval02
NCPU=${SLURM_CPUS_PER_TASK:-8}
export TMPDIR=/tmp/${X}_${SLURM_JOB_ID:-x}; mkdir -p $TMPDIR $R/bundle $R/smoke $R/out
if [ ! -d $R/code ]; then cp -a $SRC/code $R/code; rm -rf $R/code/lib/uhwr_eval/__pycache__ $R/code/lib/uhwr_exp/__pycache__ || true; fi
cp $R/addon/build_m1_units.py $R/addon/prov.py $R/addon/leak_table.py $R/addon/run_m1.sh $R/code/
SH=/dev/shm/m1_${SLURM_JOB_ID:-x}; mkdir -p $SH; trap 'rm -rf $SH' EXIT
pcopy(){ mkdir -p $2; (cd $1 && find . -mindepth 1 -type d | sed 's|^\./||' | xargs -d '\n' -P 16 -n 200 -I{} mkdir -p $2/{}; find . -mindepth 1 ! -type d -print0 | xargs -0 -P 32 -n 40 cp -a --parents -t $2); }
echo "[$(date -u +%FT%TZ)] parallel copy venv+code to $SH start host=$(hostname) slurm=${SLURM_JOB_ID:-}" | tee -a $R/out/steps.log
pcopy $VE $SH/venv; pcopy $R/code $SH/code; VS=$SH/venv
echo "[$(date -u +%FT%TZ)] copy done $(du -sm $SH | cut -f1) MB" | tee -a $R/out/steps.log
PY="apptainer exec -B $SCRATCH_ROOT -B /tmp -B /dev/shm $IMG env PYTHONPATH=$SH/code/lib:$SH/code TORCH_HOME=$U/cache/torch PYTHONNOUSERSITE=1 NUMBA_CACHE_DIR=$TMPDIR/numba MPLCONFIGDIR=$TMPDIR/mpl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 UHWR_TORCH_THREADS=1 CUDA_VISIBLE_DEVICES= SLURM_CPUS_PER_TASK=$NCPU $VS/bin/python"
(cd $R/code && tar -czf $R/code_$X.tar.gz --sort=name --mtime='2026-11-07 00:00Z' --owner=0 --group=0 --numeric-owner --exclude __pycache__ .)
SETS="ouhdl casia_zh casia_en tamil iam500 unipen_de onhw"
domode(){
MODE=$1
if [ "$MODE" = smoke ]; then O=$R/smoke; LIM="--limit 5"; BOOT=200; else O=$R/out; LIM=""; BOOT=2000; fi
(cd $SRC/code && find . -name '*.py' -not -path '*__pycache__*' | sort | xargs sha256sum) > $O/code_sha256_E1_v2_source.txt
(cd $R/code && find . -name '*.py' -not -path '*__pycache__*' | sort | xargs sha256sum) > $O/code_sha256_M1.txt
(cd $R/code/lib && tar --sort=name --mtime='2026-10-19 00:00Z' --owner=0 --group=0 --numeric-owner --exclude __pycache__ -cf $O/uhwr_eval-0.3.0.tar uhwr_eval); gzip -nf $O/uhwr_eval-0.3.0.tar
sha256sum $O/uhwr_eval-0.3.0.tar.gz $SRC/out/uhwr_eval-0.3.0.tar.gz | tee $O/uhwr_eval_tarball_check.txt
A=$(sha256sum $O/uhwr_eval-0.3.0.tar.gz | cut -c1-64); B=$(sha256sum $SRC/out/uhwr_eval-0.3.0.tar.gz | cut -c1-64)
[ "$A" = "$B" ] || { echo "uhwr_eval tarball differs from E1_v2's"; exit 3; }
HPY=$(command -v python3 || echo ""); PROV="$HPY $R/code/prov.py --exp $X --out $O/PROVENANCE_$X.json --run-dir $O --code-tar $R/code_$X.tar.gz"
echo "[$(date -u +%FT%TZ)] start $X mode=$MODE host=$(hostname) slurm=${SLURM_JOB_ID:-} ncpu=$NCPU" | tee -a $O/steps.log
[ -f $R/bundle/e1_units.pkl ] || (cd $SH/code && $PY build_m1_units.py --out $R/bundle 2>&1 | tee $O/build.log)
cp $R/bundle/m1_units_info.json $O/
CMD="python e1_v2_run.py --bundle $R/bundle --out $O --datasets $SETS --workers $NCPU --boot $BOOT $LIM"
$PROV --stage start --inputs $R/bundle/e1_units.pkl $R/bundle/m1_units_info.json $O/uhwr_eval-0.3.0.tar.gz $SRC/out/PROVENANCE_E1_v2.json \
  --params "{\"mode\":\"$MODE\",\"boot\":$BOOT,\"boot_seed\":0,\"seed\":20261016,\"evaluator\":\"uhwr_eval 0.3.0 (tarball identical to exp/E1_v2/out)\",\"datasets\":[\"ouhdl\",\"casia_zh\",\"casia_en\",\"tamil\",\"iam500\",\"unipen_de\",\"onhw\"],\"units\":\"500 per set from the RQF GT npz (see m1_units_info.json); numpy default_rng(20261016) choice sorted when pool > 500\",\"perturbations\":\"uhwr_exp.e1_run.FACTORS (7 types = 5 factors x 5 strengths), unchanged\",\"protocols\":[\"native\",\"standardised\"],\"lpips\":true,\"rule\":\"A1.2: off-factor leak <= 0.10 (native), leak = max over off-factor perturbations |delta5| / max over own-factor perturbations |delta5|\",\"venv_copy\":\"venv+code copied to /dev/shm with parallel cp (BeeGFS small-file latency); contents identical\"}" \
  --seeds 20261016 0 --command "$CMD" --venv $VE --image $IMG \
  --deviations "Units rebuilt from the RQF GT npz (not the legacy e_bundle_v1 units): ink = gt_points on the 64x64 canvas; no timestamps except UNIPEN-de/OnHW" "OnHW cluster = unit (2 writers)" "OUHD-L, UNIPEN-de, OnHW: 500-unit sample of the test pool"
cd $SH/code; $PY e1_v2_run.py --bundle $R/bundle --out $O --datasets $SETS --workers $NCPU --boot $BOOT $LIM 2>&1 | tee $O/run.log
echo "[$(date -u +%FT%TZ)] run finished rc=0 mode=$MODE" | tee -a $O/steps.log
$PY leak_table.py --sens $O/e1_v2_sensitivity.csv --out $O 2>&1 | tee $O/leak.log
$PROV --stage end --outputs "$O/*.csv" "$O/*.csv.gz" "$O/*.json" "$O/*.txt" "$O/*.md" "$O/run.log" "$O/uhwr_eval-0.3.0.tar.gz" --notes "end of $MODE run $(date -u +%FT%TZ)"
}
domode smoke
domode full
ls -la $R/out
