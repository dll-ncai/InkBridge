#!/bin/bash
# K1 frame kill check: LUO ms6 zero-shot on OUHD-L val250 (first 250 of e0_val) + test 4,406 in identity / mirror_x / vt
# (inverse applied before scoring) + IAM val250 in mirror_x and identity (symmetric LTR control); seeds 0 1 2; e0_score.py
# native+standardised rows; frame_k1_summary.py readout (means, writer-cluster CIs, paired deltas, kill rule, loop subset).
# Caller sets PYG PYC CODE STAGE OUT [SEEDS="0 1 2"] [PROCS=14] [SMOKE_UNITS=""] [SETS="val250_ouhdl test_ouhdl"] [LTR_SETS="val250_iam"]
set -uo pipefail
: "${PYG:?}" "${PYC:?}" "${CODE:?}" "${STAGE:?}" "${OUT:?}"
SEEDS=${SEEDS:-"0 1 2"}; PROCS=${PROCS:-14}; SMOKE_UNITS=${SMOKE_UNITS:-}; SETS=${SETS:-"val250_ouhdl test_ouhdl"}; LTR_SETS=${LTR_SETS:-"val250_iam"}
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }; mkdir -p $OUT; log() { echo "$(utc) $* host=$(hostname) slurm=${SLURM_JOB_ID:-none}" | tee -a $OUT/job_history.txt; }
(cd $CODE && sha256sum -c --quiet SHA256SUMS.txt) || { log "CODE_SHA_MISMATCH"; exit 3; }
SMK=(); [ -n "$SMOKE_UNITS" ] && SMK=(--max-units $SMOKE_UNITS); SFX=""; [ -n "$SMOKE_UNITS" ] && SFX="_smoke${SMOKE_UNITS}"
OU=$STAGE/data/ouhdl_e0_v3; D=$STAGE/data/m5_multiscript_v1
npz_of() { case $1 in val250_ouhdl|val1000_ouhdl) echo $OU/e0_val.npz;; test_ouhdl) echo $OU/e0_test.npz;; gold_ouhdl) echo $OU/e0_gold.npz;; val250_iam) echo $D/m5_val_iam.npz;; test_iam500) echo $D/m5_test_iam500.npz;; german_unipen_de_test) echo $STAGE/data/d2_v2/unipen_de/e0_test.npz;; esac; }
log "k1_start sets=[$SETS] ltr=[$LTR_SETS] seeds=[$SEEDS] gpu=[$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ';')]"
$PYG $CODE/frame_sample.py --label LUO_ms6 --run-dir exp/M5/runs/luo --milestone 6 --stage $STAGE --sets $SETS $LTR_SETS \
   --frames identity mirror_x vt --ltr-frames mirror_x identity --seeds $SEEDS --out-dir $OUT --batch-size 256 ${SMK[@]+"${SMK[@]}"} 2>&1 | grep -av "sampling loop" | tail -4
for s in $SETS $LTR_SETS; do
  for P in $OUT/preds/${s}__LUO_ms6_*__seed*_preds${SFX}.npz; do
    [ -e "$P" ] || continue; b=$(basename $P .npz); R=$OUT/preds/${b%_preds*}_rows${SFX}.csv; [ -s $R ] && continue
    lab=${b#${s}__}; lab=${lab%%__*}
    $PYC $CODE/e0_score.py e0 --npz $(npz_of $s) --preds $P --method $lab --input rendered --out ${R%.csv}.part.csv --procs $PROCS 2>&1 | tail -1 && mv ${R%.csv}.part.csv $R
  done
  $PYC $CODE/frame_k1_summary.py --rows "$OUT/preds/${s}__LUO_ms6_*__seed*_rows${SFX}.csv" --set $s --out $OUT/k1_summary_${s}${SFX}.json 2>&1 | tail -2
done
$PYC $CODE/e0_score.py summary --rows $OUT/preds/*_rows${SFX}.csv --out $OUT/summary_k1${SFX}.csv | tail -1
$PYC - <<PY
import json, hashlib, glob, os, platform, time
out = "$OUT"
def sha(p):
    h = hashlib.sha256(); f = open(p, 'rb')
    for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()
ck = "$STAGE/exp/M5/runs/luo/checkpoints/checkpoint_000006.pt"
json.dump(dict(check="K1 frame kill check", checkpoint=ck, checkpoint_sha256=sha(ck), sets="$SETS".split(), ltr_sets="$LTR_SETS".split(), seeds="$SEEDS".split(),
               frames=["identity", "mirror_x", "vt"], ltr_frames=["mirror_x", "identity"], inverse_applied_before_scoring=True,
               preds={os.path.basename(f): sha(f) for f in sorted(glob.glob(out + "/preds/*_preds*.npz"))},
               summaries=sorted(os.path.basename(f) for f in glob.glob(out + "/k1_summary_*.json")), code_dir="$CODE",
               code_sha256sums=open("$CODE/SHA256SUMS.txt").read().splitlines(), host=platform.node(), slurm_job=os.environ.get("SLURM_JOB_ID"),
               gpu=os.popen("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null").read().strip() or None, smoke_units="$SMOKE_UNITS" or None,
               protocol="uhwr_eval 0.3.1 native (order metrics) + standardised, e0_score.py; writer-cluster bootstrap B=2000 seed 0",
               at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), open(out + "/PROVENANCE_K1${SFX}.json", "w"), indent=1)
print("PROVENANCE_K1 written")
PY
log "k1_end"
