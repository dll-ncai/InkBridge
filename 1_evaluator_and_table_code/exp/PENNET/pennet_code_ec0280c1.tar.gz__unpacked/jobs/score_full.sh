#!/bin/bash
# PENNET scoring (CPU, afterok on the training job): val1000 rows for every milestone -> selection (E0 rule) ->
# test/gold x rendered/dataset rows for the selected milestone -> writer-cluster + line-cluster summaries -> provenance.
# e0_score.py (unchanged copy) under uhwr_eval 0.3.1; venv copied to /dev/shm with node-local NUMBA_CACHE_DIR
# (libllvmlite workaround of exp/M1b/code_v4_scorefix). Idempotent.
set -u -o pipefail
U=$U; SIF=$U/images/torch27.sif; D=$U/data/ouhdl_e0_v3; UE=$U/shared/uhwr_eval-0.3.1
EV=$E/eval; mkdir -p $EV/rows $E/results
LIB_SHA=977e27481b9c4c94dfc5bfe62541b02533524ab243c66d9bbc0f67098ea37160
echo "$(date -Is) job=${SLURM_JOB_ID:-none} host=$(hostname) phase=score_start" >> $E/run/job_history.txt
LOCAL=/dev/shm/pennet_${SLURM_JOB_ID:-x}; trap 'rm -rf $LOCAL' EXIT
mkdir -p $LOCAL/numba_cache
ok=0
for t in 1 2 3; do
  rm -rf $LOCAL/venv_e0; cp -a $U/venvs/e0 $LOCAL/venv_e0
  h=$(sha256sum $LOCAL/venv_e0/lib/python3.11/site-packages/llvmlite/binding/libllvmlite.so | cut -d' ' -f1)
  [ "$h" = "$LIB_SHA" ] && { ok=1; break; }; sleep 20
done
[ $ok -eq 1 ] || { echo "venv copy failed"; exit 3; }
RUN="apptainer exec -B $SCRATCH_ROOT -B $LOCAL $SIF env PYTHONPATH=$C:$UE:$E/lib OMP_NUM_THREADS=1 NUMBA_CACHE_DIR=$LOCAL/numba_cache $LOCAL/venv_e0/bin/python"
$RUN -c "import uhwr_eval,sys; print('uhwr_eval', uhwr_eval.__version__, uhwr_eval.__file__); sys.exit(0 if uhwr_eval.__version__=='0.3.1' and uhwr_eval.__file__.startswith('$UE/') else 2)" | tee $EV/uhwr_eval_version.txt || exit 2
score() {  # npz preds method input out
  local tmp=$LOCAL/$(basename $5)
  [ -f $5 ] && return 0
  for a in 1 2 3; do
    $RUN $C/e0_score.py e0 --npz $1 --preds $2 --method $3 --input $4 --out $tmp --procs 16 > $LOCAL/score.log 2>&1
    rc=$?; n=0; [ -f $tmp ] && n=$(wc -l < $tmp)
    nu=$($RUN -c "import numpy as np,sys;print(len(np.load(sys.argv[1])['unit_ids']))" $2 | tail -1)
    if [ $rc -eq 0 ] && [ "$n" -eq $((2*nu+1)) ]; then cp $tmp $5.part && mv $5.part $5; echo "$5 $((n-1)) rows"; return 0; fi
    tail -5 $LOCAL/score.log; sleep 30
  done
  return 1
}
for P in $EV/preds/val1000_rendered_ms*_preds.npz; do
  k=$(basename $P _preds.npz | sed 's/.*_ms//')
  score $D/e0_val.npz $P pennet_ms${k} rendered $EV/rows/val1000_ms${k}_rows.csv || exit 4
done
BEST=$($RUN $C/pennet_select.py --rows $EV/rows/val1000_ms*_rows.csv --out $EV/selection.json | tail -1)
echo "$(date -Is) job=${SLURM_JOB_ID:-none} selected milestone $BEST" >> $E/run/job_history.txt
for S in test gold; do for COND in rendered dataset; do
  score $D/e0_${S}.npz $EV/preds/${S}_${COND}_ms${BEST}_preds.npz pennet_v3_ms${BEST}_greedy $COND $EV/rows/${S}_${COND}_ms${BEST}_rows.csv || exit 5
done; done
ROWS="$EV/rows/test_rendered_ms${BEST}_rows.csv $EV/rows/test_dataset_ms${BEST}_rows.csv $EV/rows/gold_rendered_ms${BEST}_rows.csv $EV/rows/gold_dataset_ms${BEST}_rows.csv"
$RUN $C/pennet_summary_wc.py --rows $ROWS --out-prefix $EV/summary_PENNET_ms${BEST}_writercluster || exit 6
$RUN $C/e0_score.py summary --rows $ROWS --out $EV/summary_PENNET_ms${BEST}_linecluster.csv || exit 6
$RUN -c "
import numpy as np, json, glob, os
out = {}
for f in sorted(glob.glob('$EV/preds/*_ms${BEST}_preds.npz')) + sorted(glob.glob('$EV/preds/val1000_*_preds.npz')):
    out[os.path.basename(f)] = json.loads(str(np.load(f)['meta']))
json.dump(out, open('$EV/decode_meta.json', 'w'), indent=1)"
python3 $C/pennet_provenance.py --exp $E --code $C --mode full || exit 7
cp $EV/selection.json $EV/summary_PENNET_* $EV/decode_meta.json $EV/uhwr_eval_version.txt $E/PROVENANCE_PENNET.json $E/results/ 2>/dev/null
cp $EV/rows/*_ms${BEST}_rows.csv $E/results/ 2>/dev/null
echo "$(date -Is) job=${SLURM_JOB_ID:-none} phase=score_done" >> $E/run/job_history.txt
ls -lh $E/results
