#!/bin/bash
# RQF R1 scoring array task (uhwr_eval 0.3.1, lane_score.py = D3/Route A per-unit scorer). TASKS line = "<preds glob>|<JOB>". Waits (polls, pulls VPS-lane preds) until 3 seed files exist.
set -uo pipefail; exec 2>&1
R=$U; O=$R/exp/RQF/R1; Q=$R/exp/RQF/code/rqf1_v1; TASKS=${TASKS:?}
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $TASKS); PAT=${LINE%%|*}; JOB=${LINE##*|}
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
SSHV="ssh -o BatchMode=yes -o ConnectTimeout=20 -i $HOME/.ssh/uhwr_staging_vps_ed25519 Administrator@<ip>"
t0=$(date +%s); mkdir -p $O/score_out
while :; do
  n=$(ls $O/preds/$PAT 2>/dev/null | wc -l); nj=$(ls $O/preds/${PAT%.npz}.json 2>/dev/null | wc -l)
  [ "$n" -ge 3 ] && [ "$nj" -ge 3 ] && break          # 3 seeds: npz + json sidecars
  [ $(( $(date +%s) - t0 )) -gt 108000 ] && { echo "$(utc) timeout waiting for $PAT ($n files)"; exit 5; }
  sleep 300
done
sleep 20   # sidecar/npz writes are atomic (os.replace); small grace
UE=$R/shared/uhwr_eval-0.3.1; E0C=$R/e0/code_4cd40f57; IMG=$R/images/torch27.sif; PY=$R/venvs/uhwr_eval02/bin/python; LS=$R/exp/ORDER/code/lane_score.py
export TMPDIR=/tmp/R1S_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}; mkdir -p $TMPDIR/numba $TMPDIR/mpl; trap 'rm -rf $TMPDIR' EXIT
ENVV="PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=$TMPDIR/numba MPLCONFIGDIR=$TMPDIR/mpl CUDA_VISIBLE_DEVICES= PYTHONPATH=$UE"
AX="apptainer exec -B $SCRATCH_ROOT -B /tmp $IMG env $ENVV $PY"
echo "$(utc) start host=$(hostname) slurm=${SLURM_ARRAY_JOB_ID:-x}_${SLURM_ARRAY_TASK_ID} pat=$PAT lane_score_sha256=$(sha256sum $LS | cut -c1-64)" | tee -a $O/score_out/job_history.txt
$AX $LS --preds-dir $O/preds --out $O/score_out --e0code $E0C --procs ${SLURM_CPUS_PER_TASK:-16} --only "$PAT"; rc=$?
echo "$(utc) end pat=$PAT rc=$rc" | tee -a $O/score_out/job_history.txt; exit $rc
