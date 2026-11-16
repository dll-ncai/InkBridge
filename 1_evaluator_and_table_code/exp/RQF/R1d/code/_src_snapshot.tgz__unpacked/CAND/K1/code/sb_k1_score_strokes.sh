#!/bin/bash
#SBATCH --job-name=cand_k1_strokes
#SBATCH --partition=epyc-256
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=$U/exp/CAND/K1/logs/slurm-%x-%A_%a.out
# CAND K1 per-stroke class scorer array (k1_strokeclass.py; uhwr_eval 0.3.1, same env as lane_score_array.sh). Env: PREDS OUT TASKS.
set -o pipefail
U=$U; C=$U/exp/CAND/K1/code
IMG=$U/images/torch27.sif; PY=$U/venvs/uhwr_eval02/bin/python; UE=$U/shared/uhwr_eval-0.3.1; E0C=$U/e0/code_4cd40f57
PREDS=${PREDS:?}; OUT=${OUT:?}; TASKS=${TASKS:?}
ONLY=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $TASKS); T=$SLURM_ARRAY_TASK_ID
mkdir -p $OUT; utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
export TMPDIR=/tmp/KS_${SLURM_JOB_ID}; mkdir -p $TMPDIR/numba $TMPDIR/mpl; trap 'rm -rf $TMPDIR' EXIT
ENVV="PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=$TMPDIR/numba MPLCONFIGDIR=$TMPDIR/mpl CUDA_VISIBLE_DEVICES= PYTHONPATH=$UE"
AX="apptainer exec -B $SCRATCH_ROOT -B /tmp $IMG env $ENVV $PY"
echo "$(utc) start host=$(hostname) slurm=$SLURM_ARRAY_JOB_ID task=$T only=$ONLY code_sha256=$(sha256sum $C/k1_strokeclass.py | cut -c1-64)" | tee -a $OUT/job_history_strokes.txt
timeout 3h $AX $C/k1_strokeclass.py --preds-dir $PREDS --out $OUT --e0code $E0C --procs ${SLURM_CPUS_PER_TASK:-16} --only "$ONLY"; rc=$?
echo "$(utc) end task=$T only=$ONLY rc=$rc" | tee -a $OUT/job_history_strokes.txt
exit $rc
