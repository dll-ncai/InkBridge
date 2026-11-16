#!/bin/bash
#SBATCH --job-name=lane_score
#SBATCH --partition=epyc-256
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=$U/exp/D3_ROUTE/routeA/logs/slurm-%x-%A_%a.out
# Generic per-unit scoring array (uhwr_eval 0.3.1, Stage A recipe). Env: PREDS=<preds dir> OUT=<score_out dir> TASKS=<tasks.txt>.
# Submit: PREDS=... OUT=... TASKS=... sbatch --array=1-N --export=ALL,PREDS,OUT,TASKS lane_score_array.sh
set -o pipefail
U=$U; C=$U/exp/D3_ROUTE/routeA/code
IMG=$U/images/torch27.sif; PY=$U/venvs/uhwr_eval02/bin/python; UE=$U/shared/uhwr_eval-0.3.1; E0C=$U/e0/code_4cd40f57
PREDS=${PREDS:?}; OUT=${OUT:?}; TASKS=${TASKS:?}
ONLY=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $TASKS); T=$SLURM_ARRAY_TASK_ID
mkdir -p $OUT; utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
export TMPDIR=/tmp/LS_${SLURM_JOB_ID}; mkdir -p $TMPDIR/numba $TMPDIR/mpl; trap 'rm -rf $TMPDIR' EXIT
ENVV="PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=$TMPDIR/numba MPLCONFIGDIR=$TMPDIR/mpl CUDA_VISIBLE_DEVICES= PYTHONPATH=$UE"
AX="apptainer exec -B $SCRATCH_ROOT -B /tmp $IMG env $ENVV $PY"
echo "$(utc) start host=$(hostname) slurm=$SLURM_ARRAY_JOB_ID task=$T only=$ONLY preds=$PREDS code_sha256=$(sha256sum $C/lane_score.py | cut -c1-64)" | tee -a $OUT/job_history.txt
$AX -c "import uhwr_eval;print('uhwr_eval',uhwr_eval.__version__,uhwr_eval.__file__)" | tee $OUT/env_$T.txt
timeout 3h $AX $C/lane_score.py --preds-dir $PREDS --out $OUT --e0code $E0C --procs ${SLURM_CPUS_PER_TASK:-16} --only "$ONLY"; rc=$?
echo "$(utc) end task=$T only=$ONLY rc=$rc" | tee -a $OUT/job_history.txt
exit $rc
