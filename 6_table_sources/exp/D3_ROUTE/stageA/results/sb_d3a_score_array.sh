#!/bin/bash
#SBATCH --job-name=d3a_score
#SBATCH --array=1-26
#SBATCH --partition=epyc-256
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=$U/exp/D3_ROUTE/stageA/logs/slurm-%x-%A_%a.out
# D3 Stage A per-unit scoring (uhwr_eval 0.3.1). Environment = the table-builder recipe (shared/table_builder/code_7d405fd6/
# run_table_builder.sh): torch27.sif + venvs/uhwr_eval02, PYTHONPATH=shared/uhwr_eval-0.3.1, e0 adapter code_4cd40f57 (no node-local venv copy).
set -o pipefail
U=$U; S=$U/exp/D3_ROUTE/stageA; C=$S/code/score; O=$S/score_out
IMG=$U/images/torch27.sif; PY=$U/venvs/uhwr_eval02/bin/python; UE=$U/shared/uhwr_eval-0.3.1; E0C=$U/e0/code_4cd40f57
ONLY=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $C/tasks.txt); T=$SLURM_ARRAY_TASK_ID
mkdir -p $O; utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
export TMPDIR=/tmp/D3A_${SLURM_JOB_ID}; mkdir -p $TMPDIR/numba $TMPDIR/mpl; trap 'rm -rf $TMPDIR' EXIT
ENVV="PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=$TMPDIR/numba MPLCONFIGDIR=$TMPDIR/mpl CUDA_VISIBLE_DEVICES= PYTHONPATH=$UE"
AX="apptainer exec -B $SCRATCH_ROOT -B /tmp $IMG env $ENVV $PY"
echo "$(utc) start host=$(hostname) slurm=$SLURM_ARRAY_JOB_ID task=$T only=$ONLY code_sha256=$(sha256sum $C/d3a_score.py | cut -c1-64)" | tee -a $O/job_history.txt
$AX -c "import uhwr_eval,numpy,pandas;print('uhwr_eval',uhwr_eval.__version__,uhwr_eval.__file__,'numpy',numpy.__version__,'pandas',pandas.__version__)" | tee $O/env_$T.txt
[ $T = 1 ] && sha256sum $IMG > $O/container_sha256.txt
$AX $C/d3a_score.py --preds-dir $S/preds --out $O --e0code $E0C --procs ${SLURM_CPUS_PER_TASK:-16} --only "$ONLY"; rc=$?
echo "$(utc) end task=$T only=$ONLY rc=$rc" | tee -a $O/job_history.txt
exit $rc
