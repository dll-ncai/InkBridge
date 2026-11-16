#!/bin/bash
#SBATCH --job-name=routeA_tables
#SBATCH --partition=epyc-256
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=$U/exp/D3_ROUTE/routeA/logs/slurm-%x-%j.out
set -o pipefail
U=$U; R=$U/exp/D3_ROUTE/routeA; C=$R/code
IMG=$U/images/torch27.sif; PY=$U/venvs/uhwr_eval02/bin/python
export TMPDIR=/tmp/RAT_${SLURM_JOB_ID}; mkdir -p $TMPDIR/mpl; trap 'rm -rf $TMPDIR' EXIT
CTRL=${CTRL_ROWS:-$U/exp/D2_MOE/controls/score_out/rows}
apptainer exec -B $SCRATCH_ROOT -B /tmp $IMG env PYTHONNOUSERSITE=1 MPLCONFIGDIR=$TMPDIR/mpl OMP_NUM_THREADS=4 $PY $C/routeA_tables.py \
  --rows-routea $R/score_out/rows --rows-stagea $U/exp/D3_ROUTE/stageA/score_out/rows --rows-rb4 $U/exp/RB4_baselines/german/rows \
  --rows-ctrl $CTRL --out $R/results --boot 2000
rc=$?; echo "$(date -u +%FT%TZ) routeA_tables rc=$rc job=$SLURM_JOB_ID" >> $R/results/job_history.txt; exit $rc
