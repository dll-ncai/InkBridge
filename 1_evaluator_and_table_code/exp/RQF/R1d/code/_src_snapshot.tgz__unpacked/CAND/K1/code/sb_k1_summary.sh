#!/bin/bash
#SBATCH --job-name=cand_k1_summary
#SBATCH --partition=epyc-256
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=$U/exp/CAND/K1/logs/slurm-%x-%j.out
set -o pipefail
U=$U; K=$U/exp/CAND/K1; C=$K/code
IMG=$U/images/torch27.sif; PY=$U/venvs/uhwr_eval02/bin/python
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "$(utc) start host=$(hostname) slurm=$SLURM_JOB_ID code_sha256=$(sha256sum $C/k1_summary_cand.py | cut -c1-64)" | tee -a $K/summary/job_history.txt
apptainer exec -B $SCRATCH_ROOT $IMG env PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 $PY $C/k1_summary_cand.py --rows $K/score/rows --strokes $K/score/strokes --out $K/summary --boot 2000; rc=$?
echo "$(utc) end rc=$rc" | tee -a $K/summary/job_history.txt
exit $rc
