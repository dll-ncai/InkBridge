#!/bin/bash
#SBATCH -p cpuidle
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH -J checks_v12
set -uo pipefail; U=$U; D=$U/exp/RV12/checks_v12; mkdir -p $D/code $D/out_r2 $D/smoke_r2
cp ./checks_v12.py $D/code/checks_v12.py
export TMPDIR=/tmp/chk_${SLURM_JOB_ID:-x}; mkdir -p $TMPDIR
PY="apptainer exec -B /scratch -B /tmp $U/images/torch27.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_CACHE_DIR=$TMPDIR MPLCONFIGDIR=$TMPDIR CUDA_VISIBLE_DEVICES= PYTHONPATH=$U/shared/uhwr_eval-0.3.1 $U/venvs/uhwr_eval02/bin/python"
echo "job ${SLURM_JOB_ID:-none} host $(hostname) $(date -u +%FT%TZ)"
$PY $D/code/checks_v12.py $D/smoke_r2 --smoke --procs 8 > $D/smoke_r2/log.txt 2>&1; src=$?; echo smoke rc $src; tail -4 $D/smoke_r2/log.txt | cut -c1-1500; if [ $src -ne 0 ]; then mkdir -p out; cp $D/smoke_r2/log.txt out/smoke_log.txt; cp $D/smoke_r2/checks_v12_facts.json out/smoke_facts.json 2>/dev/null; exit 3; fi
$PY $D/code/checks_v12.py $D/out_r2 --procs 14 > $D/out_r2/log.txt 2>&1; rc=$?; echo full rc $rc; tail -6 $D/out_r2/log.txt
mkdir -p out; cp $D/out_r2/checks_v12_* $D/out_r2/PROVENANCE_* $D/out_r2/log.txt out/; cp $D/smoke_r2/log.txt out/smoke_log.txt
rm -rf $TMPDIR; exit $rc
