#!/bin/bash
# RQF R1 tables + trace rescoring + provenance (CPU). Waits for TRACE rescoring itself (step inside).
set -uo pipefail; exec 2>&1
R=$U; O=$R/exp/RQF/R1; Q=$R/exp/RQF/code/rqf1_v1; IMG=$R/images/torch27.sif; PY=$R/venvs/uhwr_eval02/bin/python
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
export TMPDIR=/tmp/R1T_${SLURM_JOB_ID}; mkdir -p $TMPDIR; trap 'rm -rf $TMPDIR' EXIT
ENVV="PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 NUMBA_CACHE_DIR=$TMPDIR MPLCONFIGDIR=$TMPDIR"
AX() { apptainer exec -B $SCRATCH_ROOT -B /tmp $IMG env $ENVV "$@"; }
TR=$O/trace_score
if ! ls $TR/rows/*__rows031.csv >/dev/null 2>&1; then
  echo "$(utc) TRACE triples (uhwr_eval 0.2.0 GT env)"
  AX PYTHONPATH=$Q:$R/exp/L_code/code_6938866b $PY $Q/trace_rescore.py triples --out $TR 2>&1 | tail -3
  echo "$(utc) TRACE score (uhwr_eval 0.3.1)"
  AX PYTHONPATH=$R/shared/uhwr_eval-0.3.1:$Q $PY $Q/trace_rescore.py score --out $TR --workers ${SLURM_CPUS_PER_TASK:-8} 2>&1 | tail -6
fi
echo "$(utc) tables"
AX PYTHONPATH=$R/shared/uhwr_eval-0.3.1 $PY $Q/rqf1_tables.py --rows $O/score_out $TR/rows $R/exp/CAND/K1/score/rows --out $O/tables 2>&1 | tail -4
echo "$(utc) provenance"
AX PYTHONPATH=$R/shared/uhwr_eval-0.3.1 $PY $Q/r1_prov.py --o $O 2>&1 | tail -3
echo "$(utc) done"
