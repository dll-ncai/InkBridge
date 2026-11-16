#!/bin/bash
# RQF R2 INTERIM tables (no hard dependency): build_tables_rqf.py over the scoring rows that exist; rows carry evidence_label (CONFIRMATORY CLUSTER-A / EXPLORATORY VPS). Versioned output RQF_R2_interim_<UTC>_*; RQF_R2_interim_latest_* copies.
set -uo pipefail; exec 2>&1
R=$U; T=$R/exp/RQF/tables; mkdir -p $T; S=$(date -u +%Y%m%dT%H%MZ)
apptainer exec -B $SCRATCH_ROOT $R/images/torch27.sif $R/venvs/cpu/bin/python $R/exp/RQF/code/rqf_v1/build_tables_rqf.py $T/RQF_R2_interim_$S
cd $T && for f in RQF_R2_interim_${S}_*; do cp -f $f ${f/$S/latest}; done; sha256sum RQF_R2_interim_${S}_* > RQF_R2_interim_${S}_SHA256.txt; echo "TABLES_DONE $S $(date -u +%FT%TZ)"
