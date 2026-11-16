#!/bin/bash
# RQF R2 final tables (after all scoring jobs ended; afterany so partial inputs are listed under "Missing inputs", never filled)
set -uo pipefail; exec 2>&1
R=$U; mkdir -p $R/exp/RQF/tables
apptainer exec -B $SCRATCH_ROOT $R/images/torch27.sif $R/venvs/cpu/bin/python $R/exp/RQF/code/rqf_v1/build_tables_rqf.py $R/exp/RQF/tables/RQF_R2
cd $R/exp/RQF/tables && sha256sum RQF_R2_* > RQF_R2_SHA256.txt; echo "TABLES_DONE $(date -u +%FT%TZ)"
