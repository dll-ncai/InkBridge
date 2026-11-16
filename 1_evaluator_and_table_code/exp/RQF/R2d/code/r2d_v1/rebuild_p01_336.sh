#!/bin/bash
# R2d v1.2 (<author> 8 Nov 'use comparable counts'): rebuild tamil_p01 with K = 336 units (Urdu p01 count), writer-stratified, seed 20261107.
# The K=492 files are kept as *_k492* with PROVENANCE marked SUPERSEDED. Run via clusterA_stage-like wrapper on a CPU node.
set -uo pipefail; exec 2>&1
source $U/exp/RQF/R2d/code/r2d_v1/r2d_env.sh; check_code; CPUS=${SLURM_CPUS_PER_TASK:-8}
cd $DD
if [ ! -f tamil_p01_k492.npz ]; then
  for f in tamil_p01.npz tamil_p01_render224.npy tamil_p01_render224.npy.meta.json tamil_p01_render224__source.npy tamil_p01_render224__unit_ids.npy; do [ -f $f ] && mv $f ${f/tamil_p01/tamil_p01_k492}; done
  $PY3 - <<PYEOF
import json; p="PROVENANCE_tamil_p01.json"; d=json.load(open(p)); d["SUPERSEDED"]={"utc":"$(utc)","by":"tamil_p01.npz K=336 (R2d v1.2, <author> 8 Nov 'use comparable counts')","renamed_to":"tamil_p01_k492.npz"}
json.dump(d,open("PROVENANCE_tamil_p01_k492_SUPERSEDED.json","w"),indent=1)
PYEOF
  rm -f PROVENANCE_tamil_p01.json
fi
pyc $P_M5 $RC/build_r2d_data.py tamil_p01 --k 336 --out $DD --procs $CPUS || exit 2
pyc $P_BB $RC/r2d_render.py tamilp01 --npz $DD/tamil_p01.npz --out $DD/tamil_p01_render224.npy --procs $CPUS || exit 2
echo "$(utc) rebuild_p01_336 done"
