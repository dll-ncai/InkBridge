#!/bin/bash
# run one R1d python step on CPU in torch27.sif + uhwr_eval02: py.sh SCRIPT ARGS...
set -uo pipefail; exec 2>&1
source $U/exp/RQF/R1d/code/r1d_v1/r1d_env.sh
(cd $RC && sha256sum -c --quiet SHA256SUMS.txt) || { echo "code differs from SHA256SUMS.txt"; exit 3; }
S=$1; shift; apptainer exec -B $SCRATCH_ROOT -B /tmp $IMG env $ENVB CUDA_VISIBLE_DEVICES= PYTHONPATH=$UE:$FCODE OMP_NUM_THREADS=4 $VEV/bin/python $RC/$S "$@"
