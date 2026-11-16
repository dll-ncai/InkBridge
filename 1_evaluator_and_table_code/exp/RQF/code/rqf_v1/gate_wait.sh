#!/bin/bash
# RQF gate: completes (rc 0) once <RUN>/eval/best_milestone.txt exists, whichever lane (CLUSTER-A job or a relayed CLUSTER-B copy) produced it. Usage: gate_wait.sh RUN_NAME [max_hours]
source $U/exp/RQF/code/rqf_v1/rqf_env.sh
RN=$1; MAXH=${2:-140}; T0=$(date +%s)
while [ ! -f $QR/$RN/eval/best_milestone.txt ]; do [ $(( ($(date +%s) - T0) / 3600 )) -ge $MAXH ] && { echo "$(utc) GATE_TIMEOUT $RN"; exit 9; }; sleep 300; done
echo "$(utc) GATE_OPEN $RN best=$(cat $QR/$RN/eval/best_milestone.txt)"
