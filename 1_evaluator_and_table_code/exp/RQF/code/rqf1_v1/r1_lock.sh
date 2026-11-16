# RQF R1 race lock on the VPS hub (atomic mkdir on Windows; owner file lane:jobid). Echoes WIN | LOSE <owner> | ERR <detail>. Requeue-safe (owner == self -> WIN).
r1_lock() {  # r1_lock JOB LANE
  local job=$1 lane=$2 id=${SLURM_JOB_ID:-x}; local K=$HOME/.ssh/uhwr_staging_vps_ed25519
  local S="ssh -o BatchMode=yes -o ConnectTimeout=20 -i $K Administrator@<ip>"
  local d="$VPS_ROOT\\uhwr\\race\\RQF1_$job"
  local o own
  o=$(timeout 60 $S "cmd /c \"mkdir $d 2>nul && echo $lane:$id>$d\\owner.txt && echo WIN\"" 2>&1 | tr -d '\r' | tail -1)
  if [ "$o" = WIN ]; then echo WIN; return; fi
  own=$(timeout 60 $S "cmd /c type $d\\owner.txt" 2>&1 | tr -d '\r ' | tail -1)
  if [ "$own" = "$lane:$id" ]; then echo WIN; elif echo "$own" | grep -Eq '^(clusterA|clusterB):[0-9x]+$'; then echo "LOSE $own"; else echo "ERR [$o] [$own]"; fi
}
