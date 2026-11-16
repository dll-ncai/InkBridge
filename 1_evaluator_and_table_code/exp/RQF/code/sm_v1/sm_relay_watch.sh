#!/bin/bash
# RELAY-ONLY copy of sm_watch2.sh (repair <agent>, 12 Nov 2026): hold/release block removed (releases handled by rqf_seed_relwatch). Original header: RQF A2.3 VPS race+relay watcher v2 (replaces 24965132, ended at its 3-day limit). Poll 600 s; stops 2026-11-14T17:30Z (CLUSTER-A->VPS key approved for RQF pulls until 14 Nov 18:00Z)
# or when admin/SM_WATCH_STOP exists. ONEPASS=1 = a single pass (login-node catch-up).
# Per TAG in admin/sm_vps_tags.tsv: winner = copy whose start is earlier (VPS STARTED.json started_utc vs Slurm Start of sm_<TAG>).
#  - VPS started and CLUSTER-A copy PENDING -> scancel own CLUSTER-A loser; CLUSTER-A RUNNING and VPS cmd queued -> withdraw own VPS cmd.
#  - VPS SAMPLING_DONE, VPS is the winner, not relayed -> tar pull, sha256 manifest, (smr_luo: sm_fixside.py) link into exp/RQF/preds/<TAG>, sbatch scoring, then sbatch sm_read (afterany).
#  - VPS copy is the loser (CLUSTER-A copy started earlier) -> never relayed (logged once as RACE_LOST_VPS).
set -u; source $U/exp/RQF/code/sm_v1/sm_env.sh
SSHV="ssh -n -o BatchMode=yes -o ConnectTimeout=20 -i $HOME/.ssh/uhwr_staging_vps_ed25519 Administrator@<ip>"
T=$Q/admin/sm_vps_tags.tsv; VR=$VPS_ROOT/uhwr/runs/RQF/SM; END=$(date -d 2026-11-14T17:30:00Z +%s); EXC="dgx[001-004],gpu[003-004,013-014,017-023,119-148]"
L() { rqf_log "[repair <agent> sm_relay ${SLURM_JOB_ID:-login}] $*"; }
SB() { env -i PATH=/usr/bin:/bin:/usr/local/bin HOME=$HOME USER=$USER sbatch --parsable "$@"; }
while :; do
  NEWS=""
  while IFS=$'\t' read -r TAG CMD KIND; do
    [ -z "$TAG" ] && continue; M=$Q/runs/${TAG}_vps; mkdir -p $M
    [ -f $M/RELAYED ] && [ -f $M/SCORE_SUBMITTED ] && continue; [ -f $M/RACE_LOST_VPS ] && continue
    st=$($SSHV "powershell -NoProfile -Command \"if(Test-Path $VR/$TAG/SAMPLING_DONE){'DONE'}elseif(Test-Path $VR/$TAG/STARTED.json){'STARTED'}elseif(Test-Path $VPS_ROOT/uhwr/runner/queue/$CMD.cmd){'QUEUED'}else{'NONE'}\"" 2>/dev/null | tr -d '\r')
    [ -z "$st" ] && { L "SSH_FAIL status $TAG"; continue; }
    rj=$(sacct -u $USER -X -n -P -S 2026-11-07 --name=sm_$TAG -o JobID,State,Start | grep -v CANCELLED | tail -1)
    rid=${rj%%|*}; rst=$(echo "$rj" | cut -d'|' -f2); rstart=$(echo "$rj" | cut -d'|' -f3)
    if [ "$st" = STARTED ] || [ "$st" = DONE ]; then
      vstart=$($SSHV "powershell -NoProfile -Command \"(Get-Content $VR/$TAG/STARTED.json | ConvertFrom-Json).started_utc\"" 2>/dev/null | tr -d '\r')
      if [ "$rst" = PENDING ]; then scancel $rid && L "CANCEL own CLUSTER-A loser sm_$TAG job=$rid (VPS $CMD started $vstart)"
      elif [ -n "$rstart" ] && [ "$rstart" != Unknown ] && [ "$rstart" != None ] && [ $(date -d "$rstart" +%s) -lt $(date -d "$vstart" +%s) ]; then
        touch $M/RACE_LOST_VPS; L "RACE_LOST_VPS $TAG: CLUSTER-A sm_$TAG job=$rid started $rstart before VPS $vstart; VPS copy not relayed"; continue; fi
    elif [ "$st" = QUEUED ] && [ "$rst" = RUNNING ]; then
      $SSHV "powershell -NoProfile -Command \"Rename-Item $VPS_ROOT/uhwr/runner/queue/$CMD.cmd $CMD.cmd.withdrawn\"" && L "WITHDRAW own VPS loser $CMD (CLUSTER-A sm_$TAG job=$rid started first)"
    fi
    if [ "$st" = DONE ]; then
      SUB=ft; [ "$KIND" = smr_luo ] && SUB=preds
      if [ ! -f $M/RELAYED ]; then
        if $SSHV "tar -cf - -C $VR/$TAG $SUB eval STARTED.json SAMPLING_DONE job_history.txt" | tar -xf - -C $M; then
          (cd $M && find . -type f ! -name 'SHA256SUMS_relay.txt' | sort | xargs sha256sum > SHA256SUMS_relay.txt); touch $M/RELAYED
          L "RELAY VPS $TAG -> $M ($(wc -l < $M/SHA256SUMS_relay.txt) files, sha256 manifest)"
        else L "RELAY_FAILED $TAG"; continue; fi
      fi
      mkdir -p $QP/$TAG
      if [ "$KIND" = smr_luo ]; then python3 $SMC/sm_fixside.py $M/preds $M/preds_clusterA >/dev/null && ln -sfn $M/preds_clusterA $QP/$TAG/preds; SK="luoft ."
      else ln -sfn $M/ft $QP/$TAG/ft; SK="ft ft"; fi
      sid=$(SB -p cpuidle -J smscore_${TAG}_vps $SMC/sm_cpu.sbatch score $TAG $SK) && touch $M/SCORE_SUBMITTED && NEWS="$NEWS:$sid" && L "SUBMIT score ${TAG}_vps job=$sid"
    fi
  done < $T
  if [ -n "$NEWS" ]; then rid2=$(SB -p cpuidle --dependency=afterany$NEWS -J sm_read2 $SMC/sm_cpu.sbatch read) && L "SUBMIT sm_read refresh (pooled seeds) job=$rid2 afterany$NEWS"
    rid3=$(SB -p cpuidle --dependency=afterany$NEWS --export=ALL,SM_SEEDS=20261034,SM_OUT_SUFFIX=_seed1 -J sm_read_s1 $SMC/sm_cpu.sbatch read) && L "SUBMIT sm_read refresh (seed 1 only, tables/RQF_SM_seed1_*) job=$rid3 afterany$NEWS"; fi
  [ "${ONEPASS:-0}" = 1 ] && break
  [ $(date +%s) -ge $END ] || [ -e $Q/admin/SM_WATCH_STOP ] && break
  sleep 600
done
L "sm_relay watch end"
