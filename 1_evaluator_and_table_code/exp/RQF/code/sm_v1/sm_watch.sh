#!/bin/bash
# RQF A2.3 race + relay watcher for the VPS copies (CLUSTER-A CPU job; CLUSTER-A->VPS key approved by <author> for RQF until the 10 Nov 18:00Z freeze).
# Every 10 min, per TAG in admin/sm_vps_tags.tsv (TAG<TAB>VPS_CMD<TAB>KIND):
#  race : VPS STARTED.json exists and the CLUSTER-A GPU copy (sm_<TAG>) is PENDING -> scancel own CLUSTER-A loser (its afterok score job dies with it); log.
#         CLUSTER-A copy RUNNING and the VPS cmd still queued -> withdraw the VPS cmd (rename .withdrawn); log.
#  relay: VPS SAMPLING_DONE and not yet relayed -> tar pull ft|preds + eval + run json into exp/RQF/runs/<TAG>_vps, link preds into exp/RQF/preds/<TAG>, submit scoring (sm_cpu.sbatch); log.
# Stops at 2026-11-10T18:30Z or when admin/SM_WATCH_STOP exists.
set -u; source $U/exp/RQF/code/sm_v1/sm_env.sh
SSHV="ssh -o BatchMode=yes -o ConnectTimeout=20 -i $HOME/.ssh/uhwr_staging_vps_ed25519 Administrator@<ip>"
T=$Q/admin/sm_vps_tags.tsv; VR=$VPS_ROOT/uhwr/runs/RQF/SM; END=$(date -d 2026-11-10T18:30:00Z +%s)
L() { rqf_log "[rqf-sm watch job ${SLURM_JOB_ID:-x}] $*"; }
while [ $(date +%s) -lt $END ] && [ ! -e $Q/admin/SM_WATCH_STOP ]; do
  while IFS=$'\t' read -r TAG CMD KIND; do
    [ -z "$TAG" ] && continue; M=$Q/runs/${TAG}_vps; mkdir -p $M
    st=$($SSHV "powershell -NoProfile -Command \"if(Test-Path $VR/$TAG/SAMPLING_DONE){'DONE'}elseif(Test-Path $VR/$TAG/STARTED.json){'STARTED'}elseif(Test-Path $VPS_ROOT/uhwr/runner/queue/$CMD.cmd){'QUEUED'}else{'NONE'}\"" 2>/dev/null | tr -d '\r')
    rj=$(squeue -h -u $USER -n sm_$TAG -o '%i %T' | head -1)
    if [ "$st" = STARTED ] || [ "$st" = DONE ]; then
      case "$rj" in *PENDING*) scancel ${rj%% *} && L "CANCEL own CLUSTER-A loser sm_$TAG job=${rj%% *} (VPS copy $CMD started first)";; esac
    elif [ "$st" = QUEUED ]; then
      case "$rj" in *RUNNING*) $SSHV "powershell -NoProfile -Command \"Rename-Item $VPS_ROOT/uhwr/runner/queue/$CMD.cmd $CMD.cmd.withdrawn\"" && L "WITHDRAW own VPS loser $CMD (CLUSTER-A sm_$TAG job=${rj%% *} started first)";; esac
    fi
    if [ "$st" = DONE ] && [ ! -f $M/RELAYED ]; then
      SUB=ft; [ "$KIND" = smr_luo ] && SUB=preds
      if $SSHV "tar -cf - -C $VR/$TAG $SUB eval STARTED.json SAMPLING_DONE job_history.txt" | tar -xf - -C $M; then
        (cd $M && find . -type f | sort | xargs sha256sum > SHA256SUMS_relay.txt)
        mkdir -p $QP/$TAG; ln -sfn $M/$SUB $QP/$TAG/$SUB; touch $M/RELAYED
        if [ "$KIND" = smr_luo ]; then SK="luoft ."; else SK="ft ft"; fi
        sid=$(env -i PATH=$PATH HOME=$HOME USER=$USER sbatch --parsable -p cpuidle -J smscore_${TAG}_vps $SMC/sm_cpu.sbatch score $TAG $SK)
        L "RELAY VPS $TAG -> $M ($(wc -l < $M/SHA256SUMS_relay.txt) files, sha256 manifest) ; SUBMIT score job=$sid"
      else L "RELAY_FAILED $TAG"; fi
    fi
  done < $T
  sleep 600
done
L "watch end"
