#!/bin/bash
# R1d CLUSTER-A poll (one per base; releases apply via afterok). Per unit: VPS owner 'clusterB' -> scancel this unit's PENDING CLUSTER-A copies
# (own losing copies, logged) and, once race_r1d\<UNIT>\done.sha256 exists, pull every listed file from the VPS hub into
# $RD/preds/<BASE>/preds with sha256 checks; owner 'clusterA' -> resolved when no CLUSTER-A copy of the unit is left in the queue;
# no owner and no CLUSTER-A copy left (CLUSTER-B copy failed after the CLUSTER-A copies were cancelled) -> resubmit the CLUSTER-A race pair.
set -uo pipefail; exec 2>&1
source $U/exp/RQF/R1d/code/r1d_v1/r1d_env.sh; source $RC/r1d_vps.sh
B=$1; SB="env -i HOME=$HOME USER=$USER PATH=/usr/bin:/bin /usr/bin/sbatch --parsable"
UNITS=$(python3 -c "import json;print(' '.join(k for k,v in json.load(open('$RC/units.json')).items() if v['base']=='$B'))")
while true; do left=0; Q=" $(squeue -h -u $USER -o %i | tr '\n' ' ') "
  for u in $UNITS; do [ -e $RD/race/$u.resolved ] && continue; left=1; o=$(vps_owner $u); ids=$(cat $RD/race/$u.ids)
    inq=0; for j in $ids; do [[ "$Q" == *" $j "* ]] && inq=1; done
    if [ "$o" = clusterB ]; then
      for j in $ids; do [ "$(squeue -h -j $j -o %T 2>/dev/null)" = PENDING ] && /usr/bin/scancel $j && r1d_log "CANCEL job=$j (own losing CLUSTER-A copy of R1d unit $u; CLUSTER-B copy holds the VPS lock)"; done
      if vps_has $u done.sha256; then
        $SCPV $VPSH:$VPS_ROOT/uhwr/race_r1d/$u/done.sha256 $RD/race/$u.done.sha256 || continue; ok=1; mkdir -p $RD/preds/$B/preds
        while read s f; do d=$RD/preds/$B/preds/$f; [ -f $d ] && [ "$(sha256sum $d | cut -c1-64)" = $s ] && continue
          $SCPV $VPSH:$WS/exp/RQF/R1d/preds/$B/preds/$f $d.part && [ "$(sha256sum $d.part | cut -c1-64)" = $s ] && mv $d.part $d || { ok=0; rm -f $d.part; }
        done < $RD/race/$u.done.sha256
        [ $ok = 1 ] && touch $RD/race/$u.resolved && r1d_log "PULL R1d $u from VPS hub ($(wc -l < $RD/race/$u.done.sha256) files, sha256 ok) -> $RD/preds/$B/preds"
      fi
    elif [ "$o" = clusterA ]; then [ $inq = 0 ] && touch $RD/race/$u.resolved
    elif [ $inq = 0 ]; then
      rm -rf $RD/race/$u; mapfile -t A < <(python3 -c "import json;print('\n'.join(json.load(open('$RC/units.json'))['$u']['args']))")
      T=$(python3 -c "import json;print(json.load(open('$RC/units.json'))['$u']['time'])"); OPT="--requeue --open-mode=append -o $RD/logs/%x_%j.out"
      a=$($SB -J r1d_$u -p gpuidle --gres=gpu:a4000:1 -c 8 --mem=24G -t $T $OPT $RC/samp.sh $u "${A[@]}"); h=$($SB -J r1d_$u -p gpuidle --gres=gpu:h100:1 -c 8 --mem=32G -t $T $OPT $RC/samp.sh $u "${A[@]}")
      echo "$a $h" > $RD/race/$u.ids; r1d_log "SUBMIT R1d samp $u (resubmit by poll: no lock owner, no copy left) jobs=$a,$h lane=clusterA:gpuidle-a4000/h100"
    fi
  done
  [ $left = 0 ] && { echo "$(utc) all units of $B resolved"; exit 0; }; sleep 600
done
