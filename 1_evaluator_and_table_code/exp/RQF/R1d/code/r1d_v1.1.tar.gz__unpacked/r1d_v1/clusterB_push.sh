#!/bin/bash
# R1d CLUSTER-B -> VPS return (afterok on the CLUSTER-B copy). Only if this CLUSTER-B copy won: scp each output (npz + json) to the VPS hub
# $VPS_ROOT/uhwr/stage/exp/RQF/R1d/preds/<BASE>/preds/, verify the remote sha256 (certutil), then publish race_r1d\<UNIT>\done.sha256.
set -uo pipefail; N=$U_B; RDN=$N/exp/RQF/R1d; C=$RDN/code/r1d_v1; source $C/r1d_vps.sh; U=$1
[ -e $RDN/race/$U.clusterB_won ] || { echo "not the winner: nothing to push"; exit 0; }
read B FILES < <(python3 -c "import json;u=json.load(open('$C/units.json'))['$U'];print(u['base'],' '.join(u['files']))")
D=$RDN/preds/$B/preds; M=$RDN/race/$U.done.sha256; : > $M
WD="$VPS_ROOT\\uhwr\\stage\\exp\\RQF\\R1d\\preds\\$B\\preds"; $SSHV "if not exist $WD mkdir $WD" >/dev/null 2>&1
for f in $FILES; do [ -f $D/$f ] || { echo "MISSING $f"; exit 2; }; s=$(sha256sum $D/$f | cut -c1-64)
  for t in 1 2 3; do $SCPV $D/$f $VPSH:$WS/exp/RQF/R1d/preds/$B/preds/$f && \
     r=$($SSHV "certutil -hashfile $WD\\$f SHA256" | tr -d '\r' | sed -n 2p) && [ "$r" = "$s" ] && break; r=x; sleep 30; done
  [ "$r" = "$s" ] || { echo "PUSH FAIL $f"; exit 3; }; echo "$s  $f" >> $M; done
$SCPV $M $VPSH:$VPS_ROOT/uhwr/race_r1d/$U/done.tmp && $SSHV "move /y $WR\\$U\\done.tmp $WR\\$U\\done.sha256" >/dev/null && echo "$(date -u +%FT%TZ) PUSH_OK $U $(wc -l < $M) files"
