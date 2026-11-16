#!/bin/bash
# CLUSTER-B staging from the VPS hub ($VPS_ROOT\uhwr\stage\<CLUSTER-A-relative path>) to $U_B/<same path>, sha256-checked
# against clusterB_stage_manifest.sha256 (CLUSTER-A shas). Never overwrites an existing file; a mismatching existing file is an error.
set -uo pipefail; N=$U_B; C=$N/exp/RQF/R1d/code/r1d_v1; source $C/r1d_vps.sh
while read s p; do f=$N/$p
  if [ -f $f ]; then [ "$(sha256sum $f | cut -c1-64)" = $s ] && continue; echo "MISMATCH existing $p"; exit 1; fi
  mkdir -p $(dirname $f); $SCPV $VPSH:$WS/$p $f.r1dpart || { echo "SCP FAIL $p"; exit 1; }
  [ "$(sha256sum $f.r1dpart | cut -c1-64)" = $s ] && mv -n $f.r1dpart $f && echo "staged $p" || { echo "SHA FAIL $p"; rm -f $f.r1dpart; exit 1; }
done < $C/clusterB_stage_manifest.sha256
echo "$(date -u +%FT%TZ) STAGE_OK"
