#!/bin/bash
# v1.0.1-novps (<author> decision 4 Nov): the VPS hub is OFF; this script never contacts it.
echo "hub disabled (VPS off, <author> decision 4 Nov)" >&2; exit 3
# CLUSTER-B <- VPS hub. Usage: hub_pull.sh NAME ITEM [ITEM ...]  (same NAME/ITEMs as the CLUSTER-A push); verifies every file
# against the pushed manifest (sha256sum -c) under $U_B.
set -euo pipefail
N=$U_B; NAME=$1; shift
SSHV="ssh -o BatchMode=yes -o ConnectTimeout=20 -i $HOME/.ssh/uhwr_staging_vps_ed25519 Administrator@<ip>"
M=exp/FMD/xs/admin/xfer/$NAME.sha256
$SSHV "tar -cf - -C $VPS_ROOT/uhwr/stage $* $M" | tar -xf - -C $N
cd $N && sha256sum -c --quiet $M && echo "$(date -u +%FT%TZ) pulled $NAME: $(wc -l < $M) files sha256 OK" | tee -a $N/exp/FMD/xs/admin/xfer/xfer_log_clusterB.txt
