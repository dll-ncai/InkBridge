#!/bin/bash
# v1.0.1-novps (<author> decision 4 Nov): the VPS hub is OFF; this script never contacts it.
echo "hub disabled (VPS off, <author> decision 4 Nov)" >&2; exit 3
# CLUSTER-A -> VPS hub ($VPS_ROOT/uhwr/stage/<same relative path>). Usage: hub_push.sh NAME ITEM [ITEM ...]
# ITEMs are paths relative to $U (files or dirs; slurm-* and __pycache__ excluded). Writes the sha256
# manifest exp/FMD/xs/admin/xfer/<NAME>.sha256 (relative paths), pushes it together with the items.
set -euo pipefail
R=$U; NAME=$1; shift
SSHV="ssh -o BatchMode=yes -o ConnectTimeout=20 -i $HOME/.ssh/uhwr_staging_vps_ed25519 Administrator@<ip>"
M=exp/FMD/xs/admin/xfer/$NAME.sha256; mkdir -p $R/exp/FMD/xs/admin/xfer
cd $R
find "$@" -type f ! -name 'slurm-*' ! -path '*__pycache__*' -print0 | sort -z | xargs -0 sha256sum > $M
tar --exclude='slurm-*' --exclude='__pycache__' -cf - "$@" $M | $SSHV "tar -xf - -C $VPS_ROOT/uhwr/stage"
echo "$(date -u +%FT%TZ) pushed $NAME: $(wc -l < $M) files, manifest sha256 $(sha256sum $M | cut -c1-64)" | tee -a $R/exp/FMD/xs/admin/xfer/xfer_log.txt
