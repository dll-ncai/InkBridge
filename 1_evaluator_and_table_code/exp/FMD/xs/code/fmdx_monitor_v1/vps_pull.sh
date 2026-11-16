#!/bin/bash
# FMDX warm monitor: VPS -> CLUSTER-A pull with sha check. Usage: vps_pull.sh REL_DIR   (REL_DIR relative to the VPS stage = relative to $U)
# 1) VPS side: sha_dir.ps1 writes <REL_DIR>/SHA256SUMS_vps.txt (sha256 of every file computed ON THE VPS)  2) tar over ssh into R  3) sha256sum -c on CLUSTER-A.
# One hub stream per track: run one pull at a time.
set -euo pipefail
R=$U; REL=${1%/}; WIN=${REL//\//\\}
SSHV="ssh -o BatchMode=yes -o ConnectTimeout=20 -i $HOME/.ssh/uhwr_staging_vps_ed25519 Administrator@<ip>"
$SSHV "powershell -NoProfile -File $VPS_ROOT\\uhwr\\stage\\exp\\FMD\\xs\\code\\fmdx_monitor_v1\\sha_dir.ps1 -Rel $WIN" | tr -d '\r'
mkdir -p $R/$REL
$SSHV "tar -cf - -C $VPS_ROOT/uhwr/stage $REL" | tar -xf - -C $R
sed -i 's/\r$//' $R/$REL/SHA256SUMS_vps.txt
cd $R && sha256sum -c --quiet $REL/SHA256SUMS_vps.txt && echo "$(date -u +%FT%TZ) pulled $REL: $(wc -l < $REL/SHA256SUMS_vps.txt) files sha256 OK (manifest sha256 $(sha256sum $REL/SHA256SUMS_vps.txt | cut -c1-64))" | tee -a $R/exp/FMD/xs/runs/fmdx_monitor/logs/xfer_log.txt
