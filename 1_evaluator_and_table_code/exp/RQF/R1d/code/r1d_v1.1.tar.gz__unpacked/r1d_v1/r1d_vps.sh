#!/bin/bash
# R1d cross-cluster race lock + transfer hub on the VPS (sourced on CLUSTER-A and CLUSTER-B hosts; key valid until 10 Nov 18:00Z).
VPSH=Administrator@<ip>; VK=$HOME/.ssh/uhwr_staging_vps_ed25519
SSHV="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -i $VK $VPSH"; SCPV="scp -q -o BatchMode=yes -o ConnectTimeout=20 -i $VK"
WR='$VPS_ROOT\uhwr\race_r1d'; WS='$VPS_ROOT/uhwr/stage'
vps_owner() { $SSHV "type $WR\\$1\\owner.txt" 2>/dev/null | tr -d '\r' | head -1; }
vps_has() { $SSHV "if exist $WR\\$1\\$2 (echo Y)" 2>/dev/null | tr -d '\r' | grep -q Y; }
vps_lock() { # UNIT OWNER -> WIN | LOSE <owner> | ERR   (atomic cmd mkdir on the VPS)
  $SSHV "if not exist $WR mkdir $WR" >/dev/null 2>&1 || { echo ERR; return; }
  if $SSHV "mkdir $WR\\$1" >/dev/null 2>&1; then $SSHV "echo $2> $WR\\$1\\owner.txt" >/dev/null 2>&1; echo WIN; return; fi
  local o; o=$(vps_owner $1); [ -z "$o" ] && { sleep 10; o=$(vps_owner $1); }
  if [ -z "$o" ]; then echo ERR; elif [ "$o" = "$2" ]; then echo WIN; else echo "LOSE $o"; fi; }
vps_unlock() { $SSHV "rmdir /s /q $WR\\$1" >/dev/null 2>&1; }
