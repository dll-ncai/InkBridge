#!/bin/bash
set -uo pipefail
D=$U/exp/RQF/runs/R3/C34_4090/preds; mkdir -p $D
K="-o BatchMode=yes -o ConnectTimeout=20 -i $HOME/.ssh/uhwr_staging_vps_ed25519"; V=Administrator@<ip>
W='$VPS_ROOT\uhwr\runs\RQF\R3\C34_4090\preds'
ssh $K $V "powershell -NoProfile -Command \"Get-ChildItem $W -File | ForEach-Object { (Get-FileHash -Algorithm SHA256 \$_.FullName).Hash.ToLower() + '  ' + \$_.Name }\"" | tr -d '\r' | grep -E '^[0-9a-f]{64}  ' > $D/../SHA256_vps.txt
echo "vps hashes: $(wc -l < $D/../SHA256_vps.txt)"
while read -r h f; do scp -q $K "$V:$VPS_ROOT/uhwr/runs/RQF/R3/C34_4090/preds/$f" "$D/$f" || echo "SCPFAIL $f"; done < $D/../SHA256_vps.txt
scp -q $K "$V:$VPS_ROOT/uhwr/runs/RQF/R3/C34_4090/job_history.txt" "$D/../job_history.txt" || true
cd $D && sha256sum -c ../SHA256_vps.txt > ../SHA256_check.txt 2>&1; echo "check rc=$?"; grep -c ': OK' ../SHA256_check.txt
