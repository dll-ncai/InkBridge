#!/bin/bash
D=$U_B; K=$HOME/.ssh/uhwr_staging_vps_ed25519; V=Administrator@<ip>
mkdir -p $D/exp/FMD/xs/admin/mirror $D/exp/FMD/xs/logs; cd $D
for tag in "$@"; do
  t0=$(date +%s)
  ssh -i $K -o BatchMode=yes $V "type $VPS_ROOT\\uhwr\\stage\\exp\\FMD\\xs\\admin\\mirror\\mirror_$tag.list" | tr -d '\r' > /tmp/xs_list_$tag.txt
  echo "$tag list: $(wc -l < /tmp/xs_list_$tag.txt) entries"
  ssh -i $K -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=40 $V "tar -cf - -C $VPS_ROOT/uhwr/stage -T -" < /tmp/xs_list_$tag.txt | tar -xf - -C $D
  echo "$tag pull rc=${PIPESTATUS[0]} $(( $(date +%s)-t0 ))s"
  ssh -i $K -o BatchMode=yes $V "tar -cf - -C $VPS_ROOT/uhwr/stage exp/FMD/xs/admin/mirror/mirror_$tag.sha256 exp/FMD/xs/admin/mirror/mirror_$tag.list" | tar -xf - -C $D
  sha256sum -c exp/FMD/xs/admin/mirror/mirror_$tag.sha256 > exp/FMD/xs/admin/mirror/clusterB_verify_$tag.txt 2>&1; echo "$tag sha256sum -c rc=$? bad=$(grep -vc ': OK' exp/FMD/xs/admin/mirror/clusterB_verify_$tag.txt) total=$(wc -l < exp/FMD/xs/admin/mirror/clusterB_verify_$tag.txt) $(( $(date +%s)-t0 ))s"
done
