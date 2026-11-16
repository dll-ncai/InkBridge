#!/bin/bash
#SBATCH -J xs_push
#SBATCH -p cpuidle
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 06:00:00
#SBATCH -o $U/exp/FMD/xs/logs/push_%j.out
# usage: sbatch mirror_push.sbatch <tag> ...   (CLUSTER-A -> VPS hub stage dir, sha256 listing created here and verified on the VPS)
R=$U; K=$HOME/.ssh/uhwr_staging_vps_ed25519; V=Administrator@<ip>
cd $R
for tag in "$@"; do
  L=exp/FMD/xs/admin/mirror/mirror_$tag.list; S=exp/FMD/xs/admin/mirror/mirror_$tag.sha256
  t0=$(date +%s)
  xargs -d '\n' -a $L sha256sum -b > $S; echo "$tag sha256 listing: $(wc -l < $S) files $(( $(date +%s)-t0 ))s"
  (cat $L; echo $L; echo $S) | tar -cf - -T - | ssh -i $K -o BatchMode=yes -o ServerAliveInterval=30 $V "tar -xf - -C $VPS_ROOT/uhwr/stage"
  echo "$tag push rc=${PIPESTATUS[1]} $(( $(date +%s)-t0 ))s"
done
