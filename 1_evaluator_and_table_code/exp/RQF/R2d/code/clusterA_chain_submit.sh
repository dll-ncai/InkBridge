#!/bin/bash
# R2d CLUSTER-A copy A chain (afterok). Writes admin/clusterA_chain_ids.tsv and one SUBMIT line per job to rqf_log.md
RD=$U/exp/RQF/R2d; S=$RD/code/r2d_v1/clusterA_stage.sbatch; L=$U/exp/RQF/queue/rqf_log.md; T=$RD/admin/clusterA_chain_ids.tsv
J2=$1; cd $RD/logs; G1="-p gpuidle --gres=gpu:a30:1 -c 8 --mem=60G -t 1-00:00:00"; G2="--gres=gpu:a30:2 -c 16 --mem=96G"
sb() { local name=$1 dep=$2 res=$3; shift 3; local j; j=$(sbatch --parsable -J $name $res --dependency=afterok:$dep --kill-on-invalid-dep=yes $S "$@") || { echo "SUBMIT_FAILED $name"; return 1; }
  echo -e "$name\t$j\t$dep\t$res" >> $T; echo "$(date -u +%FT%TZ) [R2d, sub-agent <agent>] SUBMIT R2d $name job=$j lane=clusterA:${res// /_} deps=afterok:$dep (copy A; race with CLUSTER-B copy B on LTO start)" >> $L; echo $j; }
LTO=$(sb r2d_lto $J2 "$G2 -t 5-00:00:00" lto); LTOW=$(sb r2d_ltow $LTO "$G2 -t 3-00:00:00" ltow)
SL=$(sb r2d_samp_lto $LTO "$G1" samp_lto); SW=$(sb r2d_samp_ltow $LTOW "$G1" samp_ltow); ALL="$SL:$SW"
for k in 20261034 20261035 20261036; do
  a=$(sb r2d_ftp_${k: -2} $LTO "$G1" ftp $k); b=$(sb r2d_smrp_${k: -2} $LTO "$G1" smrp $k)
  c=$(sb r2d_ftw_${k: -2} $LTOW "$G1" ftw $k); d=$(sb r2d_smrw_${k: -2} $LTOW "$G1" smrw $k)
  for p in "ftp:$a" "smrp:$b" "ftw:$c" "smrw:$d"; do st=${p%%:*}; j=${p#*:}; s=$(sb r2d_samp_${st}_${k: -2} $j "$G1" samp_$st $k); ALL="$ALL:$s"; done
done
SC=$(sb r2d_score $ALL "-p epyc-128 -c 16 --mem=110G -t 2-00:00:00 --export=ALL,R2D_SHM=1" score)
echo "LTO=$LTO LTOW=$LTOW SCORE=$SC n=$(wc -l < $T)"
