#!/bin/bash
# RQF A2.3 SM-R / SM-A: submit CLUSTER-A smoke (200 steps) + 64 chains (GPU train->select->sample) + afterok scoring + final read (afterany).
set -u; source $U/exp/RQF/code/sm_v1/sm_env.sh
cd $Q/logs; A=$Q/admin/sm_clusterA_jobs.tsv; : > $A
EXC="dgx[001-004],gpu[003-004,013-014,017-023,119-148]"
G="sbatch --parsable -p gpuidle --gres=gpu:1 --exclude=$EXC"
WHO="rqf-sm <agent>"
log() { rqf_log "[$WHO] $*"; }
sub() { local name=$1; shift; local id; id=$("$@" 2>&1 | tail -1); echo "$id"; }
# smoke (200 steps; CLUSTER-A GPU)
NB=$X/runs/NOFM_W_clusterA/bb_ms$(cat $X/runs/NOFM_W_clusterA/eval/best_milestone.txt).pt
E="--export=ALL,SM_SMOKE_STEPS=200,FT_STEPS=200,FT_SAVE=200,SM_SMOKE_SETS=test_ouhdl test_casia_zh,SM_SMOKE_SEEDS=0,SM_SMOKE_MAXU=100"
for spec in "smr_luo.sh SMOKE_SMR_LUO p01 20261034" "smr_warm.sh SMOKE_SMR_NOFM_W NOFM_W $NB 20261034 p01" "sma_chain.sh SMOKE_SMA_LUO luo $LUO_CKPT p01 20261034" "sma_chain.sh SMOKE_SMA_NOFM_W bb $NB p01 20261034"; do
  set -- $spec; id=$(sub x $G -t 0-04:00:00 -J sm_smoke_$2 "$E" $SMC/sm_gpu.sbatch $spec)
  printf "smoke\t%s\tclusterA:gpuidle\t%s\t-\t%s\n" $2 $id $QR/$2 >> $A; log "SUBMIT A2.3 smoke $2 job=$id lane=clusterA:gpuidle"
done
SCORE_IDS=""
BBS="LUO NOFM_W PEINIT_W c-radiov3-h_W vit-mae-large_W dinov3-vitl16-sat493m_W cleandift-sd21_W pe-spatial-b16-512_W"
for meth in SMR SMA; do for bb in $BBS; do for fs in p01:20261034 p01:20261035 p01:20261036 p10:20261034; do
  frac=${fs%:*}; seed=${fs#*:}; TAG=${meth}_${bb}_${frac}_s${seed}; nice=50; [ $fs = p01:20261034 ] && nice=0
  if [ $bb = LUO ]; then BK=$LUO_CKPT; else BK=$X/runs/${bb}_clusterA/bb_ms$(cat $X/runs/${bb}_clusterA/eval/best_milestone.txt).pt; fi
  if [ $meth = SMR ] && [ $bb = LUO ]; then args="smr_luo.sh $TAG $frac $seed"; SK="luoft ."
  elif [ $meth = SMR ]; then args="smr_warm.sh $TAG $bb $BK $seed $frac"; SK="ft ft"
  elif [ $bb = LUO ]; then args="sma_chain.sh $TAG luo $BK $frac $seed"; SK="ft ft"
  else args="sma_chain.sh $TAG bb $BK $frac $seed"; SK="ft ft"; fi
  gid=$(sub x $G --nice=$nice -J sm_$TAG $SMC/sm_gpu.sbatch $args)
  sid=$(sub x sbatch --parsable -p cpuidle --nice=$nice --kill-on-invalid-dep=yes --dependency=afterok:$gid -J smscore_$TAG $SMC/sm_cpu.sbatch score $TAG $SK)
  printf "A2.3-%s\t%s\tclusterA:gpuidle\t%s\t-\t%s\n" $meth $TAG $gid $QP/$TAG >> $A
  printf "A2.3-%s-score\t%s\tclusterA:cpuidle\t%s\tafterok:%s\t%s\n" $meth $TAG $sid $gid $QS/rows >> $A
  log "SUBMIT A2.3 $meth $TAG gpu=$gid score=$sid(afterok:$gid) lane=clusterA:gpuidle nice=$nice backbone=$BK"
  SCORE_IDS="$SCORE_IDS:$sid"
done; done; done
rid=$(sub x sbatch --parsable -p cpuidle --dependency=afterany$SCORE_IDS -J sm_read $SMC/sm_cpu.sbatch read)
printf "A2.3-read\tall\tclusterA:cpuidle\t%s\tafterany:all_score\t%s\n" $rid $Q/tables/RQF_SM >> $A
log "SUBMIT A2.3 read sm_read.py job=$rid deps=afterany(all 64 score jobs)"
echo SUBMIT_DONE $(wc -l < $A)
