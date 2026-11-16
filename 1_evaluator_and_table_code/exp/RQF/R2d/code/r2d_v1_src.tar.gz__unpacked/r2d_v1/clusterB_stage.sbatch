#!/bin/bash
#SBATCH --signal=TERM@300
#SBATCH -o $U_B/exp/RQF/R2d/logs/%x_%j.out
# RQF R2d CLUSTER-B stage segment (Pyxis container, $U_B bind-mounted at $U).
# Usage: bash clusterB_sub.sh STAGE [SEED]. Segment logic: STOP_clusterB -> exit; done -> trigger downstream; else queue the
# continuation (afterany:self, same options), run the stage; on success trigger downstream and cancel the continuation;
# on GATE_FAIL or a non-timeout error cancel the continuation and write FAILED_<stage>. 'lto' waits for admin/SMOKE_PASS
# (written after the CLUSTER-A CPU smoke passed) and builds the LTO train set + renders in the same job.
set -u
ST=$1; SEED=${2:-}; source $U_B/exp/RQF/R2d/code/r2d_v1/clusterB_sub.sh
IMG=$N/images/pytorch_2.7.1-cuda12.8-cudnn9-runtime.sqsh; EVL=$RDN/admin/r2d_events.log; mkdir -p $RDN/admin $RDN/logs
ev() { echo "$(date -u +%FT%TZ) [R2d clusterB] $* slurm=${SLURM_JOB_ID:-none} restart=${SLURM_RESTART_COUNT:-0} host=$(hostname) part=${SLURM_JOB_PARTITION:-none}" >> $EVL; }
runname() { case $ST in lto) echo LTO;; ltow) echo LTO_W;; ftp) echo LTO_FT_s$SEED;; ftw) echo LTO_W_FT_s$SEED;; smrp) echo SMR_LTO_s$SEED;; smrw) echo SMR_LTO_W_s$SEED;; esac; }
sampname() { case $ST in samp_lto) echo LTO;; samp_ltow) echo LTO_W;; samp_ftp) echo LTO_FT_s$SEED;; samp_ftw) echo LTO_W_FT_s$SEED;; samp_smrp) echo SMR_LTO_s$SEED;; samp_smrw) echo SMR_LTO_W_s$SEED;; esac; }
is_done() { case $ST in score) [ -s $RDN/score_out/tables/R2d_rules.json ];; samp_*) [ -f $RDN/preds/$(sampname)/SAMPLING_DONE ];; *) [ -s $RDN/runs/$(runname)/eval/best_milestone.txt ];; esac; }
is_gate() { case $ST in samp_*|score) false;; *) [ -f $RDN/runs/$(runname)/GATE_FAIL ];; esac; }
SEEDS="20261034 20261035 20261036"
ALLS=(LTO LTO_W); for k in $SEEDS; do ALLS+=(LTO_FT_s$k LTO_W_FT_s$k SMR_LTO_s$k SMR_LTO_W_s$k); done
trig() { local d=$RDN/admin/.trig_${ST}_${SEED:-x}; mkdir $d 2>/dev/null || return 0; local L=()
  case $ST in lto) L=("ltow ''" "samp_lto ''"); for k in $SEEDS; do L+=("ftp $k" "smrp $k"); done;;
    ltow) L=("samp_ltow ''"); for k in $SEEDS; do L+=("ftw $k" "smrw $k"); done;;
    ftp|ftw|smrp|smrw) L=("samp_$ST $SEED");;
    samp_*) for n in "${ALLS[@]}"; do [ -f $RDN/preds/$n/SAMPLING_DONE ] || return 0; done; mkdir $RDN/admin/.score_submitted 2>/dev/null && L=("score ''");; esac
  for x in "${L[@]}"; do eval set -- $x; j=$(sub $1 "$2") && ev "TRIGGER $ST${SEED:+ $SEED} -> submitted $1 $2 job=$j" || ev "TRIGGER_FAILED $ST -> $1 $2"; done; }
[ -e $RDN/admin/STOP_clusterB ] && { ev "STOP_clusterB present: $ST $SEED not run"; exit 0; }
if is_done; then ev "already done $ST $SEED"; trig; exit 0; fi
if is_gate; then ev "GATE_FAIL present $ST $SEED: nothing to run"; exit 0; fi
if [ "$ST" = lto ] && [ ! -f $RDN/admin/SMOKE_PASS ]; then
  j=$(sub lto "" --begin=now+30minutes); ev "WAIT_SMOKE lto: SMOKE_PASS absent, resubmitted as $j"; exit 0; fi
NSEG=$(grep -c "SEGMENT_START $ST${SEED:+ $SEED} " $EVL 2>/dev/null); NSEG=${NSEG:-0}
[ $NSEG -ge 8 ] && { ev "SEGMENT_LIMIT $ST $SEED"; touch $RDN/admin/FAILED_${ST}_${SEED:-x}; exit 1; }
ev "SEGMENT_START $ST${SEED:+ $SEED} "
NEXT=$(sub $ST "$SEED" --dependency=afterany:$SLURM_JOB_ID) && ev "continuation $ST $SEED queued $NEXT"
TERMED=0; trap 'TERMED=1' TERM
CR=$U/exp/RQF/R2d/code/r2d_v1
if [ "$ST" = lto ]; then CMD="bash $CR/stage.sh data_lto_build && { bash $CR/stage.sh data_lto_render > $U/exp/RQF/R2d/logs/data_lto_render_\$SLURM_JOB_ID.log 2>&1 & RP=\$!; bash $CR/stage.sh lto; rc=\$?; wait \$RP; rr=\$?; [ \$rc = 0 ] && exit \$rr; exit \$rc; }"
else CMD="bash $CR/stage.sh $ST $SEED"; fi
srun --ntasks=1 --container-image=$IMG --container-mounts=$SCRATCH_B:$SCRATCH_B,$N:$U --container-workdir=$U/exp/RQF/R2d \
  bash -c "export R2D_SITE=clusterB HOME=$HOME SLURM_JOB_ID=$SLURM_JOB_ID; $CMD" &
SP=$!; wait $SP; rc=$?; [ $TERMED = 1 ] && { wait $SP; rc=$?; }
ev "SEGMENT_END $ST $SEED rc=$rc termed=$TERMED"
if is_done; then scancel $NEXT 2>/dev/null; trig; exit 0; fi
if is_gate; then scancel $NEXT 2>/dev/null; ev "GATE_FAIL $ST $SEED (chain stops)"; exit 20; fi
if [ $TERMED = 0 ] && [ $rc != 0 ]; then scancel $NEXT 2>/dev/null; touch $RDN/admin/FAILED_${ST}_${SEED:-x}; ev "FAILED $ST $SEED rc=$rc (continuation cancelled)"; exit $rc; fi
exit $rc
