#!/bin/bash
#SBATCH --partition=H100-RP,A100-RP
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --signal=TERM@300
#SBATCH -o $U_B/exp/FMD/xs/logs/arms/%x_%j.out
# FMDX backbone arm, CLUSTER-B copy (*-RP partitions: the only CLUSTER-B partitions with same-day starts on 3 Nov; lead) (Pyxis, $U_B bind-mounted at $U).
# Usage: sbatch -J fx_<ARM> [--nice N] clusterB_arm.sbatch <ARM>
# 1-day segments: each segment first queues its continuation (afterany:self, same name) unless the arm is finished,
# gate-failed, race-lost or STOPped; training resumes from checkpoints/latest.pt.
set -u
ARM=$1; N=$U_B; IMG=$N/images/pytorch_2.7.1-cuda12.8-cudnn9-runtime.sqsh
RUN=$N/exp/FMD/xs/runs/${ARM}_clusterB; mkdir -p $RUN $N/exp/FMD/xs/logs/arms
SELF=$N/exp/FMD/xs/code/fmdx_backbone_v1/jobs/clusterB_arm.sbatch
echo "$(date -u +%FT%TZ) segment $SLURM_JOB_ID arm $ARM host $(hostname) partition $SLURM_JOB_PARTITION"
done_marker() { [ -f $RUN/eval/best_milestone.txt ] || [ -f $RUN/GATE_FAIL ] || [ -f $RUN/RACE_LOST.json ] || [ -e $RUN/STOP ]; }
ft_trigger() {  # v1.0.4 (addendum_8, lead 5 Nov): a selected backbone milestone exists -> submit the Urdu p01 FT + relay job once (on CLUSTER-B)
  [ -f $RUN/eval/best_milestone.txt ] || return 0
  mkdir $RUN/.ft_submitted 2>/dev/null || return 0
  FJ=$(sbatch --parsable -J fxft_$ARM --nice=${NICE:-0} $N/exp/FMD/xs/code/fmdx_clusterB_post_v1/clusterB_ft.sbatch $ARM) && echo "$(date -u +%FT%TZ) FT job $FJ" >> $RUN/segments.txt
}
if done_marker; then ft_trigger; echo "arm finished / gate-failed / race-lost / STOP: nothing to do"; exit 0; fi
source $N/exp/FMD/xs/code/fmdx_backbone_v1/jobs/common.sh      # host side: only race_lock / utc are used here
X=$N/exp/FMD/xs; IMG=$N/images/pytorch_2.7.1-cuda12.8-cudnn9-runtime.sqsh   # v1.0.3 (addendum_8): common.sh sets CLUSTER-A-side X ($SCRATCH_ROOT) and IMG (torch27.sif); host-side race_lock and srun need the CLUSTER-B paths
if [ ! -f $RUN/training_complete.json ]; then
  L=$(race_lock $ARM clusterB); echo "$(date -u +%FT%TZ) race_lock $L slurm=$SLURM_JOB_ID host=$(hostname)" >> $RUN/job_history.txt
  case "$L" in LOSE*) echo "{\"arm\":\"$ARM\",\"lane\":\"clusterB\",\"slurm\":\"$SLURM_JOB_ID\",\"owner\":\"${L#LOSE }\",\"utc\":\"$(date -u +%FT%TZ)\"}" > $RUN/RACE_LOST.json
    scancel --name=fx_$ARM --state=PENDING -u $USER 2>/dev/null || true; exit 0;; esac
fi
NICE=$(squeue -h -j $SLURM_JOB_ID -o %y 2>/dev/null | tr -d ' '); NICE=${NICE:-0}
NEXT=$(sbatch --parsable -J fx_$ARM --nice=$NICE --dependency=afterany:$SLURM_JOB_ID $SELF $ARM) && echo "$(date -u +%FT%TZ) $SLURM_JOB_ID -> next segment $NEXT" >> $RUN/segments.txt
srun --ntasks=1 --container-image=$IMG --container-mounts=$SCRATCH_B:$SCRATCH_B,$N:$U,$HOME/.ssh:$HOME/.ssh:ro \
     --container-workdir=$U/exp/FMD/xs \
     bash -c "test -d $U/exp/FMD/xs/code/fmdx_backbone_v1 || { echo 'bind mount missing'; exit 4; }; export FMDX_SITE=clusterB FMDX_LOCK_DONE=1 HOME=$HOME; bash $U/exp/FMD/xs/code/fmdx_backbone_v1/jobs/arm.sh $ARM clusterB"
rc=$?; echo "$(date -u +%FT%TZ) arm.sh exit $rc"
ft_trigger
if done_marker; then scancel --name=fx_$ARM --state=PENDING -u $USER 2>/dev/null || true; fi
exit $rc
