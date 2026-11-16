#!/bin/bash
#SBATCH --partition=A100-40GB,A100-PCI,A100-SDS,A100-RP,H100-RP,H200-SDS,A100-80GB,H100,H100-PCI
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o $U_B/exp/RQF/R1/logs/%x_%j.out
# RQF R1 CLUSTER-B race copy. usage: sbatch -J rqf1d_<JOB> clusterB_r1.sbatch <JOB>. Race lock on VPS hub (first to START wins; loser exits 0). Winner pushes preds to VPS runs\RQF1_CLUSTER-B (CLUSTER-A gate job pulls them). No VPS GPU use.
set -u
JOB=$1; N=$U_B; IMG=$N/images/pytorch_2.7.1-cuda12.8-cudnn9-runtime.sqsh
K=$HOME/.ssh/uhwr_staging_vps_ed25519; V=Administrator@<ip>; SSHV="ssh -o BatchMode=yes -o ConnectTimeout=20 -i $K $V"
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mkdir -p $N/exp/RQF/R1/logs $N/exp/RQF/R1/clusterB_out; H=$N/exp/RQF/R1/logs/race_R1.txt
source $N/exp/RQF/code/rqf1_v1/r1_lock.sh
L=$(r1_lock $JOB clusterB); echo "$(utc) clusterB $JOB lock=$L slurm=$SLURM_JOB_ID host=$(hostname) part=$SLURM_JOB_PARTITION" | tee -a $H
case "$L" in LOSE*) echo "$(utc) clusterB copy lost; exit"; exit 0;; esac
srun --ntasks=1 --container-image=$IMG --container-mounts=$SCRATCH_B:$SCRATCH_B,$N:$U --container-workdir=$U/exp/RQF/R1 \
  bash $U/exp/RQF/code/rqf1_v1/r1_clusterB_inner.sh $JOB
rc=$?; echo "$(utc) sampling rc=$rc"
[ $rc -eq 0 ] || { echo "$(utc) clusterB sampling failed rc=$rc; releasing race lock so the CLUSTER-A copy can run" | tee -a $H; $SSHV "cmd /c rmdir /s /q $VPS_ROOT\\uhwr\\race\\RQF1_$JOB"; exit $rc; }
P=$N/exp/RQF/R1/clusterB_out/preds
(cd $P && sha256sum *_preds.npz *_preds.json > ../SHA256SUMS_preds_$JOB.txt)
$SSHV 'powershell -NoProfile -Command "New-Item -ItemType Directory -Force $VPS_ROOT/uhwr/runs/RQF1_CLUSTER-B/preds | Out-Null"'
scp -q -i $K -o BatchMode=yes -r $P/. $V:$VPS_ROOT/uhwr/runs/RQF1_CLUSTER-B/preds/; rc2=$?
echo "$(utc) push to VPS rc=$rc2" | tee -a $H
[ $rc2 -eq 0 ] && $SSHV "cmd /c echo $JOB done $(utc)> $VPS_ROOT\\uhwr\\runs\\RQF1_CLUSTER-B\\${JOB}.done"
exit $rc2
