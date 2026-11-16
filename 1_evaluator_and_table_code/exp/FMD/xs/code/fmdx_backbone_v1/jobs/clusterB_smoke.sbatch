#!/bin/bash
#SBATCH -J fmdx_bb_smoke
#SBATCH --partition=H100-RP,A100-RP
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -o $U_B/exp/FMD/xs/logs/build/smoke_%j.out
set -u
N=$U_B; IMG=$N/images/pytorch_2.7.1-cuda12.8-cudnn9-runtime.sqsh
srun --ntasks=1 --container-image=$IMG --container-mounts=$SCRATCH_B:$SCRATCH_B,$N:$U,$HOME/.ssh:$HOME/.ssh:ro \
     --container-workdir=$U/exp/FMD/xs \
     bash -c "[ -d $U/exp/FMD/xs/code/pydeps_timm/timm ] || $U/venvs/e0/bin/python -m pip install --quiet --no-deps --target $U/exp/FMD/xs/code/pydeps_timm timm==1.0.30; export FMDX_SITE=clusterB HOME=$HOME SMOKE_P01_NPZ=$U/data/m5_multiscript_v1/m5_val_casia_en.npz SMOKE_P01_R224=$U/exp/FMD/xs/data/backbone/val250_casia_en_render224.npy; bash $U/exp/FMD/xs/code/fmdx_backbone_v1/jobs/smoke.sh $U/exp/FMD/xs/runs/smoke/clusterB_$SLURM_JOB_ID clusterB ${1:-}"
# race lock test on the host (ssh to the VPS hub)
source $N/exp/FMD/xs/code/fmdx_backbone_v1/jobs/common.sh
T="SMOKE_clusterB_$SLURM_JOB_ID"; a=$(race_lock $T clusterB); b=$(race_lock $T clusterB); c=$(race_lock $T clusterA)
echo "RACE_LOCK first=$a again=$b other_lane=$c" | tee -a $N/exp/FMD/xs/runs/smoke/clusterB_$SLURM_JOB_ID/smoke_results.txt
[ "$a" = WIN ] && [ "$b" = WIN ] && [[ "$c" == LOSE* ]]; echo "SMOKE race_lock_host rc=$?" | tee -a $N/exp/FMD/xs/runs/smoke/clusterB_$SLURM_JOB_ID/smoke_results.txt
$SSHV "cmd /c rmdir /s /q $VPS_ROOT\\uhwr\\stage\\exp\\FMD\\xs\\race\\$T" >/dev/null 2>&1
