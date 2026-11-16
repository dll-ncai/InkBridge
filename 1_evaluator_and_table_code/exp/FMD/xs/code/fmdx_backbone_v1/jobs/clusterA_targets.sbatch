#!/bin/bash
#SBATCH -J fmdx_tgt
#SBATCH -p gpuidle
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx[001-005],gpu[004-005,013-014,017-023,119-148]
#SBATCH --cpus-per-task=8
#SBATCH --mem=46G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o $U/exp/FMD/xs/logs/targets/%x_%A_%a.out
# REPA targets (prereg A4.2 / A5.2): array task -> line $SLURM_ARRAY_TASK_ID of TASKS = "<fm> <shard> <n_shards>" (capacity-note
# order; slow FMs such as the diffusion ones are split into shards; the last shard to finish merges).
# Gate-aware: prep_repa_targets.py reads verdict_<fm>.json first and skips FAIL FMs. Requeue-safe (chunk markers).
# Usage: sbatch --array=1-K[%P] clusterA_targets.sbatch TASKS_FILE SUBSET_POS_NPY SUBSET_NAME
set -u
source $U/exp/FMD/xs/code/fmdx_backbone_v1/jobs/common.sh
check_code || exit 3
read -r FM SH NSH <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" $1)"; [ -n "$FM" ] || { echo "no fm for task $SLURM_ARRAY_TASK_ID"; exit 2; }; SH=${SH:-0}; NSH=${NSH:-1}
mkdir -p $X/targets/repa $X/logs/targets
echo "$(utc) task $SLURM_ARRAY_TASK_ID fm $FM shard $SH/$NSH host $(hostname) gpu $(gpu_names) restart ${SLURM_RESTART_COUNT:-0}"
PPX=""; case "$FM" in *deepseek*) PPX=$X/pylibs_ds: ;; esac      # as the screen gen-VLM jobs (GV_DS)
$WRAPFM FMD_HARNESS=$HC HF_HOME=$X/weights/.hf_home_targets PYTHONPATH=${PPX}$BB:$HC:$R/exp/FMD/code/fmd_harness_xs:$UE:$X/pylibs \
   $FMDVENV/bin/python -u $BB/prep_repa_targets.py --fm $FM --registry $BB/fm_registry.json --render-npy $LUO_R224 --render-unit-ids $LUO_UIDS --subset-pos $2 --subset-name $3 \
   --out-dir $X/targets/repa --verdict-dir $X/verdicts --chunk 256 --shard $SH --n-shards $NSH
rc=$?; echo "$(utc) end fm $FM rc $rc"; exit $rc
