#!/bin/bash
#SBATCH -J fmdx_pseudo_pool
#SBATCH -p cpuidle
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o $U/exp/FMD/xs/logs/pseudo/%x_%j.out
# InkSight pseudo-label pool (prereg A4.2): 20,000 TRAIN-pool units per LTR source not in m5_luo_train, GT ink rebuilt, 224 renders,
# then the PNG manifest for the InkSight runner. Gate-aware: exits 0 if verdict_inksight_small_p.json says FAIL.
set -u
source $U/exp/FMD/xs/code/fmdx_backbone_v1/jobs/common.sh
check_code || exit 3
PD=$X/data/pseudo; mkdir -p $PD
python3 -c "import json,sys; d=json.load(open('$X/verdicts/verdict_inksight_small_p.json')); sys.exit(1 if str(d.get('status','')).upper()=='FAIL' else 0)" 2>/dev/null
[ $? -eq 1 ] && { echo "$(utc) InkSight verdict FAIL -> skipped" | tee $PD/SKIPPED_GATE_FAIL.txt; exit 0; }
[ -f $PD/pseudo_pool_summary.json ] || $PY $BB/prep_gt_ink.py pool --out-dir $PD --procs ${SLURM_CPUS_PER_TASK:-16} || exit 4
[ -f $PD/manifest.json.meta.json ] || $PY $BB/pseudo_label.py manifest --pool-dir $PD --out-dir $PD || exit 5
echo "$(utc) pool + manifest done: $(python3 -c "import json;print(len(json.load(open('$PD/manifest.json'))))") images"
