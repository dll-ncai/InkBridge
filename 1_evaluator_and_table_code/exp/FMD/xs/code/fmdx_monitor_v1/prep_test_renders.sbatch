#!/bin/bash
#SBATCH -J fmdxm_prep_r224
#SBATCH -p cpuidle
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o $U/exp/FMD/xs/runs/fmdx_monitor/logs/prep_test_renders_%j.out
# FMDX warm monitor: 224-px renders (prep_renders.py gtset, frozen fmdx_backbone_v1 code, unchanged) for the six non-Urdu test sets.
# Sets and gt npz as prereg A4.1. Output: X/data/monitor_v1/test_<set>_render224.{npz,npy}(+ .meta.json). Idempotent.
set -u
source $U/exp/FMD/xs/code/fmdx_backbone_v1/jobs/common.sh
check_code || exit 3
MD=$X/data/monitor_v1; mkdir -p $MD
echo "$(utc) start host $(hostname) code_sha=$(cd $BB && sha256sum SHA256SUMS.txt | cut -c1-64)"
while read -r S NPZ; do
  [ -f $MD/test_${S}_render224.npz.meta.json ] || $PY $BB/prep_renders.py gtset --npz $NPZ --out $MD/test_${S}_render224.npz --procs ${SLURM_CPUS_PER_TASK:-16} || { echo "FAILED $S"; exit 4; }
done <<EOF
test_casia_zh $R/exp/X4a/data/x4a_casia_zh.npz
test_casia_en $R/exp/X4a/data/x4a_casia_en.npz
test_hpl_tamil $R/exp/X4a/data/x4a_hpl_tamil.npz
test_iam500 $R/data/m5_multiscript_v1/m5_test_iam500.npz
german_unipen_de_test $R/data/d2_v2/unipen_de/e0_test.npz
german_onhw_all $R/data/d2_v2/onhw/e0_all.npz
EOF
(cd $MD && sha256sum test_*_render224.np[yz] > SHA256SUMS_test_renders.txt)
echo "$(utc) end"
