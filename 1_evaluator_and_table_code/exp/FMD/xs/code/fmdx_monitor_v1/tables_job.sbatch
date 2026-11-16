#!/bin/bash
#SBATCH -p cpuidle
#SBATCH --requeue
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 03:00:00
#SBATCH -J fxm_tables
#SBATCH -o $U/exp/FMD/xs/runs/fmdx_monitor/logs/tables_%j.log
# FMDX warm monitor: rebuild warm tables (no verdicts). Arg1 = read label.
set -e; cd $U/exp/FMD/xs/runs/fmdx_monitor/tables
apptainer exec -B $SCRATCH_ROOT $U/images/torch27.sif $U/venvs/cpu/bin/python $U/exp/FMD/xs/code/fmdx_monitor_v1/build_tables.py --rows-dir $U/exp/FMD/xs/runs/fmdx_monitor/score_out/rows --out $U/exp/FMD/xs/runs/fmdx_monitor/tables/FMDX_warm --arm-meta $U/exp/FMD/xs/runs/fmdx_monitor/tables/arm_meta.json --read-label "$1" --luo-dir $U/exp/FMD/xs/runs/fmdx_monitor/score_out_cmp/rows --comp-dir $U/exp/FMD/xs/runs/fmdx_monitor/score_out_cmp/rows --luo-tag LUO_ms6_identity --comp-device-note "comparators re-sampled on CLUSTER-A RTX A4000 (prereg A8)"
sha256sum FMDX_warm_absolute.csv FMDX_warm_contrasts.csv FMDX_warm_contrasts.md FMDX_warm_inputs.json > FMDX_warm_SHA256.txt
echo TABLES_DONE $(date -u +%FT%TZ)
