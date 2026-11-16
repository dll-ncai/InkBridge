#!/bin/bash
#SBATCH -J fmdx_controller
#SBATCH -p batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=3-00:00:00
#SBATCH -o $U_B/exp/FMD/xs/logs/controller/%x_%j.out
# FMDX controller, CLUSTER-B twin: pulls verdicts + arms.tsv + priority.json from the VPS hub (sha-checked against SHA256SUMS_hub.txt),
# applies gate cancellations / race losses / priority to CLUSTER-B fx_<arm> jobs, pushes its log to the hub for the CLUSTER-A side.
# 3-day partition limit: each job queues its continuation (afterany) until 2026-11-10T23:59Z.
set -u
N=$U_B; X=$N/exp/FMD/xs; mkdir -p $X/logs/controller $X/admin
[ "$(date -u +%Y%m%d%H)" \< "2026111023" ] && sbatch --dependency=afterany:$SLURM_JOB_ID $X/code/fmdx_backbone_v1/jobs/clusterB_controller.sbatch >> $X/admin/controller_clusterB_segments.txt
python3 $X/code/fmdx_backbone_v1/controller/gate_controller.py --site clusterB --root $N --arms $X/admin/arms.tsv --cap 8 --period 300 --until 2026-11-10T23:59:00Z
