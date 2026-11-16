#!/bin/bash
#SBATCH -p cpuidle
#SBATCH --requeue
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 1-00:00:00
#SBATCH -J fxm_autopilot
#SBATCH -o $U/exp/FMD/xs/runs/fmdx_monitor/logs/autopilot_%j.out
# FMDX warm monitor autopilot (stdlib python3); state in runs/fmdx_monitor/state/warm_state.json (resumes after requeue).
python3 $U/exp/FMD/xs/code/fmdx_monitor_v1/fxm_autopilot.py --until 2026-11-10T18:00:00Z --cycle 600
