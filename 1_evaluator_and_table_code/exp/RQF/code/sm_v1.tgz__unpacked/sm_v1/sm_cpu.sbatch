#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --open-mode=append
#SBATCH -o $U/exp/RQF/logs/%x_%j.out
# RQF A2.3 CPU step: score one run (rqf_v1/post_run.sh unchanged) or run the A2.2 read. Usage: sm_cpu.sbatch score TAG KIND SUB | read
set -u; exec 2>&1
echo "$(date -u +%FT%TZ) job $SLURM_JOB_ID host $(hostname) $*"
if [ "$1" = score ]; then bash $U/exp/RQF/code/rqf_v1/post_run.sh $2 $3 $4; rc=$?
else apptainer exec -B $SCRATCH_ROOT $U/images/torch27.sif $U/venvs/cpu/bin/python $U/exp/RQF/code/sm_v1/sm_read.py; rc=$?; fi
echo "$(date -u +%FT%TZ) exit $rc"; exit $rc
