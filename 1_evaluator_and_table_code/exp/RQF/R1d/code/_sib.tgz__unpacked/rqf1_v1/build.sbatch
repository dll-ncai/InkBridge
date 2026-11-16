#!/bin/bash
#SBATCH --job-name=rqf1_build
#SBATCH --partition=epyc-256
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=$U/exp/RQF/R1/logs/%x_%j.out
set -uo pipefail; exec 2>&1
B=$U; FC=$B/exp/CAND/code/frame_code_a44e78b0; Q=$B/exp/RQF/code/rqf1_v1
echo "start $(date -u +%FT%TZ) host $(hostname) job $SLURM_JOB_ID"
apptainer exec -B $SCRATCH_ROOT -B /tmp $B/images/torch27.sif env PYTHONPATH=$FC/lib:$FC:$B/repos/nagamatsu/src OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 $B/venvs/e0/bin/python $Q/build_mirrored.py --stage $B --out $B/exp/RQF/R1/data
echo "end rc=$? $(date -u +%FT%TZ)"
