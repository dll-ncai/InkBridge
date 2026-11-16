#!/bin/bash
#SBATCH -J rqf_pairext
#SBATCH -p epyc-256
#SBATCH -N 1 --cpus-per-task=4 --mem=48G --time=02:00:00
#SBATCH -o $U/exp/RQF/logs/rqf_pairext_%j.out
exec 2>&1
U=$U
apptainer exec -B $SCRATCH_ROOT $U/images/torch27.sif bash -c "source $U/venvs/cpu/bin/activate && python $U/exp/RQF/code/rqf_paired_ext.py $U/exp/RQF/tables/paired_ext"
