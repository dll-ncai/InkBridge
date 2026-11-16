#!/bin/bash
#SBATCH -p cpuidle
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH -J rv11_arch_s2b
#SBATCH -o $U/tmp/rv11_archive/stage2b_%j.out
R=$U
echo "$(date -u +%FT%TZ) start"; apptainer exec -B $SCRATCH_ROOT $R/images/torch27.sif $R/venvs/cpu/bin/python $R/tmp/rv11_archive/stage2b.py; echo "$(date -u +%FT%TZ) exit $?"
