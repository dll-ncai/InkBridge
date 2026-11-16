#!/bin/bash
#SBATCH -J xs_pkgs2
#SBATCH -p cpuidle
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o $U/exp/FMD/xs/logs/pkgs2_%j.out
R=$U
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python -m pip install --no-deps --target $R/exp/FMD/xs/pylibs hydra-core omegaconf jaxtyping wadler_lindig antlr4-python3-runtime==4.9.3 2>&1 | tail -3
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python -m pip freeze --path $R/exp/FMD/xs/pylibs > $R/exp/FMD/xs/pylibs_freeze.txt; cat $R/exp/FMD/xs/pylibs_freeze.txt
