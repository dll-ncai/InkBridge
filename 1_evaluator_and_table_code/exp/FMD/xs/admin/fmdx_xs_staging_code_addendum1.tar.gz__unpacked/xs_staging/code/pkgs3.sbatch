#!/bin/bash
#SBATCH -J xs_pkgs3
#SBATCH -p cpuidle
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:40:00
#SBATCH -o $U/exp/FMD/xs/logs/pkgs3_%j.out
R=$U
P="apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python -m pip install --no-deps"
$P --target $R/exp/FMD/xs/pylibs matplotlib kiwisolver cycler pyparsing contourpy fonttools python-dateutil 2>&1 | tail -2
$P --target $R/exp/FMD/xs/pylibs_ds "transformers==4.46.3" "tokenizers==0.20.3" "huggingface_hub==0.26.5" 2>&1 | tail -2
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python -m pip freeze --path $R/exp/FMD/xs/pylibs > $R/exp/FMD/xs/pylibs_freeze.txt
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python -m pip freeze --path $R/exp/FMD/xs/pylibs_ds > $R/exp/FMD/xs/pylibs_ds_freeze.txt
cat $R/exp/FMD/xs/pylibs_freeze.txt $R/exp/FMD/xs/pylibs_ds_freeze.txt
