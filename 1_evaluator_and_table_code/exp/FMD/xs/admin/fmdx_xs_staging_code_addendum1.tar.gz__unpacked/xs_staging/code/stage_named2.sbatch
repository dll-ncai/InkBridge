#!/bin/bash
#SBATCH -J xs_stage_named2
#SBATCH -p cpuidle
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 24:00:00
#SBATCH --requeue
#SBATCH -o $U/exp/FMD/xs/logs/stage_named2_%j.out
R=$U
apptainer exec --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/stage_xs_v3.py add_named pe-core-g14-448-timm pe-lang-l14-448-timm theia-tiny-cddsv theia-tiny-cdiv theia-small-cdiv theia-base-cdiv theia-base-cddsv pe-core-s16-384 pe-core-t16-384 sd3.5-medium-diffusers-layout sd3.5-large-diffusers-layout sd3.5-large-turbo-diffusers-layout sam3.1 pe-lang-g14-448-timm
