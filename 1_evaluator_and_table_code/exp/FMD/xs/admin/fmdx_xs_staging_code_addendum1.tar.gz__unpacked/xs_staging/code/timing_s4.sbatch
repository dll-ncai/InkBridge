#!/bin/bash
#SBATCH -J xs_time_s4
#SBATCH -p gpuidle
#SBATCH --gres=gpu:a4000:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH --exclude=dgx[001-005],gpu[004-005,013-014,017-023,119-148]
#SBATCH -o $U/exp/FMD/xs/logs/timing_s4_%j.out
R=$U
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
apptainer exec --nv --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/xs_probe.py time a4000 qari-ocr-v0.3-vl-2b qari-ocr-0.4.0-vl-4b-vision arabic-english-hw-ocr-v3 qwen3-vl-4b-instruct qwen2.5-vl-3b-instruct florence-2-base
export XS_DS=1 HF_HOME=$U/exp/FMD/xs/weights/.hf_home
apptainer exec --nv --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/xs_probe.py time a4000 theia-tiny-cdiv theia-tiny-cddsv theia-small-cdiv theia-base-cdiv theia-base-cddsv florence-2-large-ft
