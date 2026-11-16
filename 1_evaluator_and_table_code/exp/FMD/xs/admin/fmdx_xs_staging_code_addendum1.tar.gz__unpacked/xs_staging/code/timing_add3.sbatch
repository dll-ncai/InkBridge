#!/bin/bash
#SBATCH -J xs_time_add3
#SBATCH -p gpuidle
#SBATCH --gres=gpu:a4000:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH --requeue
#SBATCH --exclude=dgx[001-005],gpu[004-005,013-014,017-023,119-148]
#SBATCH -o $U/exp/FMD/xs/logs/timing_add3_%j.out
R=$U
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
apptainer exec --nv --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/xs_probe.py time a4000 webssl-mae1b-224 webssl-mae3b-224 aimv2-huge-patch14-224 aimv2-1b-patch14-224 aimv2-3b-patch14-224 aimv2-large-patch14-224-distilled dinov2-reg-small dinov2-reg-base dinov2-reg-giant dinov2-giant dinov2-base qari-ocr-v0.3-vl-2b qari-ocr-0.4.0-vl-4b-vision arabic-english-hw-ocr-v3 qwen3-vl-4b-instruct qwen2.5-vl-3b-instruct
export XS_DS=1 HF_HOME=$U/exp/FMD/xs/weights/.hf_home
apptainer exec --nv --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/xs_probe.py time a4000 theia-tiny-cdiv theia-tiny-cddsv theia-small-cdiv theia-base-cdiv theia-base-cddsv
