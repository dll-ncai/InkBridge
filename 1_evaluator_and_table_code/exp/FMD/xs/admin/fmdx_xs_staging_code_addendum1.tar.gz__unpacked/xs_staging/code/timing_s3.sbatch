#!/bin/bash
#SBATCH -J xs_time_s3
#SBATCH -p gpuidle
#SBATCH --gres=gpu:a4000:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH --exclude=dgx[001-005],gpu[004-005,013-014,017-023,119-148]
#SBATCH -o $U/exp/FMD/xs/logs/timing_s3_%j.out
R=$U
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
apptainer exec --nv --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/xs_probe.py time a4000 webssl-mae1b-224 webssl-mae3b-224 aimv2-huge-patch14-224 aimv2-1b-patch14-224 aimv2-3b-patch14-224 aimv2-large-patch14-224-distilled dinov2-reg-small dinov2-reg-base dinov2-reg-giant dinov2-giant dinov2-base
