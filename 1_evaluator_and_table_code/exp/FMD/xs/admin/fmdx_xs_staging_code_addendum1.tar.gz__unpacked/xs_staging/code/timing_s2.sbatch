#!/bin/bash
#SBATCH -J xs_time_s2
#SBATCH -p gpuidle
#SBATCH --gres=gpu:a4000:1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH --exclude=dgx[001-005],gpu[004-005,013-014,017-023,119-148]
#SBATCH -o $U/exp/FMD/xs/logs/timing_s2_%j.out
R=$U
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
apptainer exec --nv --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/xs_probe.py time a4000 siglip2-large-patch16-256 siglip2-so400m-patch16-256 dinov3-vitl16-sat493m surya-ocr-2-persian-hw chandra webssl-dino3b-224 webssl-dino5b-224 webssl-dino7b-224
