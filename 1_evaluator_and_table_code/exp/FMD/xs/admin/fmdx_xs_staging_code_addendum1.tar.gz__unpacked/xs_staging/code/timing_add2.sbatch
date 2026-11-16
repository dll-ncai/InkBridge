#!/bin/bash
#SBATCH -J xs_time_add2
#SBATCH -p gpuidle
#SBATCH --gres=gpu:a4000:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH --requeue
#SBATCH --exclude=dgx[001-005],gpu[004-005,013-014,017-023,119-148]
#SBATCH -o $U/exp/FMD/xs/logs/timing_add2_%j.out
R=$U
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
apptainer exec --nv --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/xs_probe.py time a4000 sam3 sd3-medium-diffusers pe-core-g14-448 pe-lang-l14-448 pe-lang-g14-448 tipsv1-g14 siglip2-large-patch16-256 siglip2-so400m-patch16-256 dinov3-vitl16-sat493m surya-ocr-2-persian-hw chandra webssl-dino3b-224 webssl-dino5b-224 webssl-dino7b-224
